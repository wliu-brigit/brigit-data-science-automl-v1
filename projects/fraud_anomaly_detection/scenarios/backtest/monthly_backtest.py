"""Month-over-month scenario backtest — stripped, combined query.

Runs the registered fraud scenario predicates against the full warehouse
history, bucketed by advance month, and reports one row per (month x
scenario): the caught volume (with a denominator to normalize for changing
advance volume) and the dpd45 quality of the catch.

This is a backtest / ops tool, NOT harness-wired modelling. It runs outside
the AutoML loop, the same status as the upstream feature DDL
(data/queries/upstream_fraud_advance_feature_base.sql): kept in the repo so
the analysis is reproducible, executed by hand against the warehouse (needs
VPN). Design: docs/execution_parallel/month-over-month-backtest/README.md.

Why this is cheap: the upstream feature snapshot is large because it computes
the full feature set. The scenario predicates need only the registered ring
velocity fields, so this query computes just those instead of the full feature
base.

Run:
    uv run python -m projects.fraud_anomaly_detection.scenarios.backtest.monthly_backtest
"""

from __future__ import annotations

from pathlib import Path

from projects.fraud_anomaly_detection.scenarios import SCENARIOS_VERSION

# ─────────────────────────────────────────────────────────────────────────
# PARAMETERS — edit these, then run. The SQL below is assembled from them.
# ─────────────────────────────────────────────────────────────────────────

OUTPUT_START = "2025-01-01"  # first anchor month (inclusive)
OUTPUT_END = "2026-07-01"  # exclusive upper bound (~ now: captures through Jun 2026)
HISTORY_BUFFER_DAYS = 8  # lookback before OUTPUT_START for velocity (only the 7d window is read; +1 slack)

# Fast end-to-end smoke test: when set, OVERRIDES the window above with a short
# span so you can validate syntax / output shape / connection in seconds before
# committing to the full multi-month pull. Format: ("start", "end_exclusive");
# a single day is plenty (use a sub-day span like ("2025-12-01 00:00:00",
# "2025-12-01 01:00:00") to go faster still). Set to None for the full run.
# NOTE: a short window makes the velocity counts incomplete (the lookback is
# clipped) — it proves the query runs, not the numbers. Use the full run for those.
TEST_WINDOW: tuple[str, str] | None = ("2025-12-01", "2025-12-02")

# True  -> execute against Snowflake (needs VPN) and return/print a DataFrame.
# False -> print the rendered SQL only, for pasting into the Snowflake console.
EXECUTE = True

# Source tables (warehouse) — same as the upstream feature DDL.
FCT_LOANS = "brigit_snowflake.dbt_analytics.fct_loans"
IDENTITIES = "brigit_snowflake.dbt_analytics.base_prod__identities"
PLAID_ACCOUNTS = "brigit_snowflake.dbt_analytics.base_prod__plaid_accounts"
USER_CLIENT_METADATA = "pc_fivetran_db.wal_brigit_production_public.user_client_metadata"

# ─────────────────────────────────────────────────────────────────────────
# SCENARIOS — the codified predicates as SQL, mirroring register.yaml.
# These are hand-written to mirror the register (the engine compiles the
# register to pandas, not SQL); they are NOT auto-generated. When the register
# changes, update these and bump REGISTER_VERSION below.
# Each entry: (scenario_name, SQL boolean expression over the `flagged` rows).
# ─────────────────────────────────────────────────────────────────────────

REGISTER_VERSION = SCENARIOS_VERSION

SCENARIOS: list[tuple[str, str]] = [
    # ring_account_reuse: fresh identity (<=24h) + sizable advance + the account
    # already has a recent advance (someone drew through it within 7d).
    (
        "ring_account_reuse",
        "DATEDIFF('second', identity_created_time, feature_as_of_ts) / 3600.0 <= 24"
        " AND loan_amount > 100"
        " AND prior_advances_on_bank_account_7d > 0",
    ),
    # ring_identity_burst: >= 3 identities created within 72h on one bank account.
    (
        "ring_identity_burst",
        "users_on_bank_account_72h >= 3",
    ),
    (
        "ring_shared_persistent_account",
        "users_on_persistent_account_id_72h >= 2"
        " AND COALESCE(is_joint, 0) != 1",
    ),
    (
        "ring_device_burst",
        "users_on_device_id_72h >= 3",
    ),
]


def _resolve_window() -> tuple[str, str]:
    """(output_start, output_end_exclusive) as TIMESTAMP_NTZ literals."""
    start, end = TEST_WINDOW if TEST_WINDOW else (OUTPUT_START, OUTPUT_END)
    return f"'{start}'::TIMESTAMP_NTZ", f"'{end}'::TIMESTAMP_NTZ"


def build_sql() -> str:
    output_start_expr, output_end = _resolve_window()

    # One boolean flag column per scenario in the `flagged` CTE, plus the union.
    flag_cols = ",\n".join(
        f"        ({expr}) AS match_{name}" for name, expr in SCENARIOS
    )
    any_expr = " OR ".join(f"({expr})" for _, expr in SCENARIOS)

    # One aggregate block per scenario (+ scenario_any), UNION ALL'd. Every block
    # is identical except the flag it counts, so adding a scenario above adds one
    # block here automatically. The whole-month figures (n_advances,
    # total_loan_disbursed, baseline_*) are computed inline per block — they are
    # the same denominator/baseline across scenarios within a month.
    #
    # Column meanings (one row = one month x one scenario):
    #   n_advances            advances disbursed that month (the denominator)
    #   n_scenario            advances this scenario flagged
    #   scenario_rate         n_scenario / n_advances
    #   total_loan_disbursed  $ disbursed across ALL advances that month
    #   scenario_loan_disbursed   $ disbursed across the flagged advances
    #   n_matured             flagged advances old enough to observe DPD45 (>=45d)
    #   n_dpd45               flagged + matured + hit gross DPD45
    #   dpd45_rate            n_dpd45 / n_matured   (the bust-out cut; matured only)
    #   baseline_dpd45_rate   DPD45 rate over ALL matured advances that month
    #   scenario_never_paid_rate  RESOLVED bad-rate among flagged advances:
    #                         never_paid / (repaid + never_paid), where
    #                         never_paid = matured AND DPD45 AND not repaid.
    #                         (= 1 - repaid_rate; kept in this "lower is better"
    #                         direction to match dpd45_rate.) Of the advances
    #                         that reached a verdict (paid back, or went DPD45
    #                         unpaid), the share that went bad; still-open
    #                         advances excluded. Charge-off is NOT used as the
    #                         loss leg: it is ~never populated here (399 of 10.7M
    #                         rows), so the bad outcome is delinquency. Fraud ->
    #                         ~1. A month with no matured rows reads 0 (no bad
    #                         outcome observable yet) -- read with n_matured.
    #   baseline_never_paid_rate  same resolved rate over ALL advances (contrast)
    #   scenario_never_paid_principal  PRINCIPAL ($ loan_amount) disbursed to
    #                         flagged advances that NEVER paid (matured AND DPD45
    #                         AND not repaid) -- money out the door we likely
    #                         won't recover. Named "principal" on purpose: it is
    #                         the disbursed loan_amount, NOT a net-of-payments
    #                         balance (a gross-unpaid-balance field exists but is
    #                         not yet validated).
    #   baseline_never_paid_principal  same over ALL advances that month (contrast)
    names = [name for name, _ in SCENARIOS] + ["scenario_any"]
    blocks = []
    for name in names:
        flag = f"match_{name}"
        blocks.append(
            f"""    SELECT
        advance_month,
        '{name}' AS scenario,
        COUNT(*) AS n_advances,
        COUNT_IF({flag}) AS n_scenario,
        COUNT_IF({flag}) / NULLIF(COUNT(*), 0) AS scenario_rate,
        SUM(loan_amount) AS total_loan_disbursed,
        SUM(IFF({flag}, loan_amount, 0)) AS scenario_loan_disbursed,
        COUNT_IF({flag} AND is_matured) AS n_matured,
        COUNT_IF({flag} AND is_matured AND is_dpd45) AS n_dpd45,
        COUNT_IF({flag} AND is_matured AND is_dpd45)
            / NULLIF(COUNT_IF({flag} AND is_matured), 0) AS dpd45_rate,
        COUNT_IF(is_matured AND is_dpd45)
            / NULLIF(COUNT_IF(is_matured), 0) AS baseline_dpd45_rate,
        COUNT_IF({flag} AND is_matured AND is_dpd45 AND NOT is_repaid)
            / NULLIF(COUNT_IF({flag} AND (is_repaid OR (is_matured AND is_dpd45))), 0) AS scenario_never_paid_rate,
        COUNT_IF(is_matured AND is_dpd45 AND NOT is_repaid)
            / NULLIF(COUNT_IF(is_repaid OR (is_matured AND is_dpd45)), 0) AS baseline_never_paid_rate,
        SUM(IFF({flag} AND is_matured AND is_dpd45 AND NOT is_repaid, loan_amount, 0)) AS scenario_never_paid_principal,
        SUM(IFF(is_matured AND is_dpd45 AND NOT is_repaid, loan_amount, 0)) AS baseline_never_paid_principal
    FROM flagged
    GROUP BY advance_month"""
        )
    aggregate = "\n    UNION ALL\n".join(blocks)

    return f"""
-- Month-over-month scenario backtest (register v{REGISTER_VERSION}).
-- Generated by scenarios/backtest/monthly_backtest.py — do not hand-edit; edit
-- the params/SCENARIOS in that module and re-render.
WITH params AS (
    SELECT
        {output_start_expr} AS output_start_ts,
        {output_end} AS output_end_ts,
        DATEADD('day', -{HISTORY_BUFFER_DAYS}, {output_start_expr}) AS history_start_ts
),

/* 1. Advances for historical context (lookback for prior-advance velocity).
      Stripped to identifiers, amount, the transaction timestamp, and ONLY the
      dpd45 outcome fields. */
all_advances AS (
    SELECT
        l.id::VARCHAR AS advance_id,
        l.user_id::VARCHAR AS user_id,
        l.plaid_routing_number,
        l.plaid_account_number,
        l.loan_amount,
        l.origination_timestamp::TIMESTAMP_NTZ AS feature_as_of_ts,
        IFF(l.loan_status = 'REPAID', 1, 0) AS label_repaid_current_snapshot,
        IFF(l.is_gross_dpd45, 1, 0) AS label_gross_dpd45,
        IFF(l.is_mature_d45, 1, 0) AS label_mature_d45
    FROM {FCT_LOANS} l
    CROSS JOIN params p
    WHERE l.origination_timestamp >= p.history_start_ts
      AND l.origination_timestamp <  p.output_end_ts
),

/* 2. Anchor advances: the output population, bucketed by month. */
anchor_advances AS (
    SELECT a.*
    FROM all_advances a
    CROSS JOIN params p
    WHERE a.feature_as_of_ts >= p.output_start_ts
      AND a.feature_as_of_ts <  p.output_end_ts
),

/* 3. Entity scoping — the cost lever. We only ever report on anchor advances
      and the accounts they touch, so scope the link build to the bank accounts
      the advance-window users touch. Two passes:
        a. relevant_account_keys — accounts touched by any advance-window user.
        b. scoped_plaid           — EVERY plaid row on those accounts (so
                                     account-sharing siblings are still counted).
      The dedup to current state happens in bank_account_links (5), over this
      scoped subset — deliberately NOT over the full ~1B-row CDC view, which a
      full-view window sort would make slower (see results/OPTIMIZATION_LOG.md
      iters 3-4). Numbers are identical to the unscoped build; only the scan
      shrinks. */
relevant_account_keys AS (
    SELECT DISTINCT pa.routing_number, pa.account_number
    FROM {PLAID_ACCOUNTS} pa
    JOIN (SELECT DISTINCT user_id FROM all_advances) au
        ON pa.user_id::VARCHAR = au.user_id
    WHERE pa.routing_number IS NOT NULL
      AND pa.account_number IS NOT NULL
),
scoped_plaid AS (
    SELECT pa.*
    FROM {PLAID_ACCOUNTS} pa
    JOIN relevant_account_keys k
        ON pa.routing_number = k.routing_number
       AND pa.account_number = k.account_number
),

/* 4. One identity row per user — scoped to users on the relevant accounts. */
identities_one_per_user AS (
    SELECT
        i.user_id::VARCHAR AS user_id,
        i.created_time::TIMESTAMP_NTZ AS identity_created_time
    FROM {IDENTITIES} i
    JOIN (SELECT DISTINCT user_id::VARCHAR AS user_id FROM scoped_plaid) u
        ON i.user_id::VARCHAR = u.user_id
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY i.user_id
        ORDER BY i.is_deleted ASC, i.created_time DESC
    ) = 1
),

/* 5. User -> bank-account links (canonical key), deduped to current state over
      the SCOPED plaid rows (one row per user/routing/account, latest by
      incrementing_id). Holds every user on the relevant accounts — shared-
      account counts need them, including users with no advance of their own. */
bank_account_links AS (
    SELECT
        pa.user_id::VARCHAR AS user_id,
        pa.routing_number,
        pa.account_number,
        CONCAT(pa.routing_number, '-', pa.account_number) AS bank_account_key,
        MAX(pa.persistent_account_id) OVER (
            PARTITION BY pa.user_id, pa.routing_number, pa.account_number
        ) AS persistent_account_id,
        pa.created_at::TIMESTAMP_NTZ AS plaid_account_created_at,
        pa.is_joint,
        i.identity_created_time
    FROM scoped_plaid pa
    JOIN identities_one_per_user i
        ON pa.user_id::VARCHAR = i.user_id
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY pa.user_id, pa.routing_number, pa.account_number
        ORDER BY pa.incrementing_id DESC
    ) = 1
),

/* 5. Anchor advances mapped to candidate bank accounts (may fan out; deduped
      to one row per advance at the end). Exact routing/account match from
      fct_loans when populated, else the user's accounts as of origination. */
anchor_advance_account_candidates AS (
    SELECT
        a.*,
        ba.routing_number,
        ba.account_number,
        ba.bank_account_key,
        ba.persistent_account_id,
        ba.plaid_account_created_at,
        ba.is_joint,
        ba.identity_created_time
    FROM anchor_advances a
    JOIN bank_account_links ba
        ON a.user_id = ba.user_id
       AND ba.plaid_account_created_at <= a.feature_as_of_ts
       AND (
            (a.plaid_routing_number IS NOT NULL
             AND a.plaid_account_number IS NOT NULL
             AND a.plaid_routing_number = ba.routing_number
             AND a.plaid_account_number = ba.account_number)
            OR (a.plaid_routing_number IS NULL AND a.plaid_account_number IS NULL)
       )
),

/* The accounts our anchor advances actually touch — the only accounts prior
   velocity is ever asked about. Scoping the prior set to these turns the
   inequality (range) join from "all history x all accounts" into a small one. */
anchor_account_keys AS (
    SELECT DISTINCT bank_account_key FROM anchor_advance_account_candidates
),

/* 7. Historical advances on ANCHOR accounts — for prior-advance velocity. */
all_advance_account_candidates AS (
    SELECT
        a.advance_id,
        a.feature_as_of_ts,
        ba.bank_account_key
    FROM all_advances a
    JOIN bank_account_links ba
        ON a.user_id = ba.user_id
       AND ba.plaid_account_created_at <= a.feature_as_of_ts
       AND (
            (a.plaid_routing_number IS NOT NULL
             AND a.plaid_account_number IS NOT NULL
             AND a.plaid_routing_number = ba.routing_number
             AND a.plaid_account_number = ba.account_number)
            OR (a.plaid_routing_number IS NULL AND a.plaid_account_number IS NULL)
       )
    JOIN anchor_account_keys ak
        ON ba.bank_account_key = ak.bank_account_key
),

/* 7. Identity-burst velocity (per advance x candidate account): distinct users
      whose identity was created in the 72h before the advance. The 72h bound is
      pushed INTO the join, so the identity self-join only touches a 72h band
      instead of the account's whole history — no lifetime count is computed. */
bank_account_user_features AS (
    SELECT
        a.advance_id,
        a.bank_account_key,
        COUNT(DISTINCT other.user_id) AS users_on_bank_account_72h
    FROM anchor_advance_account_candidates a
    LEFT JOIN bank_account_links other
        ON a.routing_number = other.routing_number
       AND a.account_number = other.account_number
       AND other.plaid_account_created_at <= a.feature_as_of_ts
       AND other.identity_created_time   <= a.feature_as_of_ts
       AND other.identity_created_time   >= DATEADD(hour, -72, a.feature_as_of_ts)
    GROUP BY 1, 2
),

/* 8. Prior-advance velocity (per advance x candidate account): distinct prior
      advances on the account within 7d. The 7d bound is pushed INTO the join,
      so the range join is a tight 7-day band, not all-history. */
bank_account_advance_features AS (
    SELECT
        a.advance_id,
        a.bank_account_key,
        COUNT(DISTINCT prior.advance_id) AS prior_advances_on_bank_account_7d
    FROM anchor_advance_account_candidates a
    LEFT JOIN all_advance_account_candidates prior
        ON a.bank_account_key = prior.bank_account_key
       AND prior.feature_as_of_ts <  a.feature_as_of_ts
       AND prior.feature_as_of_ts >= DATEADD(day, -7, a.feature_as_of_ts)
    GROUP BY 1, 2
),

/* 9. Persistent-account identity burst, keyed on Plaid's stable account id. */
persistent_account_user_features AS (
    SELECT
        a.advance_id,
        a.bank_account_key,
        COUNT(DISTINCT other.user_id) AS users_on_persistent_account_id_72h
    FROM anchor_advance_account_candidates a
    LEFT JOIN bank_account_links other
        ON a.persistent_account_id = other.persistent_account_id
       AND a.persistent_account_id IS NOT NULL
       AND other.plaid_account_created_at <= a.feature_as_of_ts
       AND other.identity_created_time <= a.feature_as_of_ts
       AND other.identity_created_time >= DATEADD(hour, -72, a.feature_as_of_ts)
    GROUP BY 1, 2
),

/* 10. Device identity burst, using the latest device valid at the advance time. */
client_metadata AS (
    SELECT
        cm.identifying_id::VARCHAR AS user_id,
        cm.valid_from::TIMESTAMP_NTZ AS valid_from,
        cm.device_id
    FROM {USER_CLIENT_METADATA} cm
    WHERE cm._fivetran_deleted = FALSE
      AND cm.device_id IS NOT NULL
),
device_links AS (
    SELECT
        cm.user_id,
        cm.device_id,
        MIN(cm.valid_from) AS device_valid_from,
        MAX(i.identity_created_time) AS identity_created_time
    FROM client_metadata cm
    JOIN identities_one_per_user i
        ON cm.user_id = i.user_id
    GROUP BY cm.user_id, cm.device_id
),
anchor_device AS (
    SELECT advance_id, user_id, feature_as_of_ts, device_id
    FROM (
        SELECT
            a.advance_id,
            a.user_id,
            a.feature_as_of_ts,
            cm.device_id,
            ROW_NUMBER() OVER (
                PARTITION BY a.advance_id
                ORDER BY cm.valid_from DESC NULLS LAST
            ) AS rn
        FROM anchor_advances a
        LEFT JOIN client_metadata cm
            ON a.user_id = cm.user_id
           AND cm.valid_from <= a.feature_as_of_ts
    )
    WHERE rn = 1
),
device_user_features AS (
    SELECT
        a.advance_id,
        COUNT(DISTINCT other.user_id) AS users_on_device_id_72h
    FROM anchor_device a
    JOIN device_links other
        ON a.device_id = other.device_id
       AND other.device_valid_from <= a.feature_as_of_ts
       AND other.identity_created_time <= a.feature_as_of_ts
       AND other.identity_created_time >= DATEADD(hour, -72, a.feature_as_of_ts)
    WHERE a.device_id IS NOT NULL
    GROUP BY 1
),

/* 11. One row per advance: join the windowed features, then dedup candidate
      accounts on those windowed features (cheap deterministic tiebreak — NOT
      exact upstream-snapshot parity; the lifetime/30d tiebreakers it used are
      deliberately not computed). Differs from the snapshot only on the rare
      advance that maps to multiple candidate accounts. */
advance_level AS (
    SELECT
        a.advance_id,
        DATE_TRUNC('month', a.feature_as_of_ts) AS advance_month,
        a.feature_as_of_ts,
        a.loan_amount,
        a.identity_created_time,
        COALESCE(a.is_joint, 0) AS is_joint,
        baf.users_on_bank_account_72h,
        aaf.prior_advances_on_bank_account_7d,
        COALESCE(pauf.users_on_persistent_account_id_72h, 0)
            AS users_on_persistent_account_id_72h,
        COALESCE(duf.users_on_device_id_72h, 0) AS users_on_device_id_72h,
        (a.label_mature_d45 = 1) AS is_matured,
        (a.label_gross_dpd45 = 1) AS is_dpd45,
        (a.label_repaid_current_snapshot = 1) AS is_repaid
    FROM anchor_advance_account_candidates a
    LEFT JOIN bank_account_user_features baf
        ON a.advance_id = baf.advance_id AND a.bank_account_key = baf.bank_account_key
    LEFT JOIN bank_account_advance_features aaf
        ON a.advance_id = aaf.advance_id AND a.bank_account_key = aaf.bank_account_key
    LEFT JOIN persistent_account_user_features pauf
        ON a.advance_id = pauf.advance_id AND a.bank_account_key = pauf.bank_account_key
    LEFT JOIN device_user_features duf
        ON a.advance_id = duf.advance_id
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY a.advance_id
        ORDER BY
            duf.users_on_device_id_72h DESC NULLS LAST,
            pauf.users_on_persistent_account_id_72h DESC NULLS LAST,
            baf.users_on_bank_account_72h DESC NULLS LAST,
            aaf.prior_advances_on_bank_account_7d DESC NULLS LAST,
            a.plaid_account_created_at DESC NULLS LAST
    ) = 1
),

/* 12. Scenario flags (one boolean per scenario + the union). */
flagged AS (
    SELECT
        advance_month,
        loan_amount,
        is_matured,
        is_dpd45,
        is_repaid,
{flag_cols},
        ({any_expr}) AS match_scenario_any
    FROM advance_level
)

/* 11. Long format: one row per (month x scenario). */
SELECT * FROM (
{aggregate}
)
ORDER BY advance_month, scenario
"""


def run() -> "tuple[object, str]":
    """Run the main query, returning (df, query_id). One connection so we can
    read cur.sfqid for targeted profiling (profile.profile_operators(qid))."""
    from automl.utils.io import snowflake

    with snowflake.connect() as conn:
        cur = conn.cursor()
        try:
            cur.execute(build_sql())
            df = snowflake.coerce_decimal_columns(cur.fetch_pandas_all())
            return df, cur.sfqid
        finally:
            cur.close()


def _save(df) -> "Path":
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    start, end = (TEST_WINDOW if TEST_WINDOW else (OUTPUT_START, OUTPUT_END))
    tag = "test_" if TEST_WINDOW else ""
    label = f"{tag}{start}_{end}".replace(" ", "T").replace(":", "").replace("-", "")
    out_path = out_dir / f"backtest_{label}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def main() -> int:
    if not EXECUTE:
        print(build_sql())
        return 0

    import pandas as pd
    from dotenv import load_dotenv

    from automl.project.config import find_repo_root

    # Standalone entry point: load the repo-root .env ourselves (the harness
    # normally does this during project-config load; we don't open a session).
    load_dotenv(find_repo_root() / ".env", override=False)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    df, qid = run()
    print(f"query_id: {qid}")
    print(df.to_string(index=False))
    print(f"\nsaved -> {_save(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
