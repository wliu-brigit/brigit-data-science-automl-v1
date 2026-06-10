# To-do: neobank_ncm VPN day — materialize, replicate, verify

**Status:** ready to execute, blocked only on warehouse access (wendao calls
the day). Paths are relative to `projects/neobank_ncm/`. Delete this file
when the day is done; results land in MLflow and the wrap handoff.

The one-time switch from the offline CSV fixture to the real warehouse, plus
every parity check against the legacy v3 run. Work top to bottom; each step
has the command and the expected number. Reference values come from the
legacy artifacts bundled in `data/legacy/` (`preprocessor_meta.json →
performance` is the official block) and the legacy notebook outputs.

Until this day, everything runs offline: `NEOBANK_NCM_CSV=<path>` (see
`config.py`) swaps in the synthetic fixture, and QA runs go to `qa/`
namespaces. Sweep them afterwards with `automl project delete --scope qa`.

## 0. Flip the source toggle

- **Unset `NEOBANK_NCM_CSV`** in the shell (the toggle in `config.py` then
  resolves to the real `SnowflakeSource`).
- Fill the empty Snowflake credentials in the repo-root `.env`:
  `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, plus
  `SNOWFLAKE_DATABASE=brigit_data_science`, `SNOWFLAKE_SCHEMA=sandbox_wliu`
  (WAREHOUSE/ROLE are already set).
- Preflight (probes `SELECT 1`, GCS, MLflow, SQL files):

  ```bash
  uv run automl --project neobank_ncm validate project
  ```

## 1. Warehouse checks (read-only, before materializing)

Against the pinned legacy tables in `sandbox_hyong`:

```sql
DESCRIBE TABLE brigit_data_science.sandbox_hyong.neobank_ncm_v3_spine;
DESCRIBE TABLE brigit_data_science.sandbox_hyong.neobank_ncm_v3_risk_features;
DESCRIBE TABLE brigit_data_science.sandbox_hyong.neobank_ncm_v3_synthetic_scores_final;
-- next-phase snapshots (downstream analyses, deferred): existence check only
DESCRIBE TABLE brigit_data_science.sandbox_hyong.neobank_ncm_v3_oot_new_links_daily;
DESCRIBE TABLE brigit_data_science.sandbox_hyong.neobank_ncm_v3_oot_new_links_daily_plaid_unnested;
DESCRIBE TABLE brigit_data_science.sandbox_hyong.neobank_ncm_v3_oot_new_links_ri_scores;
```

- risk_features still carries the 96 plaid/netflow columns the locked
  feature list resolves against.
- **Derived-name collisions**: if the refreshed SA spec already ships any
  column that `data/queries/base_table.sql` aliases (the 11 derived
  features, e.g. `balancesdtodailyincomemeanratio`,
  `incomebuffertodaystopaydayratio`, `istaxseason`), the CREATE fails on a
  duplicate name — drop the redundant expression from the SQL then.
- Spine population vs the legacy run:

  | predicate | expected |
  |---|---|
  | `is_known AND split = 'train'` (full 2025) | **282,642** |
  | `NOT is_known AND split = 'train'` | ~600K+ (record the exact count) |
  | `is_known AND split = 'oot'` (Jan–Feb 2026) | **66,507** |
  | `NOT is_known AND split = 'oot'` | **117,090** |

## 2. Materialize the snapshot

```bash
uv run automl --project neobank_ncm data materialize --refresh-source
```

Creates `neobank_ncm_v3_replicate_base` under `sandbox_wliu` (legacy sources
are never written), injects `SPLIT_PCT` and `unknown_train_hash_rank`,
snapshots to GCS, registers the dataset in MLflow. Post-checks:

- `unknown_train_hash_rank` is non-NULL exactly on `NOT is_known AND
  split = 'train'` rows, ranks 1..N contiguous.
- Scored legacy pool: `rank <= 200000 AND synthetic_score IS NOT NULL`
  = **146,088** (the legacy 200K pull had 92% score coverage).

## 3. Baseline replication trial (the in-loop checkpoint)

```bash
uv run python -c "
from automl.project import use_project
from automl.runner import run_trial
use_project('neobank_ncm')
print(run_trial('neobank_ncm').status)
"
```

(`run_trial` with the bare project name dispatches the project
`MODEL_CLASS` baseline — same path `scripts/qa_local_run.py` exercises
offline; `automl trial run <path>` is for coded trials.)

Trains on `train` (Jan–Oct 2025), scores on `test` (Nov–Dec 2025,
known-only). Expected split shapes (legacy experiment notebook, cell 9):

| split | expected |
|---|---|
| train known | **202,593** (bad rate 0.251) |
| train unknown in rank ≤ 200K | **159,694**, of which scored **116,220** |
| ratio draw (80/20 of 202,593) | **50,648** unknowns → dual records |
| test known | **80,049** (bad rate 0.281) |

**Decision metric**: test AUC vs legacy constrained **0.7002**
(`test_auc_constrained`; the unconstrained variant was 0.7016 — that's the
loop's stretch reference, not the baseline checkpoint). Expect close, not
bit-exact: xgboost 3.2.0 vs the legacy env, model-matrix column order, and
the draw picking different rows within the same deterministic pool.

## 4. WoE parity

Fit fresh on full-2025 known rows and diff against the bundled legacy
mapping — same data + same code should match to float precision
(IV 0.0244, OTHER = CHIME ≈ -0.0011):

```python
from automl.data import materialize
from projects.neobank_ncm.model.preprocessing import (
    BankInstitutionWOEEncoder)

df = materialize().df
known = df[(df["is_known"]) & (df["split"] == "train")]  # full 2025
fresh = BankInstitutionWOEEncoder().fit(known["bankinstitution"], known["went_dpd45"])
legacy = BankInstitutionWOEEncoder.from_legacy_mapping()
for bank in sorted(set(fresh.mapping_) | set(legacy.mapping_)):
    a, b = fresh.mapping_.get(bank), legacy.mapping_.get(bank)
    if a is None or b is None or abs(a - b) > 1e-9:
        print(f"DIFF {bank}: fresh={a} legacy={b}")
```

## 5. Phase 4–5: final retrain + the one OOT read

After the AutoML loop picks a winner: retrain it on **full 2025**
(train + test windows — known 282,642; draw from the 146,088 scored pool =
**70,660** unknowns → **423,962** weighted rows), then evaluate `oot`
**once**:

```bash
uv run python projects/neobank_ncm/scripts/evaluate_split.py \
    --model-run-id <winning trial's MLflow run id> --split oot
```

vs legacy `performance` block: OOT known-only AUC **0.6932**
(gini 0.3863, KS 0.2859). The legacy train known-only diagnostic 0.7896 is
full-2025 in-sample — comparable only after the full-2025 retrain, not
against the Jan–Oct loop model (`--split train_known` evaluates Jan–Oct).

## 6. Wrap

- Kick off the real AutoML loop (`automl --project neobank_ncm experiment
  run ...`) against the materialized snapshot.
- Sweep QA namespaces, commit the checkpoint, rewrite `docs/HANDOFF.md`.

## 7. Phase 2 — downstream analysis (any VPN visit; full run after the loop)

The legacy post-training analyses, ported offline-tested in `analysis/`
(+ `notebooks/financial_impact_analysis.ipynb`). The three new-links
snapshots are frozen; `user_ltv.sql` is the one live pull.

1. **Parity eval of the daily snapshot** (legacy artifacts = production v3):

   ```bash
   uv run python projects/neobank_ncm/scripts/evaluate_new_links_daily.py \
       --legacy-artifacts <data-science repo>/models/underwriting/neobank/new_user/v3.0/artifacts
   ```

   Expected (legacy printed values): D2 known-only AUC ≈ **0.6935**;
   RI parity vs `synthetic_scores_final` — overlap **91,677**, Pearson
   **0.9999**, exact percentile agreement **99%**; calibration deciles
   tracking the diagonal. (RI scores table: **139,916** rows of the
   **177,528** OOT population = 78.8% coverage.)
2. **Financial impact notebook, parity mode** (`MODEL_MODE='parity'`):
   reproduces the legacy benchmark/threshold tables, revenue decomposition
   and months-needed table within float noise.
3. **Switch to the winner** (`MODEL_MODE='trial'`, `MODEL_RUN_ID=<winner>`;
   or `--model-run-id` on the script) — the real analysis. Cache the pulls
   to parquet for offline re-runs (`DAILY_PARQUET`/`LTV_PARQUET`).
