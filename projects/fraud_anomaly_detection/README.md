# projects/fraud_anomaly_detection/

One prediction problem: its recipe, its overrides, its tests. The library
stays generic; everything specific to this problem lives here.

## Layout — mirror the library's domains

```
fraud_anomaly_detection/
├── README.md                # this file
├── config.py                # the recipe: TASK / DATA / EVAL / RUN_CONFIG
├── scenarios/               # the scenario home (SCENARIOS.md is the prose stance)
│   ├── register.yaml        #   THE file to edit: definitions; doc 2 = machine-owned stats
│   ├── engine.py            #   loads/compiles/runs the register (never edit per-scenario)
│   ├── gate.py              #   fit gate trials apply (drop matched rows before fit)
│   ├── validation.py        #   refresh the evidence doc: uv run python -m ...scenarios.validation
│   └── __init__.py          #   bound register API: SCENARIOS / assign / residual_mask
├── PROJECT_INSTRUCTIONS.md  # domain guidance the agent loop reads every turn
├── data/queries/            # base_table.sql + training_data.sql (Snowflake)
├── eval/                    # custom Metric classes (automl.eval protocol)
├── model/                   # custom transformers / pipeline overrides
└── tests/                   # project-owned tests; a bare `pytest` runs them
```

Project code mirrors the library package it extends, so imports read the
same on both sides: `from projects.fraud_anomaly_detection.eval.metrics import ...`
next to `from automl.eval import ...`. Keep project tests in `tests/` here —
never in the repo-level `tests/` tree, which belongs to the core library.

## Writing PROJECT_INSTRUCTIONS.md

The proposer and coder read this file fresh **every turn** — it is how you
steer the loop between runs. Two rules, then the sections.

**Rule 1 — don't restate config.** config.py is the source of truth for
everything it defines: target, source, splits, metrics, budgets. If a fact
lives in config or SQL, don't repeat it in the instructions — duplicated
facts go stale and the loop reads the stale copy.

**Rule 2 — write what the loop can't infer.** The agent sees the recipe, the
data profile, and past trial results. Instructions earn their place only by
carrying what none of those reveal: business meaning, domain knowledge,
hard-won judgment.

Sections, and what belongs in each:

- **Goal** — what "better" means for the business, in one or two sentences.
  Interpretation, not metric definitions (config names the metric).
- **Constraints (hard)** — inviolable modeling rules: e.g. "never use the
  target in fit", latency caps, fairness/regulatory requirements. The loop
  treats these as law; a trial that violates one is invalid, whatever its
  score.
- **Domain notes** — what the data can't reveal about itself: label caveats
  and known biases, leakage rules specific to this problem, seasonality,
  segment quirks, how to interpret the secondary metrics.
- **Approaches to try** — model families, transforms, feature ideas worth
  spending trials on, in priority order.
- **Approaches to avoid** — what you've ruled out **and why**, so the loop
  doesn't rediscover dead ends.
- **Open questions** — explicit invitations to explore; the loop will pick
  these up when the leaderboard plateaus.

Style: scannable (aim under ~500 words — it's re-read every turn), concrete
("avoid models over 30s prediction latency" beats "avoid slow models"), and
living — update it after every meaningful run; git history is its changelog.

## Conventions that save a debugging session

- **Columns referenced by `RUN_CONFIG.splits` belong in
  `DATA.metadata_cols`** (unless they are SPLIT_PCT, which is pipeline
  state). Undeclared columns default to model features.
- **Custom eval metrics** read eval-only columns (outcome fields, heuristic
  scores) by declaring `required_columns`; excluded feature columns are
  still present in the eval frame.
- **The data pipeline lowercases column names.** Write predicates and column
  lists in lowercase (SPLIT_PCT is the one uppercase exception; predicate
  matching is case-insensitive either way).

## Snowflake: when your base table already exists

The harness owns its base table (CREATE OR REPLACE + SPLIT_PCT injection)
and never auto-rebuilds it. To build on a table created outside the harness,
do not point `base_table` at it — wrap it:

1. `data/queries/base_table.sql` = `SELECT * FROM <your existing table>`
   (with any filters). It must stay a single SELECT; the harness owns the
   CREATE.
2. `SnowflakeSource.base_table` = a **new** table name (e.g.
   `<your_table>_automl`). First load materializes it as a cheap
   in-warehouse copy with SPLIT_PCT injected; your table is never written.
3. Keep your upstream DDL in the project (e.g.
   `data/queries/upstream_<your_table>.sql`) as reference-only provenance.

Rebuilds happen only with the explicit `--refresh-source` flag.
