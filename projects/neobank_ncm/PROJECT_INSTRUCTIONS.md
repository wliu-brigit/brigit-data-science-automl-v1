# Project Instructions - neobank_ncm

Replication of the legacy Neobank NCM underwriting model v3
(data-science/models/underwriting/neobank/new_user/v3.0) inside this harness:
same data, same techniques, same decision metric.

## Goal

Predict `went_dpd45` at underwriting time for neobank new-customer (NCM)
loans, generalizing to the riskier segment onboarded through 2025. Better =
higher AUC on the known-only `test` split (Nov–Dec 2025). Legacy v3
reference to beat/match: test known-only AUC ≈ 0.7016 (XGBoost,
unconstrained). The `oot` split is the final test in all cases: after the
loop, the winning recipe is retrained on full 2025 (the train + test
windows) and evaluated on `oot` once (legacy Phases 4–5).

## Constraints (hard)

- Evaluation is **known-only** (`is_known = true`). Rows with synthetic labels
  must never enter a reported metric — synthetic label quality contaminates it.
- The `oot` split is for the post-AutoML re-evaluation only. Never train on
  it, never use it to pick between trials.
- Underwriting-time features only. `synthetic_score` is a label surrogate,
  never a feature; post-origination fields are out of bounds.
- Never recompute synthetic scores — consume the snapshot column as-is.

## Domain notes

- **Known rows**: booked loans with mature outcomes; `went_dpd45` is ground
  truth. **Unknown rows**: linked users with no booked loan (theoretical
  loans); `went_dpd45` is NULL and `synthetic_score` carries the
  reject-inference soft label (probability of bad).
- **Soft-label training** (the legacy technique): expand each unknown train
  row into two records — y=1 with `sample_weight = synthetic_score`, y=0 with
  `sample_weight = 1 - synthetic_score` — and pass weights to fit. Never
  hard-threshold synthetic scores.
- Legacy locked decisions are copied into `data/legacy/`: XGBoost over
  LightGBM, 80/20 known/unknown ratio, monotone constraints on 7 features,
  n_estimators=2000. `experiment_decisions.json` carries the winning params
  and the 162-feature list (pre-rename snake_case names);
  `preprocessor_meta.json` carries the final post-rename `feature_cols`,
  `model_cols`, WoE dict, imputation medians, and the legacy performance
  numbers. A faithful replication trial should start from that feature list;
  later trials may explore beyond it — the snapshot keeps every candidate
  column.
- To reproduce legacy sampling: order unknowns by `HASH(entity_id)` and take
  the first 200K (their server-side downsample), then sample to the ratio
  target with `random_state=42`. Reference counts from the final run:
  282,642 known train rows + 70,660 sampled unknowns → 423,962 weighted rows.
- `bankinstitution`: enters the model only as its WoE encoding — the
  required transformer `neobank_bankinstitution_woe`
  (model/preprocessing.py) ports the legacy semantics: fit on labeled rows
  only, sparse (<30 obs) and unseen banks map to OTHER = CHIME's WoE. When
  training on soft-label dual records, fit it on known rows before
  expansion. The legacy full-2025 fitted mapping lives beside the encoder
  (`model/bankinstitutionwoe.json`);
  `BankInstitutionWOEEncoder.from_legacy_mapping()` replays it exactly.
- `highestpayfrequency` is the one categorical feature (legacy: one-hot).
- Payday features (`daystopayday`, `dayssincepayday`, ... and derived
  `incomebuffertodaystopaydayratio`): missing = no detectable pay cycle = an
  informative signal. Keep NaN (GBMs handle natively); do not impute.
- Bad rate ≈ 10% on known rows — mild imbalance, handled natively by GBMs.

## Approaches to try

- The baseline is already authored: `model/baseline.py` (the project
  MODEL_CLASS) replays the full legacy recipe — locked features, WoE,
  dual-record soft labels at 80/20, median/payday imputation, monotone
  constraints, locked XGBoost params. Run it first; its test-split AUC is
  the replication checkpoint — compare against the **constrained** legacy
  number ≈ 0.7002 (`test_auc_constrained`; the 0.7016 in Goal is the
  unconstrained variant, the loop's stretch reference). `scripts/reeval/evaluate_split.py
  --split oot` is the one sanctioned OOT read for the winner.
- XGBoost first (legacy winner), LightGBM as challenger.
- **Neural approaches are in scope — try a reasonable one.** PyTorch is
  available (`uv sync --group neobank_ncm`). A tabular net is a genuinely
  different model family from the GBMs above and worth a real attempt; the
  architecture and feature handling are your call — apply standard tabular-NN
  hygiene as you judge best. The only non-negotiables are the project-wide
  ones: fit/scale/select on known-train rows only (never touch `test`/`oot`),
  stay within `per_trial_seconds`, and use the same `synthetic_score` soft-label
  data the GBMs train on. Flag in the trial's findings any legacy property a net
  ends up dropping (e.g. the monotone constraints).
  - **Operational must (not a modeling choice):** call `torch.set_num_threads(1)`
    at the top of `model.py`. Full-data torch `fit` otherwise hits a native
    OpenMP/MKL threading crash (SIGSEGV) under the runner; the loop also exports
    `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1`. This is harness hygiene — the same
    net trains fine at dry-run scale without it.
- Median imputation for non-payday numerics (legacy also coerced ±inf to NaN
  before imputing); the 11 derived ratio/flag candidates from the legacy EDA
  are already materialized in the base table (8 survived their selection, 3
  did not — all available).
- Monotone constraints on directionally clear features (legacy accepted them
  at an AUC cost < 0.002).

## Trial-code requirements

- To load project assets (the locked params/features in `data/legacy/*.json`,
  the WoE encoder), resolve paths via the package, never via `__file__`
  traversal — trial code executes from a deep trial directory, so
  `Path(__file__).parents[N]` math breaks. Use
  `PROJECT_DIR = Path(projects.neobank_ncm.__file__).parent`, or import the
  ready-made anchors: `from projects.neobank_ncm.model.baseline import
  DECISIONS_PATH, FEAT_RENAME`.
- The train frame mixes known rows (`went_dpd45` 0/1) and unknown rows
  (`went_dpd45` NULL, `synthetic_score` set). `fit` must handle both — drop
  unknowns or soft-label them; never feed NULL targets to the estimator.
- The runner's pre-fit check fits the model on `df.head(200)` and re-checks
  on 10 rows: `fit` must work on tiny samples that may contain only known or
  only unknown rows (guard the dual-record expansion accordingly).

## Approaches to avoid

- `scale_pos_weight` — tangles with the soft-label sample weights (legacy
  decision).
- Hard-thresholding synthetic labels (cliff effect; loses uncertainty).
- Tuning on any synthetic-labeled metric.
