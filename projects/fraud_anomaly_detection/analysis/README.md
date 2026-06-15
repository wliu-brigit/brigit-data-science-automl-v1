# analysis/ — read-only investigation scripts

Two kinds of scripts live here. All are read-only toward MLflow/GCS and the
sample; graph scripts write only the gitignored store file under
`../data/graph/`. Graph deps live in the `fraud` dependency group — run
everything graph-flavored with `uv run --group fraud ...`.

## The graph workflow (run in this order)

The reusable logic lives in `../graph/` (`build` / `load` / `queries` /
`asof` / `discover`); these scripts are its thin runners:

| script | job |
|---|---|
| `graph_store_demo.py` | build the store from the local sample + capability acceptance run |
| `graph_store_build.py` | build from a REGISTERED dataset (v3 path; needs `.env` + prod registry — preflights with a clear report) |
| `graph_store_crosscheck.py` | **validate**: engine-vs-SQL scenario equivalence, label/window QA, graph-recount vs warehouse `users_on_*_72h` (quantifies the advance-grain blind spot — TODO "LINK-GRAIN EDGES"); run after every store build |
| `graph_question_battery.py` | **measure**: pooled rates, hubs vs all three truth columns, multi-type census + residual cut, proximity — with ring-concentration and early/late decay checks built in |
| `graph_discovery_queues.py` | **act**: seven snapshot review queues (residual ring members, bad neighbours, emerging farms, multi-witness pairs, fresh rings, PPR suspicion, Fraudar-style dense blocks) |
| `graph_subgroup_sweep.py` | **search**: beam-search subgroup discovery over the leak-free graph features (`asof.leakfree_features`) joined to the store's residual+mature pool — systematic conjunction hunting instead of hypothesis poking |

Semantics to keep straight: the queues are SNAPSHOT views (review/clawback —
hindsight is legitimate); any rule derived from them gets measured leak-free
via `graph.asof.leakfree_features` (strictly-prior replay, maturity-activated
seeds) before anyone quotes a precision number.

## Rerunnable lenses (dataset-general; re-run on each new base)

Kept deliberately (wendao, 2026-06-09): these still identify scenario/residual
signal and will run again on v3. Revisit after the v3 pass — any lens that
proves dead there follows the same prune path (findings → LEARNINGS, delete,
git keeps the code).

| script | job |
|---|---|
| `feature_due_diligence.py` | leakage/QA gate before discovery runs on a new feature base |
| `unsupervised_lens.py` | IF/GMM/AE discovery lens on the gated residual |
| `supervised_lens.py` | supervised GBM as a discovery lens on the gated residual |
| `subgroup_discovery.py` | beam-search conjunctive rule discovery on the residual (search/validate machinery lives in `subgroup_core.py`, shared with `graph_subgroup_sweep.py`) |
| `rule_discovery.py` | rule extraction from a logged anomaly trial (by MLflow run id) |

## Pruned history

Completed one-off screens were deleted once their findings landed in
`../LEARNINGS.md` (all recoverable from git): the v1 graph suite
(`graph_discovery_sweep`, `graph_validate_winner`, `graph_seed_coverage` —
superseded by `../graph/`), `ceiling_probe`, `edge_precision_screen`,
`institution_screen`, `ip_screen`, `residual_next_layer`, and the DuckPGQ
probe (no build for this platform; verdict in LEARNINGS).
