"""Decision/financial re-evaluation of trial model(s) on oot_new_links.

Routes through the harness eval flow: one evaluate() per trial against the
oot_new_links_with_ltv external dataset, with the project decision EvalSpec.
Records eval/oot_new_links/report.json + index + the day2_known_auc scalar
(never set as the run's primary label — decision metrics never drive selection).
See docs/to-do/decision-metric-vocabulary.md + native-decision-reeval-plan.md.

    uv run python projects/neobank_ncm/scripts/score_trial_financials.py \
        --eval-dataset-id <id> --model-run-id <id> [--model-run-id <id> ...]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# torch trials: pin BLAS/OpenMP to one thread BEFORE numpy/torch import to avoid
# the native SIGSEGV when scoring the daily frame.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def _load_env() -> None:
    envp = Path(".env")
    if not envp.exists():
        return
    for line in envp.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            if k.strip().isidentifier():
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dataset-id", required=True)
    ap.add_argument("--model-run-id", action="append", required=True, dest="model_run_ids")
    args = ap.parse_args()

    _load_env()

    from automl.eval import evaluate
    from automl.project import use_project
    from projects.neobank_ncm.eval import decision_eval_spec

    session = use_project("neobank_ncm")
    spec = decision_eval_spec()

    for run_id in args.model_run_ids:
        try:
            import torch

            torch.set_num_threads(1)
        except ImportError:
            pass
        result = evaluate(
            session=session,
            model_run_id=run_id,
            eval_dataset_id=args.eval_dataset_id,
            label="oot_new_links",
            eval_spec=spec,
            set_as_primary_label=False,  # never a selection metric
            overwrite=True,
        )
        metrics = {m["name"]: m["value"] for m in result.metrics}
        auc = metrics["day2_known_auc"]
        head = metrics["decision_report"]["scenarios"]["2_income500_match_bad_rate"]
        uw = head["tracks"]["uw"]
        print(
            f"{run_id}  day2_known_auc={auc:.5f}  "
            f"approval_rate_delta(sc2,uw)={uw['approval_rate_delta']:+.4f}  "
            f"swap_in_bad_rate={uw['swap_in_bad_rate']:.4f}  "
            f"ltv_per_link_d90={head['ltv_per_link_d90']:.4f}"
        )


if __name__ == "__main__":
    main()
