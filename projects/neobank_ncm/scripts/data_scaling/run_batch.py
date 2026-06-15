"""Drive the data-scaling trials: run each, then OOT decision-eval it.

Resilient + idempotent: records slug -> run_id/status in a manifest; a trial
already marked FINISHED is skipped on re-run. Per-trial failures don't abort
the batch. After each FINISHED trial it runs score_trial_financials.py to
record the OOT new-links decision report onto the run.

    uv run python projects/neobank_ncm/scripts/data_scaling/run_batch.py
    uv run python projects/neobank_ncm/scripts/data_scaling/run_batch.py --only data_scaling_known100k_synth00
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

EXP_DIR = (
    Path(__file__).resolve().parents[2]  # .../scripts/data_scaling/<file> -> project dir
    / "experiments" / "neobank_ncm" / "neobank_ncm_v3_replicate"
)
MANIFEST = Path(".cache/automl/fin/data_scaling_runs.json")
EVAL_DATASET_ID = "ev_abb30380d8bc"  # oot_new_links_with_ltv snapshot

SLUGS = [
    "data_scaling_knownfull_synth00",
    "data_scaling_knownfull_synth10",
    "data_scaling_knownfull_synth30",
    "data_scaling_known050k_synth00",
    "data_scaling_known100k_synth00",
    "data_scaling_known150k_synth00",
    "data_scaling_known200k_synth00",
    "data_scaling_known250k_synth00",
]


def _load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def _save(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def _parse_kv(stdout: str) -> dict:
    out = {}
    for line in stdout.splitlines():
        if line.startswith("AUTOML_"):
            k, _, v = line.partition("=")
            out[k] = v
    return out


def run_trial(slug: str) -> dict:
    print(f"\n===== RUN {slug} =====", flush=True)
    proc = subprocess.run(
        [sys.executable, str(EXP_DIR / slug / "run.py")],
        capture_output=True, text=True,
    )
    kv = _parse_kv(proc.stdout)
    status = kv.get("AUTOML_STATUS", "UNKNOWN")
    run_id = kv.get("AUTOML_RUN_ID", "")
    print(f"  status={status} run_id={run_id}", flush=True)
    if status != "FINISHED":
        # surface the tail of stderr/stdout for diagnosis
        print("  --- stdout tail ---", flush=True)
        print("\n".join(proc.stdout.splitlines()[-15:]), flush=True)
        print("  --- stderr tail ---", flush=True)
        print("\n".join(proc.stderr.splitlines()[-15:]), flush=True)
    return {"slug": slug, "status": status, "run_id": run_id, "eval": "pending"}


def decision_eval(run_id: str) -> bool:
    print(f"  decision-eval {run_id} ...", flush=True)
    proc = subprocess.run(
        [sys.executable,
         "projects/neobank_ncm/scripts/score_trial_financials.py",
         "--eval-dataset-id", EVAL_DATASET_ID,
         "--model-run-id", run_id],
        capture_output=True, text=True,
    )
    ok = proc.returncode == 0
    tail = "\n".join((proc.stdout or proc.stderr).splitlines()[-8:])
    print(f"  eval {'OK' if ok else 'FAILED'}: {tail}", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", help="restrict to these slug(s)")
    ap.add_argument("--rerun-finished", action="store_true")
    args = ap.parse_args()

    manifest = _load_manifest()
    slugs = args.only or SLUGS
    for slug in slugs:
        prior = manifest.get(slug, {})
        if prior.get("status") == "FINISHED" and prior.get("eval") == "ok" and not args.rerun_finished:
            print(f"skip {slug} (already FINISHED + eval ok: {prior.get('run_id')})", flush=True)
            continue
        rec = run_trial(slug)
        if rec["status"] == "FINISHED" and rec["run_id"]:
            rec["eval"] = "ok" if decision_eval(rec["run_id"]) else "eval_failed"
        manifest[slug] = rec
        _save(manifest)
    print("\n===== manifest =====", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
