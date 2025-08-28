# audits/gates/cli.py
import argparse, json, os, sys, glob
from typing import Any, Dict
import yaml

# Ensure plugin modules register themselves
# (Add more here if you create additional plugins.)
import plugins.explainability  # noqa: F401
import plugins.fairness_privacy  # noqa: F401
import plugins.redteam  # noqa: F401

from audits.gates.engine import compute_metrics, run_gates
from audits.registry.store import save_run

def _load_jsons(folder: str) -> Dict[str, Any]:
    out = {}
    if not os.path.isdir(folder):
        return out
    for p in glob.glob(os.path.join(folder, "*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                out[os.path.splitext(os.path.basename(p))[0]] = json.load(f)
        except Exception:
            pass
    return out

def main():
    ap = argparse.ArgumentParser(description="CORTEX/TRACE-X Policy Gate CLI")
    ap.add_argument("--policy", required=True, help="YAML policy file")
    ap.add_argument("--artifacts", required=True, help="Folder with *.json evidence artifacts")
    ap.add_argument("--mode", default="strict", choices=["strict","fallback","demo"], help="Gate mode")
    ap.add_argument("--out", default=".audit_runs", help="Output registry folder")
    args = ap.parse_args()

    with open(args.policy, "r", encoding="utf-8") as f:
        policy = yaml.safe_load(f)

    evidence = _load_jsons(args.artifacts)
    metrics = compute_metrics(evidence)
    result = run_gates(policy, metrics, evidence, mode_name=args.mode)
    path = save_run(result, outdir=args.out)

    print(json.dumps({"saved": path, "summary": result}, indent=2))

if __name__ == "__main__":
    sys.exit(main())
