# audits/tools/batch_eval.py
"""
Batch-evaluate scenarios with the Policy Gate engine and write a consolidated CSV/JSON.

Usage:
  python -m audits.tools.batch_eval \
    --policy policies/tracex.policy.v2.yaml \
    --root ./scenarios_100_v2 \
    --mode strict \
    --out reports/audit_results
"""
from __future__ import annotations
import argparse, os, json, glob, csv
import yaml

# Ensure plugin metrics/validators are registered
import plugins.explainability  # noqa: F401
import plugins.fairness_privacy  # noqa: F401
import plugins.redteam  # noqa: F401

from audits.gates.engine import compute_metrics, run_gates

def load_artifacts(folder: str) -> dict:
    out = {}
    for p in glob.glob(os.path.join(folder, "*.json")):
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            with open(p, "r", encoding="utf-8") as f:
                out[name] = json.load(f)
        except Exception:
            pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--root", required=True, help="Folder containing scenario subfolders")
    ap.add_argument("--mode", default="strict", choices=["strict","fallback","demo"])
    ap.add_argument("--out", default="reports/audit_results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.policy, "r", encoding="utf-8") as f:
        policy = yaml.safe_load(f)

    rows, logs = [], []
    scenarios = sorted([d for d in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(d)])
    if not scenarios:
        raise SystemExit(f"No scenarios found under: {args.root}")

    for scen_dir in scenarios:
        scen = os.path.basename(scen_dir)
        evidence = load_artifacts(scen_dir)
        metrics = compute_metrics(evidence)
        res = run_gates(policy, metrics, evidence, mode_name=args.mode)

        # Recommendation band from ARS (and blockers via decision)
        ars = float(metrics.get("ARS", 0.0))
        if res["decision"] == "FAIL" or ars < 0.40:
            rec = "BLOCK"
        elif ars < 0.55:
            rec = "SANDBOX"
        elif ars < 0.70:
            rec = "PILOT"
        elif ars < 0.85:
            rec = "DEPLOY WITH CONTROLS"
        else:
            rec = "DEPLOY WITH AUDIT"

        row = {
            "scenario": scen,
            "TI": round(float(metrics.get("TI", 0.0)), 4),
            "XI": round(float(metrics.get("XI", 0.0)), 4),
            "ARS": round(float(metrics.get("ARS", 0.0)), 4),
            "R": round(float(metrics.get("R", 0.0)), 4),
            "decision": res["decision"],
            "recommendation": rec,
            "missing_evidence": ";".join(res["missing_evidence"]),
            "policy_blockers": ";".join(b["name"] for b in res["policy_blockers"]),
            "validator_blockers": ";".join(b["name"] for b in res["validator_blockers"]),
        }
        rows.append(row)
        logs.append({"scenario": scen, "metrics": metrics, "result": res})

    csv_path = os.path.join(args.out, f"summary_{args.mode}.csv")
    json_path = os.path.join(args.out, f"summary_{args.mode}.json")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"runs": logs}, f, indent=2)

    print(f"Wrote: {csv_path}\nWrote: {json_path}")

if __name__ == "__main__":
    main()
