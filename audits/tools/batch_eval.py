# audits/tools/batch_eval.py
"""
Batch-evaluate scenarios with the Policy Gate CLI and write a consolidated CSV/JSON.
Usage:
  python -m audits.tools.batch_eval \
    --policy policies/tracex.policy.v2.yaml \
    --root ./scenarios_100_v2 \
    --mode strict \
    --out reports/audit_results
"""
from __future__ import annotations
import argparse, os, json, glob, csv
from audits.gates.engine import compute_metrics, run_gates
import yaml

def load_artifacts(folder: str) -> dict:
    out = {}
    for p in glob.glob(os.path.join(folder, "*.json")):
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            out[name] = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--root", required=True, help="Root folder that contains N scenario subfolders")
    ap.add_argument("--mode", default="strict", choices=["strict","fallback","demo"])
    ap.add_argument("--out", default="reports/audit_results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    policy = yaml.safe_load(open(args.policy, "r", encoding="utf-8"))

    rows, logs = [], []
    scenarios = sorted([d for d in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(d)])
    for scen_dir in scenarios:
        scen = os.path.basename(scen_dir)
        evidence = load_artifacts(scen_dir)
        metrics = compute_metrics(evidence)
        res = run_gates(policy, metrics, evidence, mode_name=args.mode)

        row = {
            "scenario": scen,
            "TI": round(float(metrics.get("TI", 0.0)), 4),
            "XI": round(float(metrics.get("XI", 0.0)), 4),
            "ARS": round(float(metrics.get("ARS", 0.0)), 4),
            "R": round(float(metrics.get("R", 0.0)), 4),
            "decision": res["decision"],
            "recommendation": "BLOCK" if res["decision"]=="FAIL" else (
                "DEPLOY WITH AUDIT" if metrics.get("ARS",0)>=0.85 else
                "DEPLOY WITH CONTROLS" if metrics.get("ARS",0)>=0.70 else
                "PILOT" if metrics.get("ARS",0)>=0.55 else
                "SANDBOX" if metrics.get("ARS",0)>=0.40 else "BLOCK"
            ),
            "missing_evidence": ";".join(res["missing_evidence"]),
            "policy_blockers": ";".join(b["name"] for b in res["policy_blockers"]),
            "validator_blockers": ";".join(b["name"] for b in res["validator_blockers"]),
        }
        rows.append(row)
        logs.append({"scenario": scen, "metrics": metrics, "result": res})

    # write csv + json
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
