# audits/tools/batch_eval.py
"""
Batch-evaluate scenarios with the Policy Gate engine and write a consolidated CSV/JSON,
including per-scenario Fix-It notes that explain *what failed and how to fix it*.

Usage:
  python -m audits.tools.batch_eval \
    --policy policies/tracex.policy.v2.yaml \
    --root scenarios_100_v2 \
    --mode strict \
    --out reports/audit_results
"""
from __future__ import annotations
import argparse, os, json, glob, csv
from typing import Dict, List
import yaml

# Ensure plugin metrics/validators are registered
import plugins.explainability  # noqa: F401
import plugins.fairness_privacy  # noqa: F401
import plugins.redteam  # noqa: F401

from audits.gates.engine import compute_metrics, run_gates

FIELDS = [
    "scenario","mode",
    "TI","XI","ARS","R",
    "decision","recommendation",
    "delta_TI","delta_XI","delta_ARS",
    "missing_evidence","policy_blockers","validator_blockers",
    "fix_it"
]

# Short remediation playbook (used for Fix-It notes)
FIX_PLAYBOOK = {
    # evidence
    "dataset": "Add dataset.json with SHA-256; reference it in model/data cards.",
    "model_card": "Add model_card.json (intent, data, metrics, limits) tied to dataset hash.",
    "data_card": "Add data_card.json (provenance, sampling, consent, splits) tied to dataset hash.",
    "lineage": "Add lineage.json {container,python,git,seed}; pin image digest and commit hash.",
    "risk": "Add risk.json with method + uncertainty; keep composite updated.",
    "audit_trail": "Add audit_trail.json (timestamped steps: train/explain/review).",
    "replication": "Add replication.json (steps, env, seed) for one-click reruns.",
    "local_fidelity": "Raise local fidelity (better surrogates/monotonic constraints).",
    "global_stability": "Stabilize explanations across perturbations / seeds.",
    "faithfulness": "Use deletion/insertion tests; remove spurious features; faithful XAI.",
    "robustness": "Add noise/shift/adversarial tests; robust training & output guardrails.",
    "coverage": "Ensure explanation coverage across classes/regions (target >= 0.8).",
    "human_comprehensibility": "Simplify explanations; label features clearly; examples.",
    # blockers
    "high_residual_risk": "Lower R: stronger red-teaming, I/O filters, rate limits, safer tools.",
    "missing_traceability": "Provide lineage.json & link artifacts for end-to-end traceability.",
    "fairness_minority_gap": "Reduce disparity: reweighting/resampling/post-processing; collect diverse data.",
    "pii_leakage_scan": "Remove PII: de-identify training data; runtime PII filters; re-scan before deploy.",
    # metric deficits (keys not in policy but useful)
    "TI_low": "Add/complete lineage; pin environment; sign/attest evidence.",
    "XI_low": "Improve XAI quality (stability, faithfulness, coverage, comprehensibility).",
    "ARS_low": "Raise readiness: reduce risk (R), and/or improve TI/XI.",
}

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

def recommend(ars: float, decision: str) -> str:
    if decision == "FAIL" or ars < 0.40: return "BLOCK"
    if ars < 0.55: return "SANDBOX"
    if ars < 0.70: return "PILOT"
    if ars < 0.85: return "DEPLOY WITH CONTROLS"
    return "DEPLOY WITH AUDIT"

def _delta(value: float, threshold: float) -> float:
    return round(max(0.0, threshold - (value or 0.0)), 4)

def _fix_for(item: str) -> str:
    return FIX_PLAYBOOK.get(item, "Remediate per policy controls (add/repair evidence or reduce risk).")

def compile_fix_notes(metrics: Dict[str, float],
                      result: Dict,
                      missing: List[str]) -> (str, float, float, float):
    """
    Build a concise fix string from deficits + blockers + missing evidence.
    Returns: (fix_text, delta_TI, delta_XI, delta_ARS)
    """
    th = result.get("thresholds", {}) or {}
    TI = float(metrics.get("TI", 0.0))
    XI = float(metrics.get("XI", 0.0))
    ARS = float(metrics.get("ARS", 0.0))
    d_ti = _delta(TI, float(th.get("TI", 0.0)))
    d_xi = _delta(XI, float(th.get("XI", 0.0)))
    d_ars = _delta(ARS, float(th.get("ARS", 0.0)))

    parts = []

    # Metric deficits
    if d_ti > 0: parts.append(f"TI below by {d_ti} → {_fix_for('TI_low')}")
    if d_xi > 0: parts.append(f"XI below by {d_xi} → {_fix_for('XI_low')}")
    if d_ars > 0: parts.append(f"ARS below by {d_ars} → {_fix_for('ARS_low')}")

    # Missing evidence
    for ev in missing:
        parts.append(f"Missing {ev} → {_fix_for(ev)}")

    # Policy + validator blockers
    for b in result.get("policy_blockers", []):
        name = b.get("name") if isinstance(b, dict) else str(b)
        parts.append(f"Blocker {name} → {_fix_for(name)}")
    for b in result.get("validator_blockers", []):
        name = b.get("name") if isinstance(b, dict) else str(b)
        parts.append(f"Blocker {name} → {_fix_for(name)}")

    return (" | ".join(parts) if parts else "Meets thresholds; no blockers."), d_ti, d_xi, d_ars

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="Path to policy YAML")
    ap.add_argument("--root", required=True, help="Folder containing scenario subfolders")
    ap.add_argument("--mode", default="strict", choices=["strict","fallback","demo"], help="Gate mode")
    ap.add_argument("--out", default="reports/audit_results", help="Output folder")
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
        result = run_gates(policy, metrics, evidence, mode_name=args.mode)

        ars = float(metrics.get("ARS", 0.0))
        decision = result["decision"]
        rec = recommend(ars, decision)

        missing = result.get("missing_evidence", [])
        fix_text, d_ti, d_xi, d_ars = compile_fix_notes(metrics, result, missing)

        row = {
            "scenario": scen,
            "mode": args.mode,
            "TI": round(float(metrics.get("TI", 0.0)), 4),
            "XI": round(float(metrics.get("XI", 0.0)), 4),
            "ARS": round(ars, 4),
            "R": round(float(metrics.get("R", 0.0)), 4),
            "decision": decision,
            "recommendation": rec,
            "delta_TI": d_ti,
            "delta_XI": d_xi,
            "delta_ARS": d_ars,
            "missing_evidence": ";".join(missing),
            "policy_blockers": ";".join((b.get("name") if isinstance(b, dict) else str(b)) for b in result.get("policy_blockers", [])),
            "validator_blockers": ";".join((b.get("name") if isinstance(b, dict) else str(b)) for b in result.get("validator_blockers", [])),
            "fix_it": fix_text,
        }
        rows.append(row)
        logs.append({"scenario": scen, "mode": args.mode, "metrics": metrics, "result": result, "fix_it": fix_text})

    csv_path = os.path.join(args.out, f"summary_{args.mode}.csv")
    json_path = os.path.join(args.out, f"summary_{args.mode}.json")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"runs": logs}, f, indent=2)

    print(f"Wrote: {csv_path}\nWrote: {json_path}")

if __name__ == "__main__":
    main()
