# audits/tools/batch_eval.py
from __future__ import annotations
import argparse, os, json, glob, csv
from typing import Dict, List
import yaml

# Register plugins so metrics exist
import plugins.explainability  # noqa: F401
import plugins.fairness_privacy  # noqa: F401
import plugins.redteam  # noqa: F401
try:
    import plugins.core_indices  # optional; fine if missing
except Exception:
    pass

from audits.gates.engine import compute_metrics, run_gates
from audits.utils.expr import safe_eval

FIELDS = [
    "scenario","mode","TI","XI","ARS","R","decision","recommendation",
    "delta_TI","delta_XI","delta_ARS","missing_evidence",
    "policy_blockers","validator_blockers","fix_it"
]

FIX = {
    "dataset":"Add dataset.json with SHA-256; link in cards.",
    "model_card":"Add model_card.json; tie to dataset hash.",
    "data_card":"Add data_card.json; provenance/sampling/splits.",
    "lineage":"Add lineage.json {container,python,git,seed}.",
    "risk":"Add risk.json with method + uncertainty.",
    "audit_trail":"Add audit_trail.json (train/explain/review).",
    "replication":"Add replication.json (steps/env/seed).",
    "local_fidelity":"Improve local fidelity (surrogates/constraints).",
    "global_stability":"Stabilize across perturbations/seeds.",
    "faithfulness":"Deletion/insertion tests; faithful XAI.",
    "robustness":"Noise/shift/adversarial tests; guardrails.",
    "coverage":"Ensure explanation coverage (≥0.8).",
    "human_comprehensibility":"Simplify & label explanations.",
    "high_residual_risk":"Lower R via stronger red-teaming & filters.",
    "missing_traceability":"Provide lineage & link artifacts.",
    "fairness_minority_gap":"Reweight/resample/post-process; diverse data.",
    "pii_leakage_scan":"De-identify data; runtime PII filters.",
    "TI_low":"Add lineage/attest env.",
    "XI_low":"Boost stability/faithfulness/coverage.",
    "ARS_low":"Reduce R and/or improve TI/XI.",
}

def _delta(val: float, thr: float) -> float:
    import math
    val = 0.0 if val is None or (isinstance(val,float) and math.isnan(val)) else float(val)
    thr = 0.0 if thr is None else float(thr)
    return round(max(0.0, thr - val), 4)

def load_artifacts(folder: str) -> dict:
    out = {}
    for p in glob.glob(os.path.join(folder, "*.json")):
        name = os.path.splitext(os.path.basename(p))[0]
        try: out[name] = json.load(open(p,"r",encoding="utf-8"))
        except Exception: pass
    return out

def apply_readiness_formula(metrics: Dict[str,float], artifacts: Dict, policy: Dict) -> None:
    """Override metrics['ARS'] from policy.readiness.formula if present."""
    rd = (policy or {}).get("readiness", {}) or {}
    formula = rd.get("formula")
    if not formula:
        return
    risk_val = float((artifacts.get("risk", {}) or {}).get("composite", 0.0))
    names = {
        # provide aliases so "Risk" also works
        "risk": risk_val, "Risk": risk_val,
        "TI": float(metrics.get("TI", 0.0)),
        "XI": float(metrics.get("XI", 0.0)),
        "R": float(metrics.get("R", 0.0)),
    }
    try:
        val = safe_eval(formula, names)
        # optional clip
        clip_bounds = rd.get("clip")
        if isinstance(clip_bounds, (list, tuple)) and len(clip_bounds) == 2:
            lo, hi = float(clip_bounds[0]), float(clip_bounds[1])
            val = max(lo, min(hi, val))
        metrics["ARS"] = float(val)
    except Exception:
        # keep existing ARS if formula fails
        pass

def recommend(ars: float, decision: str) -> str:
    if decision == "FAIL" or ars < 0.40: return "BLOCK"
    if ars < 0.55: return "SANDBOX"
    if ars < 0.70: return "PILOT"
    if ars < 0.85: return "DEPLOY WITH CONTROLS"
    return "DEPLOY WITH AUDIT"

def fix_notes(metrics: Dict[str,float], result: Dict, missing: List[str]) -> (str,float,float,float):
    th = result.get("thresholds", {}) or {}
    TI, XI, ARS = float(metrics.get("TI",0)), float(metrics.get("XI",0)), float(metrics.get("ARS",0))
    dti, dxi, dars = _delta(TI, th.get("TI")), _delta(XI, th.get("XI")), _delta(ARS, th.get("ARS"))
    parts = []
    if dti>0: parts.append(f"TI below by {dti} → {FIX['TI_low']}")
    if dxi>0: parts.append(f"XI below by {dxi} → {FIX['XI_low']}")
    if dars>0: parts.append(f"ARS below by {dars} → {FIX['ARS_low']}")
    for ev in missing: parts.append(f"Missing {ev} → {FIX.get(ev,'Add/repair evidence')}")
    for b in result.get("policy_blockers", []):
        name = b.get("name") if isinstance(b,dict) else str(b)
        parts.append(f"Blocker {name} → {FIX.get(name,'Reduce/control per policy')}")
    for b in result.get("validator_blockers", []):
        name = b.get("name") if isinstance(b,dict) else str(b)
        parts.append(f"Blocker {name} → {FIX.get(name,'Reduce/control per policy')}")
    return (" | ".join(parts) if parts else "Meets thresholds; no blockers."), dti, dxi, dars

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", default="strict", choices=["strict","fallback","demo"])
    ap.add_argument("--out", default="reports/audit_results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    policy = yaml.safe_load(open(args.policy,"r",encoding="utf-8"))

    rows, logs = [], []
    scenarios = sorted([d for d in glob.glob(os.path.join(args.root,"*")) if os.path.isdir(d)])
    if not scenarios: raise SystemExit(f"No scenarios under: {args.root}")

    for scen_dir in scenarios:
        scen = os.path.basename(scen_dir)
        artifacts = load_artifacts(scen_dir)
        metrics = compute_metrics(artifacts)              # compute TI/XI/R/ARS (baseline)
        apply_readiness_formula(metrics, artifacts, policy)  # 👈 override ARS per policy
        result = run_gates(policy, metrics, artifacts, mode_name=args.mode)

        ars = float(metrics.get("ARS", 0.0))
        decision = result["decision"]
        rec = recommend(ars, decision)

        missing = result.get("missing_evidence", [])
        fix_text, d_ti, d_xi, d_ars = fix_notes(metrics, result, missing)

        row = {
            "scenario": scen, "mode": args.mode,
            "TI": round(float(metrics.get("TI",0.0)),4),
            "XI": round(float(metrics.get("XI",0.0)),4),
            "ARS": round(ars,4),
            "R": round(float(metrics.get("R",0.0)),4),
            "decision": decision, "recommendation": rec,
            "delta_TI": d_ti, "delta_XI": d_xi, "delta_ARS": d_ars,
            "missing_evidence": ";".join(missing),
            "policy_blockers": ";".join((b.get("name") if isinstance(b,dict) else str(b)) for b in result.get("policy_blockers", [])),
            "validator_blockers": ";".join((b.get("name") if isinstance(b,dict) else str(b)) for b in result.get("validator_blockers", [])),
            "fix_it": fix_text,
        }
        rows.append(row)
        logs.append({"scenario": scen, "mode": args.mode, "metrics": metrics, "result": result, "fix_it": fix_text})

    csv_path = os.path.join(args.out, f"summary_{args.mode}.csv")
    json_path = os.path.join(args.out, f"summary_{args.mode}.json")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"runs": logs}, f, indent=2)

    print(f"Wrote: {csv_path}\nWrote: {json_path}")

if __name__ == "__main__":
    main()
