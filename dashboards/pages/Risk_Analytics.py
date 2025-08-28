# dashboards/pages/Risk_Analytics.py
# Requirements: plotly>=5.24, pandas>=2.2 (add to requirements.txt)
import os, json, glob
import pandas as pd
import streamlit as st

# Graceful Plotly import for Streamlit Cloud
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ModuleNotFoundError:
    st.error("Plotly is required for Risk Analytics. Please add `plotly>=5.24` (and `pandas>=2.2`) to requirements.txt and redeploy.")
    st.stop()

# Engine + plugins so metrics are registered when recomputing from folders
try:
    from audits.gates.engine import compute_metrics, run_gates
    import yaml
    import plugins.explainability  # noqa: F401
    import plugins.fairness_privacy  # noqa: F401
    import plugins.redteam  # noqa: F401
    ENGINE_AVAILABLE = True
except Exception:
    ENGINE_AVAILABLE = False

st.set_page_config(page_title="Risk Analytics", layout="wide")
st.title("AI Audit-as-Code — Risk Analytics & Fix-It")

st.markdown("""
Choose a data source:
- **Upload CSV** (from `audits/tools/batch_eval.py` or `batch_eval_per_scenario.py`)
- **Recompute from folder** (reads scenario JSONs in your repo and runs the gate live)
""")

# ---------------- helpers ----------------
def parse_semicolon(val):
    s = "" if pd.isna(val) else str(val)
    return [x for x in s.split(";") if x]

def normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure numeric columns
    for c in ["TI","XI","ARS","R"]:
        if c not in df.columns: df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Ensure decision & recommendation
    if "decision" not in df.columns:
        if "recommendation" in df.columns:
            df["decision"] = df["recommendation"].apply(lambda x: "FAIL" if str(x).strip().upper()=="BLOCK" else "PASS")
        else:
            df["decision"] = "PASS"
    if "recommendation" not in df.columns:
        def rec_from(ars, decision):
            if decision == "FAIL" or float(ars) < 0.40: return "BLOCK"
            if ars < 0.55: return "SANDBOX"
            if ars < 0.70: return "PILOT"
            if ars < 0.85: return "DEPLOY WITH CONTROLS"
            return "DEPLOY WITH AUDIT"
        df["recommendation"] = [rec_from(a, d) for a, d in zip(df["ARS"], df["decision"])]

    # Ensure text columns
    for name in ["missing_evidence","policy_blockers","validator_blockers","mode","fix_it","delta_TI","delta_XI","delta_ARS"]:
        if name not in df.columns:
            df[name] = "" if not name.startswith("delta_") else 0.0

    # Scenario label
    if "scenario" not in df.columns:
        df["scenario"] = [f"scenario_{i+1}" for i in range(len(df))]
    return df

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
    "faithfulness": "Use deletion/insertion tests; faithful XAI methods; remove spurious features.",
    "robustness": "Noise/shift/adversarial tests; robust training; output guardrails.",
    "coverage": "Ensure explanation coverage across classes/regions (target ≥0.8).",
    "human_comprehensibility": "Simplify explanations; clearer feature names; examples.",
    # blockers
    "high_residual_risk": "Lower R: stronger red-teaming, I/O filters, rate limits, safer tools.",
    "missing_traceability": "Provide lineage.json & link artifacts for end-to-end traceability.",
    "fairness_minority_gap": "Reduce disparity: reweighting/resampling/post-processing; diverse data.",
    "pii_leakage_scan": "Remove PII: de-identify data; runtime PII filters; re-scan before deploy.",
    # metric deficits
    "TI_low": "Add/complete lineage; pin environment; sign/attest evidence.",
    "XI_low": "Improve XAI quality (stability, faithfulness, coverage, comprehensibility).",
    "ARS_low": "Raise readiness: reduce risk (R) and/or improve TI/XI."
}

def _delta(value: float, threshold: float) -> float:
    return round(max(0.0, threshold - (value or 0.0)), 4)

def compile_fix_notes(metrics: dict, result: dict, missing: list) -> (str, float, float, float):
    th = result.get("thresholds", {}) or {}
    TI = float(metrics.get("TI", 0.0)); XI = float(metrics.get("XI", 0.0)); ARS = float(metrics.get("ARS", 0.0))
    d_ti = _delta(TI, float(th.get("TI", 0.0)))
    d_xi = _delta(XI, float(th.get("XI", 0.0)))
    d_ars = _delta(ARS, float(th.get("ARS", 0.0)))
    parts = []
    if d_ti > 0: parts.append(f"TI below by {d_ti} → {FIX_PLAYBOOK['TI_low']}")
    if d_xi > 0: parts.append(f"XI below by {d_xi} → {FIX_PLAYBOOK['XI_low']}")
    if d_ars > 0: parts.append(f"ARS below by {d_ars} → {FIX_PLAYBOOK['ARS_low']}")
    for ev in missing:
        parts.append(f"Missing {ev} → {FIX_PLAYBOOK.get(ev, 'Add/repair evidence per policy')}")
    for b in result.get("policy_blockers", []):
        name = b.get("name") if isinstance(b, dict) else str(b)
        parts.append(f"Blocker {name} → {FIX_PLAYBOOK.get(name, 'Reduce/control per policy')}")
    for b in result.get("validator_blockers", []):
        name = b.get("name") if isinstance(b, dict) else str(b)
        parts.append(f"Blocker {name} → {FIX_PLAYBOOK.get(name, 'Reduce/control per policy')}")
    return (" | ".join(parts) if parts else "Meets thresholds; no blockers."), d_ti, d_xi, d_ars

def load_scenario_artifacts(scenario_name: str) -> dict:
    # Look under common roots committed to repo
    for base in ["scenarios_100_v2", "scenarios_100", "scenarios_manual"]:
        d = os.path.join(base, scenario_name)
        if os.path.isdir(d):
            out = {}
            for p in glob.glob(os.path.join(d, "*.json")):
                key = os.path.splitext(os.path.basename(p))[0]
                try:
                    out[key] = json.load(open(p, "r", encoding="utf-8"))
                except Exception:
                    pass
            return out
    return {}

def xi_breakdown_from_artifacts(artifacts: dict) -> dict:
    def take(k, key="score", fallback=None):
        if k not in artifacts: return fallback
        v = artifacts[k].get(key)
        if v is None:
            for alt in ["score","consistency","validity","stability"]:
                if alt in artifacts[k]:
                    v = artifacts[k][alt]; break
        return float(v) if v is not None else fallback
    parts = {
        "local_fidelity": take("local_fidelity"),
        "global_stability": take("global_stability"),
        "faithfulness": take("faithfulness"),
        "robustness": take("robustness"),
        "coverage": take("coverage"),
        "human_comprehensibility": take("human_comprehensibility"),
        "shap_consistency": take("shap", "consistency"),
        "cf_validity": take("counterfactuals", "validity"),
        "saliency_stability": take("saliency", "stability"),
    }
    return {k:v for k,v in parts.items() if v is not None}

def evidence_presence_from_artifacts(artifacts: dict) -> dict:
    keys = [
        "dataset","model_card","data_card","lineage","risk","audit_trail","replication",
        "local_fidelity","global_stability","faithfulness","robustness","coverage","human_comprehensibility",
        "fairness","pii_scan","redteam","shap","counterfactuals","saliency"
    ]
    return {k: (k in artifacts) for k in keys}

def recommend(ars: float, decision: str) -> str:
    if decision == "FAIL" or ars < 0.40: return "BLOCK"
    if ars < 0.55: return "SANDBOX"
    if ars < 0.70: return "PILOT"
    if ars < 0.85: return "DEPLOY WITH CONTROLS"
    return "DEPLOY WITH AUDIT"

def build_df_from_folder(root_dir: str, policy_path: str, mode: str, prefer_logs: bool=True) -> pd.DataFrame | None:
    if not ENGINE_AVAILABLE or not os.path.isfile(policy_path) or not os.path.isdir(root_dir):
        return None
    policy = yaml.safe_load(open(policy_path, "r", encoding="utf-8"))
    rows = []
    scen_dirs = sorted([d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)])
    for scen_dir in scen_dirs:
        scen = os.path.basename(scen_dir)
        artifacts = {}
        for p in glob.glob(os.path.join(scen_dir, "*.json")):
            key = os.path.splitext(os.path.basename(p))[0]
            try:
                artifacts[key] = json.load(open(p, "r", encoding="utf-8"))
            except Exception:
                pass

        # Use existing gate_log.json if requested and present, else recompute
        logp = os.path.join(scen_dir, "gate_log.json")
        if prefer_logs and os.path.exists(logp):
            try:
                lg = json.load(open(logp, "r", encoding="utf-8"))
                m = lg.get("metrics", {}) or {}
                result = lg.get("result", {}) or {}
                ars = float(m.get("ARS", 0.0))
                decision = result.get("decision", "FAIL")
                rec = lg.get("recommendation", recommend(ars, decision))
                missing = result.get("missing_evidence", [])
                fix_text, d_ti, d_xi, d_ars = compile_fix_notes(m, result, missing)
                rows.append({
                    "scenario": scen, "mode": mode,
                    "TI": m.get("TI", 0.0), "XI": m.get("XI", 0.0), "ARS": ars, "R": m.get("R", 0.0),
                    "decision": decision, "recommendation": rec,
                    "delta_TI": d_ti, "delta_XI": d_xi, "delta_ARS": d_ars,
                    "missing_evidence": ";".join(missing),
                    "policy_blockers": ";".join(b.get("name") if isinstance(b,dict) else str(b) for b in result.get("policy_blockers", [])),
                    "validator_blockers": ";".join(b.get("name") if isinstance(b,dict) else str(b) for b in result.get("validator_blockers", [])),
                    "fix_it": fix_text,
                })
                continue
            except Exception:
                pass

        # Recompute live
        metrics = compute_metrics(artifacts)
        result = run_gates(policy, metrics, artifacts, mode_name=mode)
        ars = float(metrics.get("ARS", 0.0))
        decision = result["decision"]
        rec = recommend(ars, decision)
        missing = result.get("missing_evidence", [])
        fix_text, d_ti, d_xi, d_ars = compile_fix_notes(metrics, result, missing)
        rows.append({
            "scenario": scen, "mode": mode,
            "TI": metrics.get("TI", 0.0), "XI": metrics.get("XI", 0.0), "ARS": ars, "R": metrics.get("R", 0.0),
            "decision": decision, "recommendation": rec,
            "delta_TI": d_ti, "delta_XI": d_xi, "delta_ARS": d_ars,
            "missing_evidence": ";".join(missing),
            "policy_blockers": ";".join(b.get("name") if isinstance(b,dict) else str(b) for b in result.get("policy_blockers", [])),
            "validator_blockers": ";".join(b.get("name") if isinstance(b,dict) else str(b) for b in result.get("validator_blockers", [])),
            "fix_it": fix_text,
        })
    if not rows: return None
    return pd.DataFrame(rows)

# ---------------- UI: data source ----------------
src = st.radio("Data source", ["Upload CSV", "Recompute from folder"], horizontal=True)

df = None
if src == "Upload CSV":
    csv_file = st.file_uploader("Upload batch summary CSV (e.g., reports/audit_results/summary_strict.csv)", type=["csv"])
    if csv_file is None:
        st.info("Upload the summary CSV produced by `audits/tools/batch_eval*.py`.")
        st.stop()
    df = pd.read_csv(csv_file)
    df = normalize_summary(df)
else:
    if not ENGINE_AVAILABLE:
        st.error("Live recompute requires the gate engine and plugins available in the app runtime.")
        st.stop()
    c1, c2, c3 = st.columns([1.2, 0.8, 0.8])
    with c1:
        root_dir = st.text_input("Scenarios root folder (inside repo)", "scenarios_100_v2")
    with c2:
        policy_path = st.text_input("Policy YAML", "policies/tracex.policy.v2.yaml")
    with c3:
        mode = st.selectbox("Gate mode", ["strict","fallback","demo"])
    prefer_logs = st.checkbox("Prefer gate_log.json if present", value=True)
    df = build_df_from_folder(root_dir, policy_path, mode, prefer_logs)
    if df is None or df.empty:
        st.info("Provide a valid folder & policy to recompute, or switch to 'Upload CSV'.")
        st.stop()

# ---------------- overview ----------------
st.markdown("### Overview")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Scenarios", f"{len(df):,}")
c2.metric("PASS", int((df["decision"]=="PASS").sum()))
c3.metric("FAIL", int((df["decision"]=="FAIL").sum()))
c4.metric("Mean ARS", f"{pd.to_numeric(df['ARS']).mean():.2f}")
c5.metric("Mean XI", f"{pd.to_numeric(df['XI']).mean():.2f}")

st.markdown("### TI vs XI Quadrant")
quad = px.scatter(
    df, x="TI", y="XI", color="recommendation",
    hover_data=["scenario","ARS","R","decision","policy_blockers","validator_blockers","fix_it"],
    opacity=0.85
)
quad.update_layout(height=480)
st.plotly_chart(quad, use_container_width=True)

st.markdown("### ARS Distribution by Recommendation")
hist = px.histogram(df, x="ARS", color="recommendation", nbins=30, barmode="overlay", opacity=0.75)
hist.update_layout(height=320)
st.plotly_chart(hist, use_container_width=True)

# ---------------- per-scenario Fix-It ----------------
st.markdown("## Scenario Fix-It Panel")
scen = st.selectbox("Pick a scenario", df["scenario"].tolist())
row = df[df["scenario"]==scen].iloc[0]

with st.expander("Thresholds (for visualization)", expanded=False):
    th_ti = st.number_input("TI threshold", value=0.75, step=0.01)
    th_xi = st.number_input("XI threshold", value=0.72, step=0.01)
    th_ars = st.number_input("ARS threshold", value=0.74, step=0.01)

ti, xi, ars, r = float(row["TI"]), float(row["XI"]), float(row["ARS"]), float(row["R"])
decision = str(row["decision"]); rec = str(row["recommendation"])

radar = go.Figure()
radar.add_trace(go.Scatterpolar(r=[ti, xi, ars], theta=["TI","XI","ARS"], fill="toself", name="Scenario"))
radar.add_trace(go.Scatterpolar(r=[th_ti, th_xi, th_ars], theta=["TI","XI","ARS"], name="Thresholds"))
radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, height=420)
st.plotly_chart(radar, use_container_width=True)

# What’s below threshold?
deficits = []
if ti < th_ti: deficits.append({"Area":"TI","Current":round(ti,3),"Target":th_ti,"Delta":round(th_ti-ti,3),"Fix":FIX_PLAYBOOK["TI_low"]})
if xi < th_xi: deficits.append({"Area":"XI","Current":round(xi,3),"Target":th_xi,"Delta":round(th_xi-xi,3),"Fix":FIX_PLAYBOOK["XI_low"]})
if ars < th_ars: deficits.append({"Area":"ARS","Current":round(ars,3),"Target":th_ars,"Delta":round(th_ars-ars,3),"Fix":FIX_PLAYBOOK["ARS_low"]})
def_df = pd.DataFrame(deficits) if deficits else pd.DataFrame([{"Area":"—","Current":"—","Target":"—","Delta":"—","Fix":"No metric below threshold"}])

cA, cB = st.columns([1.1,1])
with cA:
    st.markdown("### What’s below threshold?")
    st.dataframe(def_df, use_container_width=True, hide_index=True)
with cB:
    st.markdown("### Decision")
    st.metric("Gate Decision", decision)
    st.metric("Recommendation", rec)
    st.metric("Residual Risk (R)", f"{r:.2f}")

# Fix-It table from CSV fields (fallback computed if absent)
missing = parse_semicolon(row.get("missing_evidence",""))
pblks = parse_semicolon(row.get("policy_blockers",""))
vblks = parse_semicolon(row.get("validator_blockers",""))
fix_text = str(row.get("fix_it","")).strip()

fix_rows = []
for ev in missing:
    fix_rows.append({"Type":"Missing evidence", "Item":ev, "How to fix": FIX_PLAYBOOK.get(ev, "Add/repair per policy")})
for b in pblks + vblks:
    fix_rows.append({"Type":"Blocker", "Item":b, "How to fix": FIX_PLAYBOOK.get(b, "Reduce/control per policy")})

if not fix_rows and deficits:
    for d in deficits:
        item = "TI_low" if d["Area"]=="TI" else ("XI_low" if d["Area"]=="XI" else "ARS_low")
        fix_rows.append({"Type":"Metric deficit", "Item":item, "How to fix": FIX_PLAYBOOK.get(item, "Improve metric via evidence/risk controls")})

st.markdown("### Fix-It Recommendations")
if fix_text and fix_text != "nan":
    st.caption(f"Notes: {fix_text}")
fix_df = pd.DataFrame(fix_rows) if fix_rows else pd.DataFrame([{"Type":"—","Item":"—","How to fix":"No blockers or missing evidence; meets thresholds."}])
st.dataframe(fix_df, use_container_width=True, hide_index=True)

# Deep dive if artifacts are present in the repo
artifacts = load_scenario_artifacts(scen)
if artifacts:
    st.markdown("### XI Breakdown (from scenario artifacts)")
    parts = xi_breakdown_from_artifacts(artifacts)
    if parts:
        xidf = pd.DataFrame({"metric": list(parts.keys()), "score": [float(v) for v in parts.values()]})
        bar = px.bar(xidf, x="metric", y="score", range_y=[0,1])
        bar.update_layout(height=320, xaxis_tickangle=-30)
        st.plotly_chart(bar, use_container_width=True)
    else:
        st.caption("No detailed explainability JSONs (local_fidelity, faithfulness, etc.) found for this scenario.")

    st.markdown("### Evidence Presence")
    presence = evidence_presence_from_artifacts(artifacts)
    presdf = pd.DataFrame({"evidence": list(presence.keys()), "present": ["Yes" if v else "No" for v in presence.values()]})
    presdf["present_num"] = [1 if v=="Yes" else 0 for v in presdf["present"]]
    heat = px.imshow(presdf[["present_num"]].T, labels=dict(x="Evidence", y="Present?"),
                     x=presdf["evidence"].tolist(), y=["present"], color_continuous_scale=["#ffdddd","#a6e3a1"])
    heat.update_layout(height=180, coloraxis_showscale=False)
    st.plotly_chart(heat, use_container_width=True)
else:
    st.caption("Scenario artifacts not found in the repo (looked under scenarios_100_v2/, scenarios_100/, scenarios_manual/). "
               "Upload the summary CSV or save manual runs to a folder to enable deep-dive.")
