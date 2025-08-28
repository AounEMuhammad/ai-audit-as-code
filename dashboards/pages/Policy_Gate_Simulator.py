# dashboards/pages/Policy_Gate_Simulator.py
import json, io, yaml, streamlit as st
from audits.gates.engine import compute_metrics, run_gates

st.set_page_config(page_title="Policy Gate Simulator", layout="wide")
st.title("TRACE-X Policy Gate Simulator")

st.sidebar.header("Policy")
default_policy = """version: 0.3
presets:
  eu_ai_act_hra:
    thresholds: { TI: 0.75, XI: 0.70, ARS: 0.72 }
    required_evidence: [model_card, data_card, lineage, risk]
modes: { strict: {}, fallback: {}, demo: {} }
"""
policy_text = st.sidebar.text_area("YAML policy", value=default_policy, height=220)
mode = st.sidebar.selectbox("Gate mode", ["strict","fallback","demo"])

st.sidebar.header("Upload Artifacts (*.json)")
uploaded = st.sidebar.file_uploader("Drop multiple *.json files", type=["json"], accept_multiple_files=True)
evidence = {}
for f in uploaded or []:
    try:
        evidence[f.name.replace(".json","")] = json.load(f)
    except Exception:
        st.sidebar.warning(f"Could not parse {f.name}")

with st.expander("Raw Evidence", expanded=False):
    st.json(evidence)

try:
    policy = yaml.safe_load(policy_text)
except Exception as e:
    st.error(f"Policy YAML error: {e}")
    st.stop()

# Compute + Run
metrics = compute_metrics(evidence)
result = run_gates(policy, metrics, evidence, mode_name=mode)

c1, c2, c3, c4 = st.columns(4)
c1.metric("TI", f"{metrics.get('TI',0):.2f}")
c2.metric("XI", f"{metrics.get('XI',0):.2f}")
c3.metric("ARS", f"{metrics.get('ARS',0):.2f}")
c4.metric("R", f"{metrics.get('R',0):.2f}")

st.subheader(f"Decision: {'✅ PASS' if result['decision']=='PASS' else '❌ FAIL'}")
st.json(result)
# --- OPTIONAL: Save this result to a scenarios folder so Risk Analytics can pick it up ---
import os, json, time
label = st.text_input("Scenario label for saving (folder name)", value=f"manual_{int(time.time())}")
if st.button("Save result to repo (scenarios_manual)"):
    outdir = os.path.join("scenarios_manual", label)
    os.makedirs(outdir, exist_ok=True)
    # Save inputs (evidence) and computed metrics/result
    for k, v in (evidence or {}).items():
        with open(os.path.join(outdir, f"{k}.json"), "w", encoding="utf-8") as f:
            json.dump(v, f, indent=2)
    with open(os.path.join(outdir, "gate_log.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "result": result, "recommendation": result["decision"] if "recommendation" not in result else result["recommendation"]}, f, indent=2)
    st.success(f"Saved under {outdir}. You can point Risk Analytics to this folder or include it in batch eval.")
