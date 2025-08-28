# dashboards/pages/Risk_Analytics.py
import os, json, glob
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Risk Analytics", layout="wide")
st.title("AI Audit-as-Code — Risk Analytics")

st.markdown("""
Upload a batch **summary CSV** (from the batch evaluator) or point to a **scenario root**
folder to auto-aggregate results. Explore:
- **TI vs XI quadrant** (color by decision/recommendation)
- **ARS distribution** (by recommendation band)
- **Evidence completeness** (per scenario or aggregate)
- **Blockers frequency** + where evidence lacks
- **Radar**: compare a scenario's indices against policy thresholds
""")

# --- Inputs ---
colA, colB = st.columns([1,1])
with colA:
    csv_file = st.file_uploader("Upload summary CSV (e.g., reports/audit_results/summary_strict.csv)", type=["csv"])
with colB:
    root_dir = st.text_input("OR Enter scenarios root folder (server path)", value="")

df = None

def parse_semicolon(col):
    if pd.isna(col) or not str(col).strip():
        return []
    return [x for x in str(col).split(";") if x]

if csv_file is not None:
    df = pd.read_csv(csv_file)
elif root_dir and os.path.isdir(root_dir):
    # try to reconstruct a dataframe from gate_log.json files
    rows = []
    for scen_dir in sorted([d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)]):
        scen = os.path.basename(scen_dir)
        logp = os.path.join(scen_dir, "gate_log.json")
        if os.path.exists(logp):
            try:
                lg = json.load(open(logp, "r", encoding="utf-8"))
                m = lg.get("metrics", {})
                blocks = lg.get("blockers", [])
                rows.append({
                    "scenario": scen,
                    "TI": m.get("TI", 0.0),
                    "XI": m.get("XI", 0.0),
                    "ARS": m.get("ARS", 0.0),
                    "R": m.get("R", 0.0),
                    "decision": "FAIL" if "BLOCK" in lg.get("recommendation","") else "PASS",
                    "recommendation": lg.get("recommendation",""),
                    "missing_evidence": "",  # not in v2 logs; ok to leave empty
                    "policy_blockers": ";".join(blocks),
                    "validator_blockers": "",
                })
            except Exception:
                pass
    if rows:
        df = pd.DataFrame(rows)

if df is None or df.empty:
    st.info("Provide a summary CSV or a valid scenarios folder to see analytics.")
    st.stop()

# Ensure types
for c in ["TI","XI","ARS","R"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

st.markdown("### Overview")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Scenarios", f"{len(df):,}")
c2.metric("PASS", int((df["decision"]=="PASS").sum()))
c3.metric("FAIL", int((df["decision"]=="FAIL").sum()))
c4.metric("Mean ARS", f"{df['ARS'].mean():.2f}")
c5.metric("Mean XI", f"{df['XI'].mean():.2f}")

# --- TI vs XI Quadrant ---
st.markdown("### TI vs XI Quadrant")
quad = px.scatter(
    df, x="TI", y="XI",
    color="recommendation",
    hover_data=["scenario","ARS","R","decision","policy_blockers","validator_blockers"],
    opacity=0.85
)
quad.update_layout(height=500)
st.plotly_chart(quad, use_container_width=True)

# --- ARS Distribution ---
st.markdown("### ARS Distribution by Recommendation")
hist = px.histogram(df, x="ARS", color="recommendation", nbins=30, barmode="overlay", opacity=0.75)
hist.update_layout(height=350)
st.plotly_chart(hist, use_container_width=True)

# --- Blockers frequency ---
st.markdown("### Blockers Frequency")
def explode_counts(series):
    counts = {}
    for val in series.fillna("").tolist():
        for item in parse_semicolon(val):
            counts[item] = counts.get(item, 0) + 1
    return pd.DataFrame({"blocker": list(counts.keys()), "count": list(counts.values())}).sort_values("count", ascending=False)

blk = explode_counts(df["policy_blockers"])  # + validator_blockers if present separately
if not blk.empty:
    bar = px.bar(blk, x="blocker", y="count")
    bar.update_layout(height=300, xaxis_tickangle=-30)
    st.plotly_chart(bar, use_container_width=True)
else:
    st.write("No blockers found in the uploaded batch.")

# --- Evidence completeness heat (optional aggregate) ---
st.markdown("### Evidence Completeness (optional)")
if "missing_evidence" in df.columns:
    df["missing_count"] = df["missing_evidence"].apply(lambda s: len(parse_semicolon(s)))
    comp = px.histogram(df, x="missing_count", nbins=6, title="Missing required evidence count per scenario")
    comp.update_layout(height=300)
    st.plotly_chart(comp, use_container_width=True)

# --- Scenario Radar ---
st.markdown("### Scenario Radar")
scen = st.selectbox("Pick a scenario", df["scenario"].tolist())
row = df[df["scenario"]==scen].iloc[0]
# Thresholds: show your default v2 thresholds; user can change
with st.expander("Thresholds", expanded=False):
    th_ti = st.number_input("TI threshold", value=0.75, step=0.01)
    th_xi = st.number_input("XI threshold", value=0.72, step=0.01)
    th_ars = st.number_input("ARS threshold", value=0.74, step=0.01)

radar = go.Figure()
radar.add_trace(go.Scatterpolar(
    r=[row["TI"], row["XI"], row["ARS"]],
    theta=["TI","XI","ARS"],
    fill="toself",
    name="Scenario"
))
radar.add_trace(go.Scatterpolar(
    r=[th_ti, th_xi, th_ars],
    theta=["TI","XI","ARS"],
    name="Thresholds"
))
radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, height=430)
st.plotly_chart(radar, use_container_width=True)

st.caption("Tip: Use this to explain where the model/evidence lacks—e.g., XI below threshold due to low global_stability/faithfulness or missing lineage causing TI=0.")
