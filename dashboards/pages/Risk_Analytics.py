# dashboards/pages/Risk_Analytics.py
# Streamlit page to analyze scenario-level TI/XI/Risk, compute ARS, and show decisions.
# - Default "Decision" uses ARS bands (limited_use, pilot, deploy_with_controls, deploy_with_audits, sandbox, block).
# - Also shows Gate decision based on TI/XI thresholds per CORTEX risk tier.
# Requirements: streamlit, pandas, numpy, pyyaml, plotly

from __future__ import annotations

import io
import os
import json
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Plotly for charts
import plotly.express as px
import plotly.graph_objects as go

import yaml

# -----------------------------
# Config loading (policy v2)
# -----------------------------
DEFAULT_CONFIG: Dict = {
    "name": "TRACE-X v2 policy",
    "version": "2.0",
    "ars": {
        # ARS = sqrt(Risk * min(TI,XI))  (TRACE-X)
        "formula": "sqrt(risk * min(ti, xi))",
        # App-level ARS decision bands (UI convention):
        # We intentionally reuse CORTEX numeric bands for ARS to keep semantics familiar.
        "decision_bands": [
            {"min": 0.85, "decision": "deploy_with_audits"},
            {"min": 0.70, "decision": "deploy_with_controls"},
            {"min": 0.50, "decision": "pilot"},
            {"min": 0.30, "decision": "limited_use"},
            {"min": 0.00, "decision": "sandbox"},
        ],
    },
    "gates": {
        "risk_tiers": {
            "minimal": [0.00, 0.29],
            "low": [0.30, 0.49],
            "moderate": [0.50, 0.69],
            "high": [0.70, 0.84],
            "critical": [0.85, 1.00],
        },
        "thresholds": {
            # pass_decision is what the gate recommends when TI/XI clear the tier.
            # fail_decision is what the gate recommends otherwise.
            "minimal":  {"ti_min": 0.00, "xi_min": 0.00, "pass_decision": "sandbox",               "fail_decision": "sandbox"},
            "low":      {"ti_min": 0.50, "xi_min": 0.50, "pass_decision": "limited_use",           "fail_decision": "sandbox"},
            "moderate": {"ti_min": 0.60, "xi_min": 0.60, "pass_decision": "pilot",                 "fail_decision": "sandbox"},
            "high":     {"ti_min": 0.70, "xi_min": 0.70, "pass_decision": "deploy_with_controls",  "fail_decision": "block"},
            "critical": {"ti_min": 0.80, "xi_min": 0.75, "pass_decision": "deploy_with_audits",    "fail_decision": "block"},
        },
    },
    "labels": {
        "tier_order": ["minimal", "low", "moderate", "high", "critical"]
    }
}

def load_policy_config() -> Dict:
    candidate_paths = [
        os.path.join("policies", "tracex.policy.v2.yaml"),
        os.path.join("policies", "tracex.policy.yaml"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    return DEFAULT_CONFIG

CFG = load_policy_config()

# -----------------------------
# Helpers: tiers, thresholds, ARS, decisions
# -----------------------------
def clamp01(x: float) -> float:
    try:
        return float(np.clip(x, 0.0, 1.0))
    except Exception:
        return 0.0

def risk_tier(risk: float) -> str:
    r = clamp01(risk)
    for name, (lo, hi) in CFG["gates"]["risk_tiers"].items():
        if (r >= lo) and (r <= hi):
            return name
    # Fallback
    return "minimal"

def gate_thresholds_for_risk(risk: float) -> Tuple[float, float, str, str]:
    tier = risk_tier(risk)
    entry = CFG["gates"]["thresholds"][tier]
    return float(entry["ti_min"]), float(entry["xi_min"]), entry["pass_decision"], entry["fail_decision"]

def compute_ars(risk: float, ti: float, xi: float) -> Tuple[float, float]:
    # TRACE-X: ARS = sqrt(Risk * min(TI, XI))
    t = min(clamp01(ti), clamp01(xi))
    v = np.sqrt(clamp01(risk) * t)
    return float(v), float(t)

def ars_decision(ars: float) -> Tuple[str, str]:
    # Choose the highest band where ars >= min
    for band in CFG["ars"]["decision_bands"]:
        if ars >= band["min"]:
            return band["decision"], f"ARS={ars:.3f} mapped to decision '{band['decision']}' by ARS bands."
    return "sandbox", f"ARS={ars:.3f} below lowest band; defaulting to 'sandbox'."

def gate_decision(risk: float, ti: float, xi: float) -> Tuple[str, bool, Dict]:
    ti_min, xi_min, pass_decision, fail_decision = gate_thresholds_for_risk(risk)
    passed = (ti >= ti_min) and (xi >= xi_min)
    decision = pass_decision if passed else fail_decision
    details = {
        "tier": risk_tier(risk),
        "ti_min": ti_min,
        "xi_min": xi_min,
        "passed": passed,
    }
    return decision, passed, details

# -----------------------------
# Column normalization
# -----------------------------
ALIASES = {
    "scenario": ["scenario", "scenario_id", "id", "name", "case"],
    "domain":   ["domain", "sector", "category"],
    "risk":     ["risk", "cortex_risk", "r", "risk_score"],
    "ti":       ["ti", "traceability", "traceability_index", "traceability_idx"],
    "xi":       ["xi", "explainability", "explainability_index", "explainability_idx"],
    "ars":      ["ars", "assured_readiness_score", "readiness"],
}

def pick_col(df: pd.DataFrame, keys) -> Optional[str]:
    cols = [c for c in df.columns]
    lower = {c.lower(): c for c in cols}
    for k in keys:
        if k in lower:
            return lower[k]
    return None

def normalize_dataframe(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    # Try to locate canonical columns
    scenario_col = pick_col(df, [s.lower() for s in ALIASES["scenario"]]) or None
    risk_col     = pick_col(df, [s.lower() for s in ALIASES["risk"]]) or None
    ti_col       = pick_col(df, [s.lower() for s in ALIASES["ti"]]) or None
    xi_col       = pick_col(df, [s.lower() for s in ALIASES["xi"]]) or None
    domain_col   = pick_col(df, [s.lower() for s in ALIASES["domain"]]) or None
    ars_col      = pick_col(df, [s.lower() for s in ALIASES["ars"]]) or None

    # If scenario missing, synthesize an index
    if scenario_col is None:
        df["scenario"] = [f"scenario_{i+1}" for i in range(len(df))]
    else:
        df = df.rename(columns={scenario_col: "scenario"})

    # Required numeric inputs
    required_missing = []
    if risk_col is None: required_missing.append("risk")
    if ti_col   is None: required_missing.append("ti")
    if xi_col   is None: required_missing.append("xi")
    if required_missing:
        raise ValueError(f"Missing required columns: {', '.join(required_missing)}. "
                         f"Upload a CSV with columns for risk, ti, xi (plus optional domain, scenario).")

    df = df.rename(columns={
        risk_col: "risk",
        ti_col:   "ti",
        xi_col:   "xi",
    })

    if domain_col:
        df = df.rename(columns={domain_col: "domain"})
    else:
        df["domain"] = "unspecified"

    # Ensure numeric and clipped
    for c in ["risk", "ti", "xi"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df[c] = df[c].clip(0.0, 1.0)

    # ARS + bottleneck + tiers + decisions
    ars_vals, t_vals = [], []
    tiers, gate_pass, gate_decisions, ars_decisions, reasons = [], [], [], [], []
    ti_min_list, xi_min_list = [], []

    for _, row in df.iterrows():
        ars, t = compute_ars(row["risk"], row["ti"], row["xi"])
        ars_vals.append(ars)
        t_vals.append(t)
        tier = risk_tier(row["risk"])
        tiers.append(tier)

        gd, passed, info = gate_decision(row["risk"], row["ti"], row["xi"])
        gate_decisions.append(gd)
        gate_pass.append(bool(passed))
        ti_min_list.append(info["ti_min"])
        xi_min_list.append(info["xi_min"])

        ad, because = ars_decision(ars)
        ars_decisions.append(ad)
        reasons.append(because)

    df["t_bottleneck"]   = t_vals
    df["ars"]            = ars_vals if ars_col is None else pd.to_numeric(df[ars_col], errors="coerce").fillna(ars_vals)
    df["risk_tier"]      = tiers
    df["gate_decision"]  = gate_decisions
    df["gate_pass"]      = gate_pass
    df["gate_ti_min"]    = ti_min_list
    df["gate_xi_min"]    = xi_min_list
    df["ars_decision"]   = ars_decisions
    df["ars_reason"]     = reasons
    return df

# -----------------------------
# UI
# -----------------------------
st.title("Risk Analytics (TRACE-X)")

st.markdown(
    """
This page computes **Assured Readiness Score (ARS)** and decisions per scenario.
- **ARS (default decision):** uses bands aligned to CORTEX numeric ranges to suggest _sandbox / limited_use / pilot / deploy_with_controls / deploy_with_audits_.
- **Gate decision:** enforces **TI/XI** minimums by **risk tier** (Minimal, Low, Moderate, High, Critical).

Upload a CSV with columns: `scenario, risk, ti, xi` (plus optional `domain`). If ARS is present, it will be recomputed for consistency.
"""
)

uploaded = st.file_uploader("Upload summary CSV", type=["csv"])

decision_mode = st.radio(
    "Decision mode to display as primary",
    options=["ARS bands (default)", "Gate (TI/XI thresholds)"],
    index=0,
    horizontal=True,
)

if not uploaded:
    # Tiny template to help users
    st.info("No file uploaded yet. You can also generate a CSV with the batch evaluator.")
    ex = pd.DataFrame({
        "scenario": ["ex_1", "ex_2"],
        "domain": ["healthcare", "finance"],
        "risk": [0.62, 0.81],
        "ti": [0.71, 0.82],
        "xi": [0.74, 0.75],
    })
    st.download_button("Download CSV template", data=ex.to_csv(index=False).encode("utf-8"),
                       file_name="risk_analytics_template.csv", mime="text/csv")
    st.stop()

# Parse and normalize
try:
    df_raw = pd.read_csv(uploaded)
    df = normalize_dataframe(df_raw)
except Exception as e:
    st.error(str(e))
    st.stop()

primary_col = "ars_decision" if decision_mode.startswith("ARS") else "gate_decision"

# KPI row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Scenarios", len(df))
c2.metric("Mean ARS", f"{df['ars'].mean():.2f}")
c3.metric("Mean TI", f"{df['ti'].mean():.2f}")
c4.metric("Mean XI", f"{df['xi'].mean():.2f}")
c5.metric("Gate PASS (%)", f"{100.0 * df['gate_pass'].mean():.1f}")

# Decision distribution
st.subheader("Decisions overview")
counts = df[primary_col].value_counts().reset_index()
counts.columns = ["decision", "count"]
fig_bar = px.bar(counts, x="decision", y="count", text="count")
fig_bar.update_traces(textposition="outside")
fig_bar.update_layout(yaxis_title=None, xaxis_title=None, margin=dict(t=20, b=20, l=20, r=20))
st.plotly_chart(fig_bar, use_container_width=True)

# Quadrant / lattice chart: y=risk, x=min(TI,XI)
st.subheader("Risk vs. Assurance bottleneck (min(TI, XI))")
fig = px.scatter(
    df,
    x="t_bottleneck",
    y="risk",
    color=primary_col,
    symbol="gate_pass",
    hover_data=["scenario", "domain", "risk_tier", "ti", "xi", "ars"],
    labels={"t_bottleneck": "Traceability / Explainability (min(TI, XI))", "risk": "CORTEX Risk"},
)

# Add CORTEX tier lines (y) and TI/XI gate lines (x)
for y in [0.30, 0.50, 0.70, 0.85]:
    fig.add_hline(y=y, line=dict(dash="dot"))
for x in [0.50, 0.60, 0.70, 0.80]:
    fig.add_vline(x=x, line=dict(dash="dot"))

fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
st.plotly_chart(fig, use_container_width=True)

# Table
st.subheader("Details")
show_cols = [
    "scenario", "domain", "risk", "risk_tier",
    "ti", "xi", "t_bottleneck", "ars",
    "ars_decision", "gate_decision", "gate_pass",
    "gate_ti_min", "gate_xi_min", "ars_reason"
]
st.dataframe(df[show_cols].sort_values(["risk", "t_bottleneck"], ascending=[False, True]), use_container_width=True)

# Per-scenario inspector
st.subheader("Scenario inspector")
sel = st.selectbox("Pick a scenario", df["scenario"].tolist(), index=0)
row = df[df["scenario"] == sel].iloc[0]

# Primary (default) decision always shown as 'decision'
if primary_col == "ars_decision":
    decision = row["ars_decision"]
    reason = row["ars_reason"]
else:
    decision = row["gate_decision"]
    reason = f"Gate {'PASSED' if row['gate_pass'] else 'FAILED'} for tier '{row['risk_tier']}' (TI≥{row['gate_ti_min']}, XI≥{row['gate_xi_min']})."

payload = {
    "risk": float(row["risk"]),
    "risk_tier": row["risk_tier"],
    "ti": float(row["ti"]),
    "xi": float(row["xi"]),
    "t_bottleneck": float(row["t_bottleneck"]),
    "ars": float(row["ars"]),
    "decision_mode": "ars_bands" if primary_col == "ars_decision" else "gate_thresholds",
    "decision": decision,
    "reason": reason,
    "gate": {
        "decision": row["gate_decision"],
        "passed": bool(row["gate_pass"]),
        "ti_min": float(row["gate_ti_min"]),
        "xi_min": float(row["gate_xi_min"]),
    },
}
st.code(json.dumps(payload, indent=2), language="json")

# Download computed CSV
out = df[show_cols]
st.download_button(
    "Download computed summary CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="summary_with_ars_and_decisions.csv",
    mime="text/csv",
)
