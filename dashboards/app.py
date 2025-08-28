# --- AI Audit-as-Code: full hosted app.py (manual upload + fetch + ARS & governance gates + health panel) ---

import os, sys, json, yaml, base64, requests
import streamlit as st

# ---------------- Path guard (imports work on Streamlit Cloud) ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------- Optional bcrypt support (hashed secrets) ----------------
try:
    import bcrypt
    def verify_password(entered: str, secret: str) -> bool:
        if isinstance(secret, str) and secret.startswith("$2b$"):
            try:
                return bcrypt.checkpw(entered.encode("utf-8"), secret.encode("utf-8"))
            except Exception:
                return False
        return entered == secret
except Exception:
    def verify_password(entered: str, secret: str) -> bool:
        return entered == secret

st.set_page_config(page_title="AI Audit-as-Code", layout="wide")

# ---------------- Query params (share) ----------------
def _qp_get(name: str, default: str = "") -> str:
    v = st.query_params.get(name, default)
    if isinstance(v, (list, tuple)):  # robust
        return v[0] if v else default
    return v
mode = _qp_get("mode", ""); share_blob = _qp_get("data", "")

# ---------------- Sidebar login (secrets → env → default) ----------------
AUDITOR_USER = st.secrets.get("AUDITOR_USER", os.getenv("AUDITOR_USER", "auditor"))
AUDITOR_PASS = st.secrets.get("AUDITOR_PASS", os.getenv("AUDITOR_PASS", "change_me"))
if "authed" not in st.session_state: st.session_state.authed = False

with st.sidebar:
    if not st.session_state.authed:
        st.markdown("### Login")
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
        if submit:
            if u == AUDITOR_USER and verify_password(p, AUDITOR_PASS):
                st.session_state.authed = True; st.rerun()
            else:
                st.error("Invalid credentials")
    else:
        st.success(f"Logged in as **{AUDITOR_USER}**")
        if st.button("Logout"):
            st.session_state.clear(); st.rerun()

if not st.session_state.authed: st.stop()

# ---------------- Share mode ----------------
if mode == "share" and share_blob:
    st.title("Shared Audit Report (Read-only)")
    try:
        data = json.loads(base64.urlsafe_b64decode(share_blob.encode()).decode())
        c1,c2,c3 = st.columns(3)
        c1.metric("Risk", f'{data["risk"]["value"]:.3f}')
        c2.metric("TI", f'{data["traceability"]["TI"]:.3f}')
        c3.metric("XI", f'{data["explainability"]["XI"]:.3f}')
        st.subheader("Gate Decision"); st.json(data["gate"])
        st.subheader("Traceability Components"); st.bar_chart(data["traceability"]["components"])
        st.subheader("Explainability Components"); st.bar_chart(data["explainability"]["components"])
        st.stop()
    except Exception as e:
        st.error(f"Invalid shared data: {e}"); st.stop()

# ---------------- Persist fetched evidence across reruns ----------------
for key, default in [("fetched_single", None), ("fetched_ti", None), ("fetched_xi", None)]:
    if key not in st.session_state: st.session_state[key] = default

# ---------------- Project imports ----------------
from audits.cortex import compute_risk
# ---- Import TRACE-X helpers with fallback definitions if import fails ----
try:
    from audits.tracex import (
        weighted_index, TI_DEFAULT, XI_DEFAULT,
        apply_gate, gate_reason, ars_gate, apply_gate_progressive
    )
except Exception:
    import math

    TI_DEFAULT = dict(DV=0.20, MV=0.20, PL=0.20, AT=0.20, RR=0.20)
    XI_DEFAULT = dict(LF=0.20, GF=0.15, FA=0.20, RS=0.15, CL=0.15, HC=0.15)

    def weighted_index(values: dict, weights: dict) -> float:
        num = 0.0; den = 0.0
        for k, w in weights.items():
            v = float(values.get(k, 0.0))
            num += w * v; den += w
        return max(0.0, min(1.0, num / den if den else 0.0))

    def apply_gate(risk: float, TI: float, XI: float, tiers: dict):
        t = min(TI, XI)
        if risk >= 0.85:
            tier = "critical"; gate = tiers["critical"]
        elif risk >= 0.70:
            tier = "high"; gate = tiers["high"]
        elif risk >= 0.50:
            tier = "moderate"; gate = tiers["moderate"]
        elif risk >= 0.30:
            tier = "low"; gate = tiers["low"]
        else:
            return {"tier":"minimal","decision":"sandbox","ars":math.sqrt(max(0.0, risk*t)),"t_bottleneck":t}
        ti_min = float(gate.get("ti_min", 0.0))
        xi_min = float(gate.get("xi_min", 0.0))
        passed = (TI >= ti_min) and (XI >= xi_min)
        decision = {
            "critical": ("deploy_with_audits" if passed else "block"),
            "high":     ("deploy_with_controls" if passed else "block"),
            "moderate": ("pilot" if passed else "sandbox"),
            "low":      ("limited_use" if passed else "sandbox"),
        }[tier]
        return {"tier": tier, "decision": decision, "ars": math.sqrt(max(0.0, risk*t)), "t_bottleneck": t}

    def gate_reason(risk: float, TI: float, XI: float, tiers: dict):
        if risk >= 0.85:
            tier = "critical"; gate = tiers["critical"]
        elif risk >= 0.70:
            tier = "high"; gate = tiers["high"]
        elif risk >= 0.50:
            tier = "moderate"; gate = tiers["moderate"]
        elif risk >= 0.30:
            tier = "low"; gate = tiers["low"]
        else:
            tier = "minimal"; gate = {"ti_min": 0.0, "xi_min": 0.0}
        ti_min = float(gate.get("ti_min", 0.0))
        xi_min = float(gate.get("xi_min", 0.0))
        ti_ok  = TI >= ti_min
        xi_ok  = XI >= xi_min
        if ti_ok and xi_ok:
            msg = f"Risk tier '{tier}' and both TI≥{ti_min:.2f}, XI≥{xi_min:.2f} satisfied."
        elif not ti_ok and not xi_ok:
            msg = f"Risk tier '{tier}' but TI ({TI:.3f})<ti_min ({ti_min:.2f}) and XI ({XI:.3f})<xi_min ({xi_min:.2f})."
        elif not ti_ok:
            msg = f"Risk tier '{tier}' but TI ({TI:.3f})<ti_min ({ti_min:.2f})."
        else:
            msg = f"Risk tier '{tier}' but XI ({XI:.3f})<xi_min ({xi_min:.2f})."
        return {"tier": tier, "ti_min": ti_min, "xi_min": xi_min, "ti_ok": ti_ok, "xi_ok": xi_ok, "message": msg}

    def ars_gate(risk: float, TI: float, XI: float):
        t = min(TI, XI); ars = math.sqrt(max(0.0, risk * t))
        if ars >= 0.85:
            decision = "deploy_with_audits"; tier = "critical"
        elif ars >= 0.70:
            decision = "deploy_with_controls"; tier = "high"
        elif ars >= 0.50:
            decision = "pilot"; tier = "moderate"
        elif ars >= 0.30:
            decision = "limited_use"; tier = "low"
        else:
            decision = "sandbox"; tier = "minimal"
        reason = f"ARS={ars:.3f} mapped to decision '{decision}' (tier '{tier}') by ARS bands."
        return {"tier": tier, "decision": decision, "ars": ars, "t_bottleneck": t, "reason": reason}

    def apply_gate_progressive(risk: float, TI: float, XI: float, tiers: dict):
        t = min(TI, XI)
        if risk >= 0.85: order = ["critical","high","moderate","low"]
        elif risk >= 0.70: order = ["high","moderate","low"]
        elif risk >= 0.50: order = ["moderate","low"]
        elif risk >= 0.30: order = ["low"]
        else:
            return {"tier":"minimal","decision":"sandbox","ars":(risk*t)**0.5,"t_bottleneck":t,
                    "reason":"Risk below 0.30 ⇒ minimal tier → sandbox."}
        chosen = None
        for tier in order:
            th = tiers[tier]
            ti_ok = TI >= float(th.get("ti_min", 0.0))
            xi_ok = XI >= float(th.get("xi_min", 0.0))
            if ti_ok and xi_ok:
                chosen = tier; break
        ars = (risk * t) ** 0.5
        if not chosen:
            return {"tier": order[0], "decision":"sandbox", "ars":ars, "t_bottleneck":t,
                    "reason": f"Risk tier '{order[0]}' not satisfied; no lower tier satisfied → sandbox."}
        decision_map = {
            "critical": "deploy_with_audits",
            "high":     "deploy_with_controls",
            "moderate": "pilot",
            "low":      "limited_use",
        }
        return {"tier": chosen, "decision": decision_map[chosen], "ars":ars, "t_bottleneck":t,
                "reason": f"Risk tier '{order[0]}' not satisfied; fell back to '{chosen}' thresholds (met)."}

from audits.traceability import compute_TI_components_from_uploads
from audits.explainability import compute_XI_components_from_uploads

st.title("AI Audit-as-Code — CORTEX + TRACE-X (Hosted)")

# ---------------- Load presets ----------------
preset_path = os.path.join(REPO_ROOT, "configs", "jurisdictions.yaml")
presets = {}
if os.path.exists(preset_path):
    try:
        with open(preset_path, "r") as f:
            presets = yaml.safe_load(f) or {}
    except Exception as e:
        st.sidebar.error(f"Preset load error: {e}")

# ---------------- Default policy in session ----------------
default_policy = {
    "meta": {"name":"Hosted Demo Policy","version":"0.6"},
    "risk_inputs": {"L":0.8,"I":0.7,"k":3.0,"C":0.75,"G":0.80,"T":0.60,"E":0.70,"R":0.65},
    "tiers": {
        "critical":{"ti_min":0.80,"xi_min":0.75},
        "high":    {"ti_min":0.70,"xi_min":0.70},
        "moderate":{"ti_min":0.60,"xi_min":0.60},
        "low":     {"ti_min":0.50,"xi_min":0.50},
    },
    "ti_weights": TI_DEFAULT,
    "xi_weights": XI_DEFAULT,
}
if "policy" not in st.session_state: st.session_state.policy = default_policy

left, right = st.columns(2)

# ---------------- RIGHT: Policy + toggles ----------------
with right:
    st.subheader("Jurisdiction & Policy")
    pick = st.selectbox("Apply preset", ["(none)"] + list(presets.keys()))
    if pick != "(none)":
        p = presets[pick]
        st.session_state.policy["risk_inputs"] = p.get("risk_inputs", st.session_state.policy["risk_inputs"])
        st.session_state.policy["tiers"]       = p.get("tiers", st.session_state.policy["tiers"])
        st.session_state.policy["ti_weights"]  = p.get("ti_weights", st.session_state.policy["ti_weights"])
        st.session_state.policy["xi_weights"]  = p.get("xi_weights", st.session_state.policy["xi_weights"])
        st.success(f"Applied preset: {pick}")

    use_ars_gate = st.checkbox(
        "Use ARS-based decision (demo mode)",
        value=False,
        help="If checked, map ARS directly to decision bands (pilot/controls/audits). "
             "If unchecked, use governance gate: TI/XI must clear thresholds at the risk tier."
    )

    with st.expander("CORTEX modifiers"):
        ri = st.session_state.policy["risk_inputs"]
        row1 = st.columns(4)
        L  = row1[0].slider("L (likelihood)",  0.0,1.0, float(ri["L"]), 0.01)
        I  = row1[1].slider("I (impact)",      0.0,1.0, float(ri["I"]), 0.01)
        k  = row1[2].slider("k (risk aversion)", 1.0,8.0, float(ri["k"]), 0.1)
        Cx = row1[3].slider("C (controls)",    0.0,1.0, float(ri["C"]), 0.01)
        row2 = st.columns(4)
        G  = row2[0].slider("G (governance)",  0.0,1.0, float(ri["G"]), 0.01)
        T  = row2[1].slider("T (traceability)",0.0,1.0, float(ri["T"]), 0.01)
        E  = row2[2].slider("E (explainability)",0.0,1.0, float(ri["E"]), 0.01)
        Rm = row2[3].slider("R (residual)",    0.0,1.0, float(ri["R"]), 0.01)

    with st.expander("TRACE-X Gate Thresholds"):
        tiers = {}
        for tier in ["critical","high","moderate","low"]:
            c1, c2 = st.columns(2)
            ti_min = c1.slider(f"{tier}.ti_min", 0.0,1.0, float(st.session_state.policy["tiers"][tier]["ti_min"]), 0.01, key=f"{tier}_ti")
            xi_min = c2.slider(f"{tier}.xi_min", 0.0,1.0, float(st.session_state.policy["tiers"][tier]["xi_min"]), 0.01, key=f"{tier}_xi")
            tiers[tier] = {"ti_min": ti_min, "xi_min": xi_min}

    with st.expander("TI / XI Weights"):
        c1, c2 = st.columns(2)
        ti_w = {}; xi_w = {}
        with c1:
            st.markdown("**TI weights**")
            for k_ti, v in st.session_state.policy["ti_weights"].items():
                ti_w[k_ti] = st.slider(f"TI.{k_ti}", 0.0,1.0, float(v), 0.05, key=f"ti_{k_ti}")
        with c2:
            st.markdown("**XI weights**")
            for k_xi, v in st.session_state.policy["xi_weights"].items():
                xi_w[k_xi] = st.slider(f"XI.{k_xi}", 0.0,1.0, float(v), 0.05, key=f"xi_{k_xi}")

    st.session_state.policy["risk_inputs"] = dict(L=L, I=I, k=k, C=Cx, G=G, T=T, E=E, R=Rm)
    st.session_state.policy["tiers"] = tiers
    st.session_state.policy["ti_weights"] = ti_w
    st.session_state.policy["xi_weights"] = xi_w

    pol_yaml = yaml.safe_dump(st.session_state.policy, sort_keys=False)
    st.download_button("Download policy.yaml", pol_yaml, "policy.yaml", "text/yaml")

# ---------------- Health / Status panel ----------------
with st.expander("Health / Status", expanded=False):
    preset_name = pick if pick != "(none)" else "(custom)"
    st.write(f"**Preset:** {preset_name}")
    demo_base = "https://raw.githubusercontent.com/AounEMuhammad/ai-audit-as-code/main/evidence/demo_run"
    st.write(f"**Example base URL:** {demo_base}")
    try:
        owner_repo = "AounEMuhammad/ai-audit-as-code"; path = "evidence/demo_run"
        r = requests.get(f"https://api.github.com/repos/{owner_repo}/commits",
                         params={"path": path, "per_page": 1}, timeout=10)
        if r.ok and len(r.json()) > 0:
            ts = r.json()[0]["commit"]["committer"]["date"]
            st.write(f"**Last evidence commit (demo_run):** {ts}")
    except Exception:
        pass
    required = [
        "audit_trail.json","replication.json","dataset.hash","logs/train.log","model_card.json",
        "explainability/local_fidelity.json","explainability/global_stability.json",
        "explainability/faithfulness.json","explainability/robustness.json",
        "explainability/coverage.json","explainability/human_comprehensibility.json",
    ]
    missing = []
    for rel in required:
        url = f"{demo_base}/{rel}"
        try:
            head = requests.head(url, timeout=5)
            if head.status_code >= 400: missing.append(rel)
        except Exception:
            missing.append(rel)
    if missing: st.warning(f"Missing (demo_run): {', '.join(missing)}")
    else:       st.success("All required evidence present for demo_run.")

# ---------------- LEFT: Evidence + Run ----------------
with left:
    st.subheader("Evidence")
    tab_up, tab_url, tab_all = st.tabs(["Manual upload","Fetch via URL (single JSON)","Fetch ALL from base URL"])

    with tab_up:
        st.write("**Traceability**")
        dataset_hash = st.file_uploader("dataset.hash", type=None)
        model_card   = st.file_uploader("model_card.json", type=["json"])
        train_log    = st.file_uploader("logs/train.log", type=None)
        audit_trail  = st.file_uploader("audit_trail.json", type=["json"])
        replication  = st.file_uploader("replication.json", type=["json"])
        st.write("**Explainability**")
        lf = st.file_uploader("local_fidelity.json", type=["json"])
        gf = st.file_uploader("global_stability.json", type=["json"])
        fa = st.file_uploader("faithfulness.json", type=["json"])
        rs = st.file_uploader("robustness.json", type=["json"])
        cl = st.file_uploader("coverage.json", type=["json"])
        hc = st.file_uploader("human_comprehensibility.json", type=["json"])

    with tab_url:
        url = st.text_input("Direct JSON URL (e.g., raw GitHub link)")
        if st.button("Fetch single JSON"):
            try:
                r = requests.get(url, timeout=10); r.raise_for_status()
                fetched = r.json()
                st.session_state["fetched_single"] = fetched
                st.success("Fetched JSON:"); st.json(fetched)
            except Exception as e:
                st.error(f"Fetch failed: {e}")
        if st.button("Clear single JSON fetched"):
            st.session_state["fetched_single"] = None
            st.info("Cleared single JSON fetch.")

    with tab_all:
        st.caption("Fetch all TI & XI evidence from a base raw GitHub URL (no trailing slash).")
        scenarios = {
            "demo_run": "evidence/demo_run",
            "Scenario_A_Strong": "evidence/Scenario_A_Strong",
            "Scenario_B_Traceability_Weak": "evidence/Scenario_B_Traceability_Weak",
            "Scenario_C_Explainability_Weak": "evidence/Scenario_C_Explainability_Weak",
        }
        scen = st.selectbox("Scenario folder", list(scenarios.keys()), index=0)
        repo_base = "https://raw.githubusercontent.com/AounEMuhammad/ai-audit-as-code/main"
        base = f"{repo_base}/{scenarios[scen]}"
        st.text_input("Base URL (auto)", base, disabled=True)

        if st.button("Fetch ALL evidence from selected scenario"):
            def get_json(relpath):
                try:
                    r = requests.get(f"{base}/{relpath}", timeout=10)
                    r.raise_for_status()
                    return r.json()
                except Exception:
                    return None
            at_json = get_json("audit_trail.json")
            rr_json = get_json("replication.json")
            mc_json = get_json("model_card.json")
            dataset_present  = requests.get(f"{base}/dataset.hash", timeout=10).ok
            trainlog_present = requests.get(f"{base}/logs/train.log", timeout=10).ok

            ti_comp_all = compute_TI_components_from_uploads(
                dataset_present, mc_json, trainlog_present, at_json, rr_json
            )
            xi_comp_all = compute_XI_components_from_uploads(
                get_json("explainability/local_fidelity.json"),
                get_json("explainability/global_stability.json"),
                get_json("explainability/faithfulness.json"),
                get_json("explainability/robustness.json"),
                get_json("explainability/coverage.json"),
                get_json("explainability/human_comprehensibility.json"),
            )
            st.session_state["fetched_ti"] = ti_comp_all
            st.session_state["fetched_xi"] = xi_comp_all
            st.success(f"Fetched TI & XI from {scenarios[scen]}. Now click ‘Run Audit’.")
        if st.button("Clear ALL fetched evidence"):
            st.session_state["fetched_ti"] = None
            st.session_state["fetched_xi"] = None
            st.info("Cleared fetched TI & XI.")

    st.markdown("---")
    run_clicked = st.button("Run Audit", type="primary")

# ---------------- Run Audit ----------------
if run_clicked:
    def load_json(file):
        if not file: return None
        try: return json.load(file)
        except: return None

    # 1) Manual baseline
    at_json = load_json(audit_trail)
    rr_json = load_json(replication)
    ti_comp = compute_TI_components_from_uploads(
        dataset_hash is not None, load_json(model_card),
        train_log is not None, at_json, rr_json
    )
    xi_comp = compute_XI_components_from_uploads(
        load_json(lf), load_json(gf), load_json(fa),
        load_json(rs), load_json(cl), load_json(hc)
    )

    # 2) Merge single-URL XI (if any)
    fetched_single = st.session_state.get("fetched_single")
    if fetched_single:
        map_to = {"r2":"LF","spearman":"GF","deletion_auc":"FA","jaccard_topk":"RS","coverage":"CL","score":"HC"}
        for k, v in map_to.items():
            if k in fetched_single:
                try: xi_comp[v] = float(fetched_single[k])
                except: pass

    # 3) Prefer fetched-ALL overrides
    if st.session_state.get("fetched_ti"): ti_comp = st.session_state["fetched_ti"]
    if st.session_state.get("fetched_xi"): xi_comp = st.session_state["fetched_xi"]

    # 4) Compute indices and choose gate
    TI = weighted_index(ti_comp, st.session_state.policy["ti_weights"])
    XI = weighted_index(xi_comp, st.session_state.policy["xi_weights"])
    risk, utility = compute_risk(st.session_state.policy["risk_inputs"])
    ars_val = (risk * min(TI, XI)) ** 0.5

    if use_ars_gate:
        gate = ars_gate(risk, TI, XI)
        reason_text = gate.get("reason", f"ARS={gate['ars']:.3f} → {gate['decision']}")
    else:
        gate = apply_gate(risk, TI, XI, st.session_state.policy["tiers"])
        info = gate_reason(risk, TI, XI, st.session_state.policy["tiers"])
        reason_text = info["message"]

    # 5) Display results
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Risk (CORTEX)", f"{risk:.3f}")
    c2.metric("TI", f"{TI:.3f}")
    c3.metric("XI", f"{XI:.3f}")
    c4.metric("ARS", f"{ars_val:.3f}")

    st.subheader("Gate Decision")
    st.json(gate)

    with st.expander("Gate Details", expanded=False):
        if not use_ars_gate:
            st.write(reason_text)
            st.write(f"Thresholds: TI≥{info['ti_min']:.2f}, XI≥{info['xi_min']:.2f}")
            st.write(f"Checks: TI {'OK' if info['ti_ok'] else 'below'}, "
                     f"XI {'OK' if info['xi_ok'] else 'below'}")
        else:
            st.write(reason_text)
            st.write("Decision derived from ARS bands (demo mode).")

    st.subheader("Traceability Components")
    st.bar_chart(ti_comp)

    st.subheader("Explainability Components")
    st.bar_chart(xi_comp)

    # Report + share link
    report = {
        "risk": {"value": risk, "utility_core": utility, "inputs": st.session_state.policy["risk_inputs"]},
        "traceability": {"components": ti_comp, "TI": TI},
        "explainability": {"components": xi_comp, "XI": XI},
        "gate": gate, "ars": ars_val,
        "policy": st.session_state.policy.get("meta", {}),
    }
    blob = base64.urlsafe_b64encode(json.dumps(report).encode()).decode()
    st.download_button("Download audit_report.json", json.dumps(report, indent=2), "audit_report.json", "application/json")
    st.info("Read-only share link (append to your app URL):")
    st.code(f"?mode=share&data={blob}", language="text")

