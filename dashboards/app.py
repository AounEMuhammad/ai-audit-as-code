# --- Force repo root on PYTHONPATH so imports work on Streamlit Cloud ---
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json, yaml, base64, requests
import streamlit as st

# Optional bcrypt support (if AUDITOR_PASS is a $2b$... hash)
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

# --- Simple sidebar login (no external auth package) ---
AUDITOR_USER = os.getenv("AUDITOR_USER", "auditor")
AUDITOR_PASS = os.getenv("AUDITOR_PASS", "change_me")

st.set_page_config(page_title="AI Audit-as-Code", layout="wide")

# Read-only share mode via query params (new API)
params = st.query_params

if params.get("mode", "") == "share" and "data" in params:
    st.title("Shared Audit Report (Read-only)")
    try:
        blob = params["data"]
        data = json.loads(base64.urlsafe_b64decode(blob.encode()).decode())
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk", f'{data["risk"]["value"]:.3f}')
        c2.metric("TI", f'{data["traceability"]["TI"]:.3f}')
        c3.metric("XI", f'{data["explainability"]["XI"]:.3f}')
        st.subheader("Gate Decision"); st.json(data["gate"])
        st.subheader("Traceability Components"); st.bar_chart(data["traceability"]["components"])
        st.subheader("Explainability Components"); st.bar_chart(data["explainability"]["components"])
        st.stop()
    except Exception as e:
        st.error(f"Invalid shared data: {e}")
        st.stop()

# Sidebar login form
if "authed" not in st.session_state:
    st.session_state.authed = False

with st.sidebar:
    if not st.session_state.authed:
        st.markdown("### Login")
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
        if submit:
            if u == AUDITOR_USER and verify_password(p, AUDITOR_PASS):
                st.session_state.authed = True
                st.rerun()
            else:
                st.error("Invalid credentials")
    else:
        st.success(f"Logged in as **{AUDITOR_USER}**")
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

if not st.session_state.authed:
    st.stop()
# --- end login ---

# Imports from the project (after path guard)
from audits.cortex import compute_risk
from audits.tracex import weighted_index, TI_DEFAULT, XI_DEFAULT, apply_gate
from audits.traceability import compute_TI_components_from_uploads
from audits.explainability import compute_XI_components_from_uploads

st.title("AI Audit-as-Code — CORTEX + TRACE-X (Hosted)")

# Load presets (risk + TI/XI weights + tiers)
preset_path = os.path.join(REPO_ROOT, "configs", "jurisdictions.yaml")
presets = {}
if os.path.exists(preset_path):
    try:
        with open(preset_path, "r") as f:
            presets = yaml.safe_load(f) or {}
    except Exception as e:
        st.sidebar.error(f"Preset load error: {e}")

# Default in-session policy
default_policy = {
    "meta": {"name":"Hosted Demo Policy","version":"0.4"},
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
if "policy" not in st.session_state:
    st.session_state.policy = default_policy

left, right = st.columns(2)

# -------- Right: Presets + Policy Sliders --------
with right:
    st.subheader("Jurisdiction & Policy")
    pick = st.selectbox("Apply preset", ["(none)"] + list(presets.keys()))
    if pick != "(none)":
        p = presets[pick]
        st.session_state.policy["risk_inputs"] = p.get("risk_inputs", st.session_state.policy["risk_inputs"])
        st.session_state.policy["tiers"] = p.get("tiers", st.session_state.policy["tiers"])
        st.session_state.policy["ti_weights"] = p.get("ti_weights", st.session_state.policy["ti_weights"])
        st.session_state.policy["xi_weights"] = p.get("xi_weights", st.session_state.policy["xi_weights"])
        st.success(f"Applied preset: {pick}")

    with st.expander("CORTEX modifiers"):
        ri = st.session_state.policy["risk_inputs"]
        row1 = st.columns(4)
        L  = row1[0].slider("L",  0.0,1.0, float(ri["L"]), 0.01)
        I  = row1[1].slider("I",  0.0,1.0, float(ri["I"]), 0.01)
        k  = row1[2].slider("k",  1.0,8.0, float(ri["k"]), 0.1)
        Cx = row1[3].slider("C",  0.0,1.0, float(ri["C"]), 0.01)
        row2 = st.columns(4)
        G  = row2[0].slider("G",  0.0,1.0, float(ri["G"]), 0.01)
        T  = row2[1].slider("T",  0.0,1.0, float(ri["T"]), 0.01)
        E  = row2[2].slider("E",  0.0,1.0, float(ri["E"]), 0.01)
        Rm = row2[3].slider("R",  0.0,1.0, float(ri["R"]), 0.01)

    with st.expander("TRACE-X Gate Thresholds"):
        tiers = {}
        for tier in ["critical","high","moderate","low"]:
            c1, c2 = st.columns(2)
            ti_min = c1.slider(f"{tier}.ti_min", 0.0,1.0, float(st.session_state.policy["tiers"][tier]["ti_min"]), 0.01, key=f"{tier}_ti")
            xi_min = c2.slider(f"{tier}.xi_min", 0.0,1.0, float(st.session_state.policy["tiers"][tier]["xi_min"]), 0.01, key=f"{tier}_xi")
            tiers[tier] = {"ti_min": ti_min, "xi_min": xi_min}

    with st.expander("TI / XI Weights"):
        c1, c2 = st.columns(2)
        ti_w = {}
        xi_w = {}
        with c1:
            st.markdown("**TI weights**")
            for k_ti, v in st.session_state.policy["ti_weights"].items():
                ti_w[k_ti] = st.slider(f"TI.{k_ti}", 0.0,1.0, float(v), 0.05, key=f"ti_{k_ti}")
        with c2:
            st.markdown("**XI weights**")
            for k_xi, v in st.session_state.policy["xi_weights"].items():
                xi_w[k_xi] = st.slider(f"XI.{k_xi}", 0.0,1.0, float(v), 0.05, key=f"xi_{k_xi}")

    # Save updated policy
    st.session_state.policy["risk_inputs"] = dict(L=L, I=I, k=k, C=Cx, G=G, T=T, E=E, R=Rm)
    st.session_state.policy["tiers"] = tiers
    st.session_state.policy["ti_weights"] = ti_w
    st.session_state.policy["xi_weights"] = xi_w

    pol_yaml = yaml.safe_dump(st.session_state.policy, sort_keys=False)
    st.download_button("Download policy.yaml", pol_yaml, "policy.yaml", "text/yaml")

# -------- Left: Evidence (manual + fetch) --------
with left:
    st.subheader("Evidence")
    tab_up, tab_url = st.tabs(["Manual upload", "Fetch via URL (single JSON)"])

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

    fetched = {}
    with tab_url:
        url = st.text_input("Direct JSON URL (e.g., raw GitHub link)")
        if st.button("Fetch"):
            try:
                r = requests.get(url, timeout=10); r.raise_for_status()
                fetched = r.json()
                st.success("Fetched JSON:"); st.json(fetched)
            except Exception as e:
                st.error(f"Fetch failed: {e}")

    if st.button("Run Audit"):
        def load_json(file):
            if not file: return None
            try: return json.load(file)
            except: return None

        at_json = load_json(audit_trail)
        rr_json = load_json(replication)

        ti_comp = compute_TI_components_from_uploads(
            dataset_hash is not None,
            load_json(model_card),
            train_log is not None,
            at_json,
            rr_json
        )

        xi_comp = compute_XI_components_from_uploads(
            load_json(lf), load_json(gf), load_json(fa),
            load_json(rs), load_json(cl), load_json(hc)
        )

        # Merge fetched one-shot JSON into XI if keys match
        if fetched:
            map_to = {"r2":"LF","spearman":"GF","deletion_auc":"FA","jaccard_topk":"RS","coverage":"CL","score":"HC"}
            for k, v in map_to.items():
                if k in fetched:
                    try:
                        xi_comp[v] = float(fetched[k])
                    except:
                        pass

        TI = weighted_index(ti_comp, st.session_state.policy["ti_weights"])
        XI = weighted_index(xi_comp, st.session_state.policy["xi_weights"])
        risk, utility = compute_risk(st.session_state.policy["risk_inputs"])

        gate = apply_gate(risk, TI, XI, st.session_state.policy["tiers"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Risk", f"{risk:.3f}")
        c2.metric("TI", f"{TI:.3f}")
        c3.metric("XI", f"{XI:.3f}")
        st.subheader("Gate Decision"); st.json(gate)
        st.subheader("Traceability Components"); st.bar_chart(ti_comp)
        st.subheader("Explainability Components"); st.bar_chart(xi_comp)

        report = {
            "risk": {"value": risk, "utility_core": utility, "inputs": st.session_state.policy["risk_inputs"]},
            "traceability": {"components": ti_comp, "TI": TI},
            "explainability": {"components": xi_comp, "XI": XI},
            "gate": gate,
            "policy": st.session_state.policy.get("meta", {}),
        }
        blob = base64.urlsafe_b64encode(json.dumps(report).encode()).decode()
        st.download_button("Download audit_report.json", json.dumps(report, indent=2), "audit_report.json", "application/json")
        st.info("Read-only share link (append to your app URL):")
        st.code(f"?mode=share&data={blob}", language="text")
