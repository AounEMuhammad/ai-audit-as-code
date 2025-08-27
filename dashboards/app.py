# --- Force repo root on PYTHONPATH so imports work on Streamlit Cloud ---
import os, sys, json, yaml, base64, requests
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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

st.set_page_config(page_title="AI Audit-as-Code", layout="wide")

# ----------------- Query params (share link support) -----------------
def _qp_get(name: str, default: str = "") -> str:
    v = st.query_params.get(name, default)
    if isinstance(v, (list, tuple)):  # be robust across versions
        v = v[0] if v else default
    return v

mode = _qp_get("mode", "")
share_blob = _qp_get("data", "")

# ----------------- Simple sidebar login -----------------
AUDITOR_USER = st.secrets.get("AUDITOR_USER", os.getenv("AUDITOR_USER", "auditor"))
AUDITOR_PASS = st.secrets.get("AUDITOR_PASS", os.getenv("AUDITOR_PASS", "change_me"))

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

# ----------------- Read-only share mode -----------------
if mode == "share" and share_blob:
    st.title("Shared Audit Report (Read-only)")
    try:
        data = json.loads(base64.urlsafe_b64decode(share_blob.encode()).decode())
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

# ----------------- Keep fetched evidence across reruns -----------------
for key, default in [
    ("fetched_single", None),  # single-JSON fetched dict (XI-only)
    ("fetched_ti", None),      # fetch-ALL computed TI components
    ("fetched_xi", None),      # fetch-ALL computed XI components
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ----------------- Project imports (after path guard & login) -----------------
from audits.cortex import compute_risk
from audits.tracex import weighted_index, TI_DEFAULT, XI_DEFAULT, apply_gate
from audits.traceability import compute_TI_components_from_uploads
from audits.explainability import compute_XI_components_from_uploads

st.title("AI Audit-as-Code — CORTEX + TRACE-X (Hosted)")

# ----------------- Load presets -----------------
preset_path = os.path.join(REPO_ROOT, "configs", "jurisdictions.yaml")
presets = {}
if os.path.exists(preset_path):
    try:
        with open(preset_path, "r") as f:
            presets = yaml.safe_load(f) or {}
    except Exception as e:
        st.sidebar.error(f"Preset load error: {e}")

# Default policy if none in session
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

# ----------------- RIGHT: Jurisdiction & Policy -----------------
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

    # Save updated policy back to session
    st.session_state.policy["risk_inputs"] = dict(L=L, I=I, k=k, C=Cx, G=G, T=T, E=E, R=Rm)
    st.session_state.policy["tiers"] = tiers
    st.session_state.policy["ti_weights"] = ti_w
    st.session_state.policy["xi_weights"] = xi_w

    pol_yaml = yaml.safe_dump(st.session_state.policy, sort_keys=False)
    st.download_button("Download policy.yaml", pol_yaml, "policy.yaml", "text/yaml")

# ----------------- LEFT: Evidence + Buttons -----------------
with left:
    st.subheader("Evidence")

    tab_up, tab_url, tab_all = st.tabs([
        "Manual upload",
        "Fetch via URL (single JSON)",
        "Fetch ALL from base URL"
    ])

    # --- Manual upload tab ---
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

    # --- Single JSON fetch tab ---
        # --- Fetch ALL tab ---
    with tab_all:
        st.caption("Fetch all TI & XI evidence from a base raw GitHub URL (no trailing slash).")

        # Scenario selector (relative subfolder under evidence/)
        scenarios = {
            "demo_run": "evidence/demo_run",
            "Scenario_A_Strong": "evidence/Scenario_A_Strong",
            "Scenario_B_Traceability_Weak": "evidence/Scenario_B_Traceability_Weak",
            "Scenario_C_Explainability_Weak": "evidence/Scenario_C_Explainability_Weak",
        }
        scen = st.selectbox("Scenario folder", list(scenarios.keys()), index=0)

        # Build base URL automatically from repo + scenario folder
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

            # Traceability
            at_json = get_json("audit_trail.json")
            rr_json = get_json("replication.json")
            mc_json = get_json("model_card.json")
            dataset_present  = requests.get(f"{base}/dataset.hash", timeout=10).ok
            trainlog_present = requests.get(f"{base}/logs/train.log", timeout=10).ok

            ti_comp_all = compute_TI_components_from_uploads(
                dataset_present,
                mc_json,
                trainlog_present,
                at_json,
                rr_json
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
)

    # --- Button immediately under the tabs ---
    st.markdown("---")
    run_clicked = st.button("Run Audit", type="primary")

# ======================= Run Audit handler =======================
if run_clicked:
    def load_json(file):
        if not file:
            return None
        try:
            return json.load(file)
        except:
            return None

    # 1) Build from MANUAL uploads (default baseline)
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

    # 2) Merge SINGLE-URL fetched explainability (keeps manual TI; augments XI only)
    fetched_single = st.session_state.get("fetched_single")
    if fetched_single:
        map_to = {
            "r2": "LF",
            "spearman": "GF",
            "deletion_auc": "FA",
            "jaccard_topk": "RS",
            "coverage": "CL",
            "score": "HC",
        }
        for k, v in map_to.items():
            if k in fetched_single:
                try:
                    xi_comp[v] = float(fetched_single[k])
                except:
                    pass

    # 3) Prefer FETCH-ALL (base URL) if present — full overrides for TI & XI
    if st.session_state.get("fetched_ti"):
        ti_comp = st.session_state["fetched_ti"]
    if st.session_state.get("fetched_xi"):
        xi_comp = st.session_state["fetched_xi"]

    # 4) Compute indices and gate
    TI = weighted_index(ti_comp, st.session_state.policy["ti_weights"])
    XI = weighted_index(xi_comp, st.session_state.policy["xi_weights"])
    risk, utility = compute_risk(st.session_state.policy["risk_inputs"])
    gate = apply_gate(risk, TI, XI, st.session_state.policy["tiers"])

    # 5) Display results
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
