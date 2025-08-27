# pipelines/run_cli_audit.py
import os, json, sys, yaml, requests

# Import project code
from audits.cortex import compute_risk
from audits.tracex import weighted_index, apply_gate
from audits.traceability import compute_TI_components_from_uploads
from audits.explainability import compute_XI_components_from_uploads

# ------------------------------------------------------------------
# Configuration (override via env if needed)
# ------------------------------------------------------------------
EVIDENCE_BASE = os.environ.get(
    "EVIDENCE_BASE",
    "https://raw.githubusercontent.com/AounEMuhammad/ai-audit-as-code/main/evidence/demo_run"
)
PRESET_NAME = os.environ.get(
    "PRESET_NAME",
    "NIST AI RMF (High governance posture)"
)

def get_json(path):
    """Fetch a JSON file from EVIDENCE_BASE/path. Return None if not found."""
    try:
        r = requests.get(f"{EVIDENCE_BASE}/{path}", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# ------------------------------------------------------------------
# Fetch TI evidence
# ------------------------------------------------------------------
at_json = get_json("audit_trail.json")
rr_json = get_json("replication.json")
mc_json = get_json("model_card.json")
# “Presence” checks
dv_present = requests.get(f"{EVIDENCE_BASE}/dataset.hash", timeout=10).ok
pl_present = requests.get(f"{EVIDENCE_BASE}/logs/train.log", timeout=10).ok

ti_comp = compute_TI_components_from_uploads(
    dv_present,
    mc_json,
    pl_present,
    at_json,
    rr_json
)

# ------------------------------------------------------------------
# Fetch XI evidence
# ------------------------------------------------------------------
xi_comp = compute_XI_components_from_uploads(
    get_json("explainability/local_fidelity.json"),
    get_json("explainability/global_stability.json"),
    get_json("explainability/faithfulness.json"),
    get_json("explainability/robustness.json"),
    get_json("explainability/coverage.json"),
    get_json("explainability/human_comprehensibility.json"),
)

# ------------------------------------------------------------------
# Load preset from configs/jurisdictions.yaml
# ------------------------------------------------------------------
with open("configs/jurisdictions.yaml", "r") as f:
    presets = yaml.safe_load(f) or {}

if PRESET_NAME not in presets:
    print(json.dumps({
        "error": f"Preset '{PRESET_NAME}' not found.",
        "available_presets": list(presets.keys())
    }, indent=2))
    sys.exit(2)

preset = presets[PRESET_NAME]
tiers = preset["tiers"]
ti_w  = preset["ti_weights"]
xi_w  = preset["xi_weights"]
ri    = preset["risk_inputs"]

# ------------------------------------------------------------------
# Compute TI, XI, Risk, and Gate decision
# ------------------------------------------------------------------
TI = weighted_index(ti_comp, ti_w)
XI = weighted_index(xi_comp, xi_w)
risk, utility = compute_risk(ri)
gate = apply_gate(risk, TI, XI, tiers)

result = {
    "preset": PRESET_NAME,
    "evidence_base": EVIDENCE_BASE,
    "risk": risk,
    "TI": TI,
    "XI": XI,
    "gate": gate
}
print(json.dumps(result, indent=2))

# Fail build if gate decision is "block"
sys.exit(1 if gate["decision"] == "block" else 0)
