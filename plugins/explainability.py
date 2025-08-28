# plugins/explainability.py
import math
from audits.registry import plugin_registry

@plugin_registry.metric("XI")
def xi(artifacts: dict) -> dict:
    # Expect these (or treat missing as neutral 0.5)
    shap_cons = float(artifacts.get("shap", {}).get("consistency", 0.5))
    cf_valid  = float(artifacts.get("counterfactuals", {}).get("validity", 0.5))
    sal_stab  = float(artifacts.get("saliency", {}).get("stability", 0.5))
    xi_val = (shap_cons + cf_valid + sal_stab) / 3.0
    return {"XI": max(0.0, min(1.0, xi_val))}

@plugin_registry.metric("TI")
def ti(artifacts: dict) -> dict:
    # Traceability proxy using presence + completeness of lineage items
    lineage = artifacts.get("lineage", {})
    have = sum(1 for k in ["container","python","git","seed"] if k in lineage)
    ti_val = 0.25 * have  # up to 1.0
    return {"TI": float(ti_val)}

@plugin_registry.metric("ARS")
def ars(artifacts: dict) -> dict:
    # Simple composed readiness: could be replaced with your CORTEX output
    base = float(artifacts.get("risk", {}).get("composite", 0.6))
    xi   = float(artifacts.get("XI", artifacts.get("xi", 0.6)) or 0.6)
    ti   = float(artifacts.get("TI", artifacts.get("ti", 0.6)) or 0.6)
    return {"ARS": float(0.4*base + 0.3*xi + 0.3*ti)}
# plugins/explainability.py

from audits.registry import plugin_registry
import numpy as np

@plugin_registry.metric("XI")
def xi(artifacts):
    """
    Compute XI using multiple explainability evidence sources.
    Includes local_fidelity, global_stability, faithfulness, robustness,
    coverage, human_comprehensibility, shap, counterfactuals, saliency.
    """
    def s(path, key="score", default=0.7):
        return float(artifacts.get(path, {}).get(key, default))
    parts = [
        s("local_fidelity"), s("global_stability"), s("faithfulness"),
        s("robustness"), s("coverage"), s("human_comprehensibility"),
        float(artifacts.get("shap", {}).get("consistency", 0.7)),
        float(artifacts.get("counterfactuals", {}).get("validity", 0.7)),
        float(artifacts.get("saliency", {}).get("stability", 0.7)),
    ]
    return {"XI": float(np.mean(parts))}
