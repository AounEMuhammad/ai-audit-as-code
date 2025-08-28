# plugins/redteam.py
from audits.registry import plugin_registry

@plugin_registry.metric("R")
def residual_risk(artifacts: dict) -> dict:
    attacks = artifacts.get("redteam", {}).get("attacks", {})
    fails = sum(1 for _, r in attacks.items() if r and r.get("broke", False))
    # map to [0..1] with 0.1 per failure (cap at 1.0)
    return {"R": min(1.0, 0.1 * float(fails))}
