# plugins/fairness_privacy.py
from audits.registry import plugin_registry

@plugin_registry.validator("fairness_minority_gap")
def fairness(artifacts: dict) -> dict:
    gap = float(artifacts.get("fairness", {}).get("max_gap", 0.0))
    ok = gap <= 0.10
    return {"ok": ok, "metric": gap, "message": f"max_gap={gap:.3f}"}

@plugin_registry.validator("pii_leakage_scan")
def pii(artifacts: dict) -> dict:
    findings = int(artifacts.get("pii_scan", {}).get("high_severity", 0))
    ok = findings == 0
    return {"ok": ok, "metric": findings, "message": f"high_severity={findings}"}
# plugins/fairness_privacy.py

from audits.registry import plugin_registry

@plugin_registry.metric("fairness_gap")
def fairness_gap_metric(artifacts: dict) -> dict:
    """Expose fairness gap so policies can reference fairness_gap > 0.10"""
    return {"fairness_gap": float(artifacts.get("fairness", {}).get("max_gap", 0.0))}

@plugin_registry.metric("pii_high")
def pii_high_metric(artifacts: dict) -> dict:
    """Expose number of high-severity PII findings as pii_high"""
    return {"pii_high": int(artifacts.get("pii_scan", {}).get("high_severity", 0))}
