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
