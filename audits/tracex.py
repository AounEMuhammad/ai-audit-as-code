# audits/tracex.py
import math

# Default weights (same as before)
TI_DEFAULT = dict(DV=0.20, MV=0.20, PL=0.20, AT=0.20, RR=0.20)
XI_DEFAULT = dict(LF=0.20, GF=0.15, FA=0.20, RS=0.15, CL=0.15, HC=0.15)

def weighted_index(values: dict, weights: dict) -> float:
    num = 0.0; den = 0.0
    for k, w in weights.items():
        v = float(values.get(k, 0.0))
        num += w * v; den += w
    return max(0.0, min(1.0, num / den if den else 0.0))

def apply_gate(risk: float, TI: float, XI: float, tiers: dict):
    """Governance-correct gate: pick tier by risk, require TI & XI >= tier thresholds."""
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
    """Explain tier selection and which thresholds passed/failed."""
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

    return {
        "tier": tier,
        "ti_min": ti_min,
        "xi_min": xi_min,
        "ti_ok": ti_ok,
        "xi_ok": xi_ok,
        "message": msg,
    }

def ars_gate(risk: float, TI: float, XI: float):
    """
    ARS-based gate (demo/case-study mode).
    Maps ARS to decision bands and assigns a notional tier label.
    """
    t = min(TI, XI)
    ars = math.sqrt(max(0.0, risk * t))

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
    return {
        "tier": tier,
        "decision": decision,
        "ars": ars,
        "t_bottleneck": t,
        "reason": reason,
    }
