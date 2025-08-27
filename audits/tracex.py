import math
TI_DEFAULT = dict(DV=0.20, MV=0.20, PL=0.20, AT=0.20, RR=0.20)
XI_DEFAULT = dict(LF=0.20, GF=0.15, FA=0.20, RS=0.15, CL=0.15, HC=0.15)
def weighted_index(values: dict, weights: dict) -> float:
    num=0.0; den=0.0
    for k,w in weights.items():
        v=float(values.get(k,0.0)); num += w*v; den += w
    return max(0.0, min(1.0, num/den if den else 0.0))
def apply_gate(risk: float, TI: float, XI: float, tiers: dict):
    t=min(TI,XI)
    if risk>=0.85: tier="critical"; gate=tiers["critical"]
    elif risk>=0.70: tier="high"; gate=tiers["high"]
    elif risk>=0.50: tier="moderate"; gate=tiers["moderate"]
    elif risk>=0.30: tier="low"; gate=tiers["low"]
    else:
        return {"tier":"minimal","decision":"sandbox","ars":math.sqrt(max(0.0,risk*t)),"t_bottleneck":t}
    passed = (TI>=gate["ti_min"]) and (XI>=gate["xi_min"])
    decision = {"critical":"deploy_with_audits" if passed else "block",
                "high":"deploy_with_controls" if passed else "block",
                "moderate":"pilot" if passed else "sandbox",
                "low":"limited_use" if passed else "sandbox"}[tier]
    return {"tier":tier,"decision":decision,"ars":math.sqrt(max(0.0,risk*t)),"t_bottleneck":t}
