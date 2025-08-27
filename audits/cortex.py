import math
def utility_severity(L,I,k=3.0):
    return 1.0 - math.exp(-float(k)*float(L)*float(I))
def compute_risk(params, weights=None):
    w = dict(alpha=0.35, gamma=0.15, delta=0.15, theta=0.10, lam=0.10, rho=0.15)
    if weights: w.update(weights)
    U = utility_severity(params.get("L",0), params.get("I",0), params.get("k",3.0))
    C=params.get("C",0); G=params.get("G",0); T=params.get("T",0); E=params.get("E",0); Rm=params.get("R",0)
    risk = w["alpha"]*U + w["gamma"]*C + w["delta"]*G + w["theta"]*T + w["lam"]*E + w["rho"]*Rm
    risk = max(0.0, min(1.0, risk))
    return risk, U
