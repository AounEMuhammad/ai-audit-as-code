# audits/monitor/drift.py
from math import log
from typing import Iterable

def population_stability_index(p: Iterable[float], q: Iterable[float], eps: float = 1e-9) -> float:
    return sum((qi - pi) * log((qi + eps) / (pi + eps)) for pi, qi in zip(p, q))

def learn_threshold(history: list[dict], target: str = "ARS", pctl: float = 0.1):
    vals = sorted(h.get(target, 0.0) for h in history if h.get("decision") == "PASS")
    if not vals: return None
    idx = max(0, int(len(vals) * pctl) - 1)
    return vals[idx]
