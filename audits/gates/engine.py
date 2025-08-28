# audits/gates/engine.py
from __future__ import annotations
import ast, operator
from typing import Any, Dict, List
from audits.registry import plugin_registry

_OPS = {
    ast.Gt: operator.gt, ast.Lt: operator.lt, ast.GtE: operator.ge,
    ast.LtE: operator.le, ast.Eq: operator.eq, ast.NotEq: operator.ne
}

class _SafeEval(ast.NodeVisitor):
    def __init__(self, ctx: Dict[str, Any], has_keys: set[str]):
        self.ctx = ctx
        self.has_keys = has_keys

    def visit_Expr(self, node): return self.visit(node.value)
    def visit_Name(self, node): return self.ctx.get(node.id, None)
    def visit_Constant(self, node): return node.value
    def visit_BoolOp(self, node):
        vals = [bool(self.visit(v)) for v in node.values]
        if isinstance(node.op, ast.And):
            out = True
            for v in vals: out = out and v
            return out
        if isinstance(node.op, ast.Or):
            out = False
            for v in vals: out = out or v
            return out
        raise ValueError("Unsupported BoolOp")
    def visit_Compare(self, node):
        left = self.visit(node.left)
        out = True
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            out = out and _OPS[type(op)](left, right)
            left = right
        return out
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "has":
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant):
                raise ValueError("has() expects a constant key")
            return str(node.args[0].value) in self.has_keys
        raise ValueError("Only has('key') calls are allowed")
    def generic_visit(self, node):
        raise ValueError(f"Disallowed node: {type(node).__name__}")

def _eval_condition(expr: str, ctx: Dict[str, Any], has_keys: set[str]) -> bool:
    tree = ast.parse(expr, mode="eval")
    return bool(_SafeEval(ctx, has_keys).visit(tree.body))

def compute_metrics(artifacts: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name, fn in plugin_registry.metric_fns.items():
        try:
            metrics.update(fn(artifacts))
        except Exception as e:
            metrics[name] = float("nan")
    return metrics

def run_validators(artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = []
    for name, fn in plugin_registry.validators.items():
        try:
            out = fn(artifacts)
            out = out if isinstance(out, dict) else {"ok": bool(out)}
            out.setdefault("name", name)
            results.append(out)
        except Exception as e:
            results.append({"name": name, "ok": False, "message": f"validator crashed: {e}"})
    return results

def run_gates(policy: Dict[str, Any], metrics: Dict[str, float], evidence: Dict[str, Any], mode_name: str = "strict") -> Dict[str, Any]:
    preset = policy.get("presets", {}).get("eu_ai_act_hra", {})
    mode = policy.get("modes", {}).get(mode_name, {})
    thresholds = mode.get("thresholds", preset.get("thresholds", {}))
    required = mode.get("required_evidence", preset.get("required_evidence", []))
    blockers_cfg = mode.get("blockers", [])

    present_keys = set(evidence.keys())
    missing = [k for k in required if k not in present_keys]
    ctx = {**metrics}

    # Evaluate policy-defined blockers
    policy_blockers = []
    for b in blockers_cfg:
        label = b.get("name", "blocker")
        cond = b.get("if", "False")
        try:
            if _eval_condition(cond, ctx, present_keys):
                policy_blockers.append({"name": label, "reason": cond})
        except Exception as e:
            policy_blockers.append({"name": label, "reason": f"invalid condition: {cond} ({e})"})

    # Run plugin validators
    validator_results = run_validators({**evidence, **metrics})
    failed_validators = [ {"name": r.get("name"), "reason": r.get("message","validator failed"), "metric": r.get("metric")} 
                          for r in validator_results if not r.get("ok", False) ]

    thresholds_ok = all(metrics.get(k, float("-inf")) >= v for k, v in thresholds.items())
    decision = "PASS" if thresholds_ok and not missing and not policy_blockers and not failed_validators else "FAIL"

    return {
        "decision": decision,
        "thresholds": thresholds,
        "thresholds_ok": thresholds_ok,
        "metrics": metrics,
        "missing_evidence": missing,
        "policy_blockers": policy_blockers,
        "validator_blockers": failed_validators,
        "mode": mode_name
    }
