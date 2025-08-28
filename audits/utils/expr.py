# audits/utils/expr.py
# Safe, tiny expression evaluator for readiness formulas in policy.
import ast, operator as op, math

# Allowed operators
_ALLOWED_BINOPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Mod: op.mod, ast.Pow: op.pow
}
_ALLOWED_UNARY = {ast.UAdd: op.pos, ast.USub: op.neg}

def clip(x, lo, hi):  # helper exposed to formulas: clip(x, lo, hi)
    return max(lo, min(hi, x))

# Allowed functions inside formulas
_ALLOWED_FUNCS = {
    "min": min,
    "max": max,
    "clip": clip,
    "sqrt": math.sqrt,  # 👈 needed for your ARS
}

class _SafeEval(ast.NodeVisitor):
    def __init__(self, names):
        self.names = names

    def visit_Expression(self, node): return self.visit(node.body)

    # Numbers
    def visit_Num(self, node): return float(node.n)
    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Only numeric constants allowed")

    # Variables
    def visit_Name(self, node):
        if node.id in self.names:
            return float(self.names[node.id])
        raise ValueError(f"Unknown variable: {node.id}")

    # a + b, a * b, ...
    def visit_BinOp(self, node):
        if type(node.op) not in _ALLOWED_BINOPS:
            raise ValueError("Operator not allowed")
        return _ALLOWED_BINOPS[type(node.op)](self.visit(node.left), self.visit(node.right))

    # -a, +a
    def visit_UnaryOp(self, node):
        if type(node.op) not in _ALLOWED_UNARY:
            raise ValueError("Unary operator not allowed")
        return _ALLOWED_UNARY[type(node.op)](self.visit(node.operand))

    # fn(...)
    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError("Function not allowed")
        fn = _ALLOWED_FUNCS[node.func.id]
        args = [self.visit(a) for a in node.args]
        return float(fn(*args))

    def generic_visit(self, node):
        raise ValueError("Unsupported syntax in formula")

def safe_eval(expr: str, names: dict) -> float:
    tree = ast.parse(expr, mode="eval")
    return float(_SafeEval(names).visit(tree))
