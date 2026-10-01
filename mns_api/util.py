"""Small shared pieces of the API: errors with hints, JSON that numpy can go into, value parsing and the
safe evaluation of the parameter expressions in design files."""
import ast
import json
import math
import operator
from pathlib import Path

import numpy as np


class ApiError(Exception):
    """An error the caller can act on: `hint` says what to do about it."""

    def __init__(self, message, hint=""):
        super().__init__(message)
        self.hint = hint


# -----------------------------
# JSON
# -----------------------------

def jsonable(x):
    """x with numpy arrays/scalars, tuples, Paths and non-finite floats made JSON-friendly."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    if hasattr(x, "item") and callable(x.item) and getattr(x, "shape", None) == ():
        return jsonable(x.item())
    return x


def write_json(path, data, indent=1):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(jsonable(data), indent=indent, ensure_ascii=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rounded(x, digits=5):
    """Numbers rounded to `digits` significant digits (short command output)."""
    if isinstance(x, dict):
        return {k: rounded(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v, digits) for v in x]
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if not math.isfinite(x) or x == 0.0:
            return x if math.isfinite(x) else None
        return float(f"{x:.{digits}g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


# -----------------------------
# Command-line values
# -----------------------------

def parse_range(text):
    """'from:to:points' (or a single value, or 'a,b,c') -> list of floats."""
    text = str(text).strip()
    try:
        if ":" in text:
            parts = text.split(":")
            if len(parts) != 3:
                raise ValueError
            lo, hi, n = float(parts[0]), float(parts[1]), int(parts[2])
            if n < 1:
                raise ValueError
            return np.linspace(lo, hi, n).tolist()
        if "," in text:
            return [float(v) for v in text.split(",")]
        return [float(text)]
    except ValueError:
        raise ApiError(f"Not a range: {text!r}", "Write from:to:points (e.g. 0:20:5), a list 0,5,10 or one value.")


def parse_assignment(text):
    """'name=value' -> (name, value) with the value as a number when it is one."""
    if "=" not in text:
        raise ApiError(f"Expected name=value, got {text!r}")
    name, value = text.split("=", 1)
    return name.strip(), parse_value(value.strip())


def parse_value(text):
    if not isinstance(text, str):
        return text
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(text) if text.lstrip("+-").isdigit() else float(text)
    except ValueError:
        return text


# -----------------------------
# Parameter expressions
# -----------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {name: getattr(math, name) for name in ("sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sqrt", "exp",
                                                  "log", "log10", "radians", "degrees", "hypot", "floor", "ceil")}
_FUNCS.update(abs=abs, min=min, max=max, round=round)
_CONSTANTS = {"pi": math.pi, "e": math.e}


def evaluate(expression, names):
    """A number from an expression of numbers, the parameters in `names`, + - * / ** %, and math functions
    (sin, cos, sqrt, ..., min, max, pi). Nothing else is allowed."""
    if isinstance(expression, (int, float)) and not isinstance(expression, bool):
        return float(expression)
    if not isinstance(expression, str):
        raise ApiError(f"Not a number or expression: {expression!r}")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise ApiError(f"Cannot read the expression {expression!r}",
                       "Use numbers, parameter names, + - * / ** and math functions such as sqrt(), sin(), pi.")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in names:
                return float(names[node.id])
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise ApiError(f"Unknown parameter {node.id!r} in {expression!r}",
                           "Define it under 'parameters' (above the ones that use it).")
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS \
                and not node.keywords:
            return float(_FUNCS[node.func.id](*[ev(a) for a in node.args]))
        raise ApiError(f"Not allowed in an expression: {ast.unparse(node)!r} (in {expression!r})")

    value = ev(tree)
    if not math.isfinite(value):
        raise ApiError(f"{expression!r} is not a finite number")
    return value


def evaluate_parameters(parameters):
    """{name: value} from the design file's parameters (each may use the ones above it)."""
    values = {}
    for name, expression in (parameters or {}).items():
        if not str(name).isidentifier():
            raise ApiError(f"Parameter name {name!r} is not a valid name", "Use letters, digits and _ (e.g. t_mem).")
        values[name] = evaluate(expression, values)
    return values


def as_number(x, names, what=""):
    try:
        return evaluate(x, names)
    except ApiError as exc:
        raise ApiError(f"{what}: {exc}" if what else str(exc), exc.hint)


def as_vector(x, names, n=3, what=""):
    if not isinstance(x, (list, tuple)) or len(x) != n:
        raise ApiError(f"{what or 'vector'}: expected a list of {n} numbers, got {x!r}")
    return [as_number(v, names, what) for v in x]
