"""
Neuron network (no GUI): characterised neurons used as blocks, Simulink style, and wired into a network that is
evaluated from the stored data alone - nothing is simulated here.

Components (the network's library) are lookup tables with the same layout as a stored characterisation:
  - a full neuron (*.mfn with a characterisation): every swept parameter is an axis, every recorded quantity a key;
  - a pre-activation-to-activation design (*.mad): one axis, the pre-activation Δp across its membrane, and its named
    outputs, tube area and mass flow as keys.
A component is copied into the network file when it is added (the source is not needed afterwards; `source_status`
says when the source was saved with a different dataset since, and `update_component` takes the new one).

Blocks: a neuron or valve block uses a component; each of its axes is either an input port (a signal wired from
another block; while unconnected it is held at the block's value for it) or a parameter (held at a set value), and
each chosen recorded key is an output port. The other blocks (TYPES) are sources and probes (Input = a network input,
set or swept; Constant; Output = a probe), math (sum, gain, product, functions, a free expression), activation
functions (sigmoid, tanh, ReLU, saturation, ...), logic and comparison (1 = true, 0 = false), a lookup table, and
subsystems: a block built from other blocks. A subsystem block is an instance of a definition (its own blocks and
wires, with In port / Out port blocks as its ports); every instance of a definition is the same (like a Simulink
library link), and a definition can be saved as a library block (*.mnb) for other networks. Evaluation flattens the
hierarchy first, so loops may cross subsystem boundaries.

Evaluation is static (every block is a steady-state map), multilinear on the component's grid (the same
interpolation as Characterisation.evaluate) and either extrapolates linearly or clamps to the characterised range
outside it (flagged). A connection is ideal: the receiving block does not load the sending one. That is exact for
a valve's outputs (a supplied gas pressure; an input chamber draws no flow at steady state) but not for a sealed
chamber's pressure (e.g. P:Pre-Activation) wired into another chamber, and a valve block fed from a neuron without
a linked design leaves out the valve membrane's compliance (characterise the linked full neuron for that).
Feedback loops are solved as algebraic loops (the outputs on the loop are unknowns, scipy root); sweeps warm-start
each point from the previous one, so a bistable loop follows one branch (hysteresis).
"""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SUFFIX = ".mnn"
FILE_TYPE = "neuron_network"
FILTER = f"Neuron network (*{SUFFIX})"

# what a block's variable is: an input port (a wired signal), a parameter (held at a value that training may change,
# within its bounds) or a constant (held at a value, never trained)
ROLE_INPUT, ROLE_PARAMETER, ROLE_CONSTANT = "input", "parameter", "constant"
ROLES = (ROLE_INPUT, ROLE_PARAMETER, ROLE_CONSTANT)
EXTRAPOLATE, CLAMP = "extrapolate", "clamp"
CURRENT, OUTDATED, MISSING = "current", "out of date", "not found"
LOOP_TOLERANCE = 1e-8     # algebraic loops: residual (in the signals' units)


# -----------------------------
# Components (lookup tables)
# -----------------------------

def _axis_name(axis):
    parts = [axis.get("part")] + [p for p in axis.get("also") or () if p != axis.get("part")]
    return f"{'+'.join(str(p) for p in parts)}.{axis['field']}"


def _sha1(data) -> str:
    return hashlib.sha1(json.dumps(data, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _table_of_characterisation(c) -> dict:
    """The parts of a stored characterisation a block needs (no deformed shapes)."""
    axes = [{"name": _axis_name(a), "label": a.get("label") or _axis_name(a), "unit": a.get("unit", ""),
             "values": [float(v) for v in a["values"]]} for a in c["axes"]]
    return {"axes": axes, "keys": [dict(k) for k in c["keys"]], "values": copy.deepcopy(c["values"]),
            "converged": list(c["converged"]), "extrapolated": list(c.get("extrapolated") or [False] * len(c["converged"])),
            "complete": bool(c.get("complete", True))}


def component_from_mfn(path) -> dict:
    """A full neuron's stored characterisation as a network component."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "full_neuron":
        raise ValueError(f"{path.name} is not a full-neuron file (*.mfn).")
    c = data.get("characterisation")
    if not c:
        raise ValueError(f"{path.name} has no stored characterisation: run it in the Full neuron tab first.")
    table = _table_of_characterisation(c)
    if not table["keys"] or not any(v is not None for v in table["values"][table["keys"][0]["key"]]):
        raise ValueError(f"{path.name}'s characterisation has no solved points.")
    return dict(table, kind=NEURON, name=path.stem, source=str(path.resolve()), sha1=source_sha1(NEURON, path),
                design=(c.get("design") or {}).get("file"))


def component_from_design(design, name, source=None, sha1=None) -> dict:
    """A simulated valve (app.activation.ActivationDesign) as a one-axis network component."""
    keys, values = [], {}
    for n in design.output_names:
        keys.append({"key": f"out:{n}", "label": f"output {n}", "unit": "kPa"})
        values[f"out:{n}"] = [float(v) for v in design.p_out[n]]
    for key, label, unit, v in (("area", "tube area", "mm2", design.A), ("mdot", "mass flow", "kg/s", design.mdot),
                                ("p_end", "tube-end pressure", "kPa", design.p_end)):
        keys.append({"key": key, "label": label, "unit": unit})
        values[key] = [float(x) for x in v]
    if design.volume is not None:
        keys.append({"key": "volume", "label": "membrane swept volume", "unit": "mm3"})
        values["volume"] = [float(x) for x in design.volume]
    n = len(design.dp)
    return {"kind": VALVE, "name": name, "source": source, "sha1": sha1,
            "axes": [{"name": "dp", "label": "pre-activation Δp (across the membrane)", "unit": "kPa",
                      "values": [float(x) for x in design.dp]}],
            "keys": keys, "values": values, "converged": [True] * n, "extrapolated": [False] * n, "complete": True}


def component_from_mad(path) -> dict:
    """A simulated pre-activation-to-activation design (*.mad) as a network component (loads the simulator)."""
    from .activation import ActivationDesign     # torch: only when a valve is added
    path = Path(path)
    return component_from_design(ActivationDesign.load(path), path.stem, str(path.resolve()), source_sha1(VALVE, path))


def component_from_file(path) -> dict:
    lower = str(path).lower()
    if lower.endswith(".mfn"):
        return component_from_mfn(path)
    if lower.endswith(".mad"):
        return component_from_mad(path)
    raise ValueError(f"{Path(path).name}: add a characterised full neuron (*.mfn) or a valve design (*.mad).")


def source_sha1(kind, path) -> str:
    """What decides whether a source changed: a neuron's characterisation (without shapes), a valve's file bytes."""
    path = Path(path)
    if kind == VALVE:
        return hashlib.sha1(path.read_bytes()).hexdigest()
    c = json.loads(path.read_text(encoding="utf-8")).get("characterisation") or {}
    return _sha1({k: v for k, v in c.items() if k not in ("shapes", "created", "seconds")})


_status_cache = {}


def source_status(component) -> str:
    """current / out of date / not found (cached per file modification time and size)."""
    source = component.get("source")
    if not source or not Path(source).exists():
        return MISSING
    st = Path(source).stat()
    key = (source, st.st_mtime_ns, st.st_size)
    if key not in _status_cache:
        try:
            _status_cache[key] = source_sha1(component["kind"], source)
        except Exception:
            return MISSING
    return CURRENT if _status_cache[key] == component.get("sha1") else OUTDATED


class Table:
    """Multilinear lookup in a component (as Characterisation.evaluate; axes of one value are fixed)."""

    def __init__(self, component):
        from scipy.interpolate import RegularGridInterpolator
        self.component = component
        self.axes = component["axes"]
        self.names = [a["name"] for a in self.axes]
        self.values = [np.asarray(a["values"], float) for a in self.axes]
        self.shape = tuple(len(v) for v in self.values)
        self.keys = [k["key"] for k in component["keys"]]
        self.units = {k["key"]: k.get("unit", "") for k in component["keys"]}
        self.live = [k for k, v in enumerate(self.values) if len(v) > 1]
        self.lo = np.array([v.min() for v in self.values])
        self.hi = np.array([v.max() for v in self.values])
        order = [np.argsort(self.values[k]) for k in self.live]
        self._interp = {}
        for key in self.keys:
            g = np.asarray([math.nan if v is None else v for v in component["values"][key]], float).reshape(self.shape)
            g = g[tuple(slice(None) if k in self.live else 0 for k in range(len(self.shape)))]
            if not self.live:
                self._interp[key] = float(g)
                continue
            for d, o in enumerate(order):
                g = np.take(g, o, axis=d)
            self._interp[key] = RegularGridInterpolator([self.values[k][o] for k, o in zip(self.live, order)], g,
                                                        bounds_error=False, fill_value=None)

    def evaluate(self, x, keys, clamp=False):
        """{key: value} at x (one value per axis) and the names of the axes outside the characterised range."""
        x = np.asarray(x, float)
        tol = 1e-9 * np.maximum(1.0, np.maximum(abs(self.lo), abs(self.hi)))
        out_of_range = [self.names[k] for k in range(len(x)) if x[k] < self.lo[k] - tol[k] or x[k] > self.hi[k] + tol[k]]
        if clamp:
            x = np.clip(x, self.lo, self.hi)
        point = x[self.live]
        out = {}
        for key in keys:
            f = self._interp[key]
            out[key] = f if isinstance(f, float) else float(f(point)[0])
        return out, out_of_range


# -----------------------------
# Block types
# -----------------------------

def _p(key, label, default, kind="float", choices=None, tip=""):
    """A block parameter: kind float | int | text | choice | list (comma-separated numbers)."""
    return {"key": key, "label": label, "default": default, "kind": kind, "choices": choices, "tip": tip}


class BlockType:
    """What a kind of block is: its library group, parameters, ports and function.
    fn(block, u) -> output value, with u the input values in port order (math, logic, ...)."""

    def __init__(self, kind, title, group, params=(), inputs=1, fn=None, symbol=None, labels=None, curve=None, tip=""):
        self.kind, self.title, self.group, self.params = kind, title, group, list(params)
        self._inputs, self.fn, self._symbol, self.labels, self.curve, self.tip = inputs, fn, symbol, labels, curve, tip

    def inputs(self, block):
        n = self._inputs(block) if callable(self._inputs) else self._inputs
        return ["in"] if n == 1 else [f"in{i + 1}" for i in range(n)]

    def symbol(self, block):
        return self._symbol(block) if callable(self._symbol) else (self._symbol or "")


def _n(block, key="inputs", lo=1, hi=24):
    try:
        return min(max(int(block.get(key, 2)), lo), hi)
    except (TypeError, ValueError):
        return 2


def _numbers(text):
    """'0, 1, 2.5' -> [0.0, 1.0, 2.5] (also accepts a list)."""
    if isinstance(text, (list, tuple)):
        return [float(v) for v in text]
    return [float(v) for v in str(text).replace(";", ",").split(",") if v.strip()]


def _truth(block, v):
    return v > float(block.get("threshold", 0.5))


def _sigmoid(b, u):
    z = -float(b["k"]) * (u[0] - float(b["x0"]))
    s = 1.0 / (1.0 + math.exp(z)) if z < 700 else 0.0
    return float(b["low"]) + (float(b["high"]) - float(b["low"])) * s


def _lookup(b, u):
    x, y = _numbers(b["x"]), _numbers(b["y"])
    if len(x) != len(y) or len(x) < 2:
        return math.nan
    order = np.argsort(x)
    return float(np.interp(u[0], np.asarray(x)[order], np.asarray(y)[order]))


def _safe(f):
    """A math function that returns NaN instead of raising (log of a negative number, ...)."""
    def g(b, u):
        try:
            v = f(b, u)
            return float(v) if v is not None else math.nan
        except (ValueError, OverflowError, ZeroDivisionError, TypeError):
            return math.nan
    return g


_COMPARE = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
            "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
_FUNCTIONS = {name: getattr(np, name) for name in ("exp", "log", "log10", "sqrt", "sin", "cos", "tan", "tanh", "sinh",
                                                   "cosh", "arctan", "abs", "sign", "floor", "ceil", "minimum",
                                                   "maximum", "clip", "where", "hypot")}
_FUNCTIONS.update(pi=math.pi, e=math.e, min=min, max=max, abs=abs, sigmoid=lambda x: 1.0 / (1.0 + np.exp(-x)))
_compiled = {}


def _expression(b, u):
    """The Function block: an expression in u1, u2, ... (and u, the list), numpy functions and pi, e."""
    text = str(b.get("expr", "u1"))
    if text not in _compiled:
        _compiled[text] = compile(text, "<function block>", "eval")
    names = dict(_FUNCTIONS, u=list(u), **{f"u{i + 1}": v for i, v in enumerate(u)})
    if len(u) == 1:
        names["x"] = u[0]
    return float(eval(_compiled[text], {"__builtins__": {}}, names))   # noqa: S307 (the user's own formula)


def _product(b, u):
    ops = str(b.get("ops", "**")) or "*"
    v = 1.0
    for op, x in zip(ops, u):
        v = v / x if op == "/" else v * x
    return v


def _switch(b, u):
    c = u[1]
    rule = b.get("criterion", "u2 >= threshold")
    t = float(b.get("threshold", 0.0))
    on = c >= t if rule == "u2 >= threshold" else c > t if rule == "u2 > threshold" else c != 0
    return u[0] if on else u[2]


SOURCES, MATH, ACTIVATION, LOGIC, LOOKUP, SUBSYSTEMS = ("Sources and probes", "Math", "Activation functions",
                                                        "Logic and comparison", "Lookup", "Subsystems")
GROUPS = (SOURCES, MATH, ACTIVATION, LOGIC, LOOKUP, SUBSYSTEMS)
NEURON, VALVE = "neuron", "valve"
INPUT, CONSTANT, OUTPUT, SUM, GAIN = "input", "constant", "output", "sum", "gain"
INPORT, OUTPORT, SUBSYSTEM = "inport", "outport", "subsystem"
OPS = list(_COMPARE)

TYPES = {}


def _register(t):
    TYPES[t.kind] = t
    return t


for _t in (
    # sources and probes
    BlockType(INPUT, "Input", SOURCES, [_p("value", "Value", 0.0), _p("unit", "Unit", "kPa", "text")], inputs=0,
              tip="A network input: set on the Run tab or swept"),
    BlockType(CONSTANT, "Constant", SOURCES, [_p("value", "Value", 0.0), _p("unit", "Unit", "kPa", "text")], inputs=0),
    BlockType(OUTPUT, "Output", SOURCES, tip="A probe: shown and plotted"),
    # math
    BlockType(SUM, "Sum", MATH, [_p("signs", "Signs", "++", "text", tip="One sign per input, e.g. +-+")],
              inputs=lambda b: max(len(str(b.get("signs") or "+")), 1),
              fn=lambda b, u: sum((-1.0 if s == "-" else 1.0) * x for s, x in zip(str(b.get("signs") or "+"), u)),
              symbol=lambda b: "Σ " + str(b.get("signs", ""))),
    BlockType(GAIN, "Gain", MATH, [_p("k", "Gain k", 1.0)], fn=lambda b, u: float(b["k"]) * u[0],
              symbol=lambda b: f"× {float(b['k']):.4g}"),
    BlockType("bias", "Bias", MATH, [_p("b", "Bias b", 0.0)], fn=lambda b, u: u[0] + float(b["b"]),
              symbol=lambda b: f"u {'+' if float(b['b']) >= 0 else '−'} {abs(float(b['b'])):.4g}"),
    BlockType("product", "Product", MATH, [_p("ops", "Operations", "**", "text", tip="One per input: * multiplies, "
                                                                                     "/ divides, e.g. **/")],
              inputs=lambda b: max(len(str(b.get("ops") or "*")), 1), fn=_safe(_product),
              symbol=lambda b: "Π " + str(b.get("ops", ""))),
    BlockType("abs", "Abs", MATH, fn=lambda b, u: abs(u[0]), symbol="|u|"),
    BlockType("sign", "Sign", MATH, fn=lambda b, u: float(np.sign(u[0])), symbol="sign(u)"),
    BlockType("sqrt", "Square root", MATH, fn=_safe(lambda b, u: math.sqrt(u[0])), symbol="√u"),
    BlockType("power", "Power", MATH, [_p("n", "Exponent n", 2.0)], fn=_safe(lambda b, u: u[0] ** float(b["n"])),
              symbol=lambda b: f"u^{float(b['n']):.4g}"),
    BlockType("exp", "Exp", MATH, fn=_safe(lambda b, u: math.exp(u[0])), symbol="exp(u)"),
    BlockType("log", "Log", MATH, fn=_safe(lambda b, u: math.log(u[0])), symbol="ln(u)"),
    BlockType("min", "Min", MATH, [_p("inputs", "Inputs", 2, "int")], inputs=_n, fn=lambda b, u: min(u),
              symbol="min"),
    BlockType("max", "Max", MATH, [_p("inputs", "Inputs", 2, "int")], inputs=_n, fn=lambda b, u: max(u),
              symbol="max"),
    BlockType("function", "Function", MATH, [_p("inputs", "Inputs", 1, "int"),
                                             _p("expr", "f(u1, u2, …) =", "u1", "text",
                                                tip="e.g. 20*sigmoid(0.3*(u1 - 5)) or where(u1 > u2, u1, 0); numpy "
                                                    "functions, pi and e may be used")],
              inputs=lambda b: _n(b, lo=1), fn=_safe(_expression), symbol=lambda b: str(b.get("expr", "")),
              labels=lambda b, p: "u" + (p[2:] if p.startswith("in") and len(p) > 2 else "1")),
    # activation functions
    BlockType("sigmoid", "Sigmoid", ACTIVATION, [_p("k", "Slope k (1/unit)", 0.5), _p("x0", "Centre x0", 0.0),
                                                 _p("low", "Low level", 0.0), _p("high", "High level", 1.0)],
              fn=_sigmoid, symbol="sigmoid",
              curve=lambda b: (float(b["x0"]) - 6 / max(abs(float(b["k"])), 1e-9),
                               float(b["x0"]) + 6 / max(abs(float(b["k"])), 1e-9)),
              tip="low + (high - low) / (1 + exp(-k (u - x0)))"),
    BlockType("tanh", "Tanh", ACTIVATION, [_p("k", "Slope k (1/unit)", 0.5), _p("x0", "Centre x0", 0.0),
                                           _p("amplitude", "Amplitude A", 1.0)],
              fn=lambda b, u: float(b["amplitude"]) * math.tanh(float(b["k"]) * (u[0] - float(b["x0"]))),
              symbol="tanh", curve=lambda b: (float(b["x0"]) - 3 / max(abs(float(b["k"])), 1e-9),
                                              float(b["x0"]) + 3 / max(abs(float(b["k"])), 1e-9)),
              tip="A tanh(k (u - x0))"),
    BlockType("relu", "ReLU", ACTIVATION, [_p("threshold", "Threshold", 0.0), _p("slope", "Slope", 1.0)],
              fn=lambda b, u: float(b["slope"]) * max(u[0] - float(b["threshold"]), 0.0), symbol="ReLU",
              curve=lambda b: (float(b["threshold"]) - 10, float(b["threshold"]) + 10),
              tip="slope · max(u - threshold, 0)"),
    BlockType("saturation", "Saturation", ACTIVATION, [_p("lower", "Lower limit", 0.0), _p("upper", "Upper limit", 1.0)],
              fn=lambda b, u: min(max(u[0], float(b["lower"])), float(b["upper"])), symbol="sat",
              curve=lambda b: (2 * float(b["lower"]) - float(b["upper"]), 2 * float(b["upper"]) - float(b["lower"]))),
    BlockType("deadzone", "Dead zone", ACTIVATION, [_p("start", "Start", -1.0), _p("end", "End", 1.0)],
              fn=lambda b, u: (u[0] - float(b["end"]) if u[0] > float(b["end"]) else
                               u[0] - float(b["start"]) if u[0] < float(b["start"]) else 0.0), symbol="dead zone",
              curve=lambda b: (2 * float(b["start"]) - float(b["end"]), 2 * float(b["end"]) - float(b["start"]))),
    BlockType("step", "Step", ACTIVATION, [_p("threshold", "Threshold", 0.0), _p("low", "Below", 0.0),
                                           _p("high", "At or above", 1.0)],
              fn=lambda b, u: float(b["high"]) if u[0] >= float(b["threshold"]) else float(b["low"]), symbol="step",
              curve=lambda b: (float(b["threshold"]) - 10, float(b["threshold"]) + 10)),
    # logic and comparison
    BlockType("compare", "Compare", LOGIC, [_p("op", "u1 ? u2", ">", "choice", OPS)], inputs=2,
              fn=lambda b, u: 1.0 if _COMPARE[b["op"]](u[0], u[1]) else 0.0, symbol=lambda b: f"u1 {b['op']} u2",
              labels=lambda b, p: {"in1": "u1", "in2": "u2"}[p]),
    BlockType("compare_const", "Compare to constant", LOGIC, [_p("op", "u ? c", ">", "choice", OPS),
                                                              _p("c", "Constant c", 0.0)],
              fn=lambda b, u: 1.0 if _COMPARE[b["op"]](u[0], float(b["c"])) else 0.0,
              symbol=lambda b: f"u {b['op']} {float(b['c']):.4g}"),
    BlockType("and", "AND", LOGIC, [_p("inputs", "Inputs", 2, "int"), _p("threshold", "True above", 0.5)], inputs=_n,
              fn=lambda b, u: 1.0 if all(_truth(b, x) for x in u) else 0.0, symbol="AND"),
    BlockType("or", "OR", LOGIC, [_p("inputs", "Inputs", 2, "int"), _p("threshold", "True above", 0.5)], inputs=_n,
              fn=lambda b, u: 1.0 if any(_truth(b, x) for x in u) else 0.0, symbol="OR"),
    BlockType("xor", "XOR", LOGIC, [_p("inputs", "Inputs", 2, "int"), _p("threshold", "True above", 0.5)], inputs=_n,
              fn=lambda b, u: 1.0 if sum(_truth(b, x) for x in u) % 2 == 1 else 0.0, symbol="XOR"),
    BlockType("nand", "NAND", LOGIC, [_p("inputs", "Inputs", 2, "int"), _p("threshold", "True above", 0.5)],
              inputs=_n, fn=lambda b, u: 0.0 if all(_truth(b, x) for x in u) else 1.0, symbol="NAND"),
    BlockType("nor", "NOR", LOGIC, [_p("inputs", "Inputs", 2, "int"), _p("threshold", "True above", 0.5)], inputs=_n,
              fn=lambda b, u: 0.0 if any(_truth(b, x) for x in u) else 1.0, symbol="NOR"),
    BlockType("not", "NOT", LOGIC, [_p("threshold", "True above", 0.5)],
              fn=lambda b, u: 0.0 if _truth(b, u[0]) else 1.0, symbol="NOT"),
    BlockType("switch", "Switch", LOGIC, [_p("criterion", "Pass u1 when", "u2 >= threshold", "choice",
                                             ["u2 >= threshold", "u2 > threshold", "u2 != 0"]),
                                          _p("threshold", "Threshold", 0.5)], inputs=3, fn=_switch, symbol="switch",
              labels=lambda b, p: {"in1": "u1 (pass)", "in2": "u2 (control)", "in3": "u3 (else)"}[p]),
    # lookup
    BlockType("lookup", "Lookup table", LOOKUP, [_p("x", "Breakpoints x", "0, 10, 20", "list"),
                                                 _p("y", "Values y", "0, 5, 20", "list")], fn=_lookup, symbol="lookup",
              curve=lambda b: (min(_numbers(b["x"]) or [0]), max(_numbers(b["x"]) or [1])),
              tip="Linear between the breakpoints, held at the end values outside"),
    # subsystems
    BlockType(INPORT, "In port", SUBSYSTEMS, inputs=0, tip="Inside a subsystem: one of its input ports"),
    BlockType(OUTPORT, "Out port", SUBSYSTEMS, tip="Inside a subsystem: one of its output ports"),
    BlockType(SUBSYSTEM, "Subsystem", SUBSYSTEMS, inputs=0, tip="A block built from other blocks"),
    BlockType(NEURON, "Neuron", "Neurons"),
    BlockType(VALVE, "Valve", "Valves"),
):
    _register(_t)

TYPE_NAMES = {k: t.title for k, t in TYPES.items()}
FUNCTION_TYPES = {k for k, t in TYPES.items() if t.fn is not None}      # math, activation, logic, lookup


def new_block(kind, component=None, name="", pos=(0.0, 0.0)) -> dict:
    if kind not in TYPES:
        raise ValueError(f"Unknown block type {kind}")
    block = {"id": "", "type": kind, "name": name, "pos": [float(pos[0]), float(pos[1])]}
    for p in TYPES[kind].params:
        block[p["key"]] = p["default"]
    if kind in (NEURON, VALVE):
        block["component"] = None
        block["axes"] = {}
        block["outputs"] = []
        block["range"] = CLAMP if kind == VALVE else EXTRAPOLATE
        if component is not None:
            set_component(block, component)
    elif kind == INPUT:
        block["sweep"] = {"from": -20.0, "to": 20.0, "points": 41}
    elif kind == SUBSYSTEM:
        block["definition"] = None
    return block


def default_outputs(component):
    """A valve's named outputs, otherwise a neuron's design outputs or its closed chambers' pressures."""
    keys = [k["key"] for k in component["keys"]]
    outs = [k for k in keys if k.startswith("out:")]
    if outs:
        return outs
    pressures = [k for k in keys if k.startswith("P:")]
    return pressures[:1] or keys[:1]


def set_component(block, component):
    """Use `component` (a library entry with an "id"): pressure axes become input ports, the rest constants held at
    their middle simulated value; settings of axes and outputs of the same name are kept."""
    old_axes, old_outputs = block.get("axes") or {}, block.get("outputs") or []
    block["component"] = component.get("id")
    axes = {}
    for a in component["axes"]:
        values = sorted(a["values"])
        if a["name"] in old_axes:
            axes[a["name"]] = dict(old_axes[a["name"]])
            continue
        is_input = a.get("unit") == "kPa" and len(values) > 1
        mid = values[len(values) // 2]
        axes[a["name"]] = {"role": ROLE_INPUT if is_input else ROLE_CONSTANT,
                           "value": 0.0 if is_input and values[0] <= 0.0 <= values[-1] else float(mid)}
    block["axes"] = axes
    keys = [k["key"] for k in component["keys"]]
    kept = [k for k in old_outputs if k in keys]
    block["outputs"] = kept or default_outputs(component)


def short_label(text):
    """A port label: 'IntermediateChamber1.pressure' -> 'IntermediateChamber1', 'P:Pre-Activation' -> 'P Pre-Activation'."""
    if text.endswith(".pressure"):
        return text[:-len(".pressure")]
    return text.replace(":", " ", 1) if ":" in text else text


def _ordered_ports(definition, kind):
    """A subsystem's port blocks of one kind in port order (their "index", then their height on the canvas)."""
    blocks = [b for b in definition["blocks"] if b["type"] == kind]
    return sorted(blocks, key=lambda b: (b.get("index", 1e9), b["pos"][1], b["id"]))


# -----------------------------
# Network
# -----------------------------

class Network:
    """The top level ("root": blocks and wires), the library (components; subsystem definitions, each with its own
    blocks and wires) and the sweep settings. A container is the root or a definition: {"blocks", "wires"}."""

    def __init__(self):
        self.components = {}   # id -> component
        self.definitions = {}  # id -> {"name", "blocks", "wires"}
        self.root = {"blocks": [], "wires": []}
        self.sweep = {"x": None, "y": None, "y_from": -20.0, "y_to": 20.0, "y_points": 5}
        self.path = None
        self._tables = {}

    @property
    def blocks(self):
        return self.root["blocks"]

    @blocks.setter
    def blocks(self, value):
        self.root["blocks"] = value

    @property
    def wires(self):
        return self.root["wires"]

    @wires.setter
    def wires(self, value):
        self.root["wires"] = value

    def containers(self):
        return [self.root] + list(self.definitions.values())

    # --- library: components ---

    def add_component(self, component) -> str:
        """Add a component (or return the id of the same dataset already in the library)."""
        for cid, c in self.components.items():
            if c.get("sha1") and c.get("sha1") == component.get("sha1") and c.get("source") == component.get("source"):
                return cid
        cid = self._new_id("c", self.components)
        component = dict(component, id=cid)
        names = {c["name"] for c in self.components.values()}
        base, n = component["name"], 2
        while component["name"] in names:
            component["name"] = f"{base} ({n})"
            n += 1
        self.components[cid] = component
        return cid

    def _all_blocks(self):
        for container in self.containers():
            yield from container["blocks"]

    def remove_component(self, cid):
        if any(b.get("component") == cid for b in self._all_blocks()):
            raise ValueError("The component is used by a block: delete those blocks first.")
        self.components.pop(cid, None)
        self._tables.pop(cid, None)

    def update_component(self, cid):
        """Read the component's source again (blocks keep their port settings where the axes and keys remain)."""
        old = self.components[cid]
        new = component_from_file(old["source"])
        new.update(id=cid, name=old["name"])
        self.components[cid] = new
        self._tables.pop(cid, None)
        for block in self._all_blocks():
            if block.get("component") == cid:
                block["axes"] = {n: s for n, s in block["axes"].items() if n in {a["name"] for a in new["axes"]}}
                set_component(block, new)
        self.prune_wires()

    def table(self, cid) -> Table:
        if cid not in self._tables:
            self._tables[cid] = Table(self.components[cid])
        return self._tables[cid]

    def component_of(self, block):
        return self.components.get(block.get("component")) if block["type"] in (NEURON, VALVE) else None

    # --- library: subsystem definitions ---

    def new_definition(self, name="Subsystem", blocks=None, wires=None) -> str:
        did = self._new_id("d", self.definitions)
        names = {d["name"] for d in self.definitions.values()}
        base, n = name, 2
        while name in names:
            name = f"{base} {n}"
            n += 1
        self.definitions[did] = {"name": name, "blocks": blocks or [], "wires": wires or []}
        return did

    def definition_of(self, block):
        return self.definitions.get(block.get("definition")) if block["type"] == SUBSYSTEM else None

    def uses(self, did, target) -> bool:
        """Whether definition `did` contains an instance of `target` (directly or deeper)."""
        if did == target:
            return True
        d = self.definitions.get(did)
        return bool(d) and any(b["type"] == SUBSYSTEM and self.uses(b.get("definition"), target)
                               for b in d["blocks"])

    def instances(self, did):
        return [b for b in self._all_blocks() if b["type"] == SUBSYSTEM and b.get("definition") == did]

    def remove_definition(self, did):
        if self.instances(did):
            raise ValueError("The subsystem is used by a block: delete those blocks first.")
        self.definitions.pop(did, None)

    def duplicate_definition(self, did) -> str:
        d = self.definitions[did]
        return self.new_definition(d["name"] + " copy", copy.deepcopy(d["blocks"]), copy.deepcopy(d["wires"]))

    # --- blocks and wires ---

    @staticmethod
    def _new_id(prefix, existing):
        n = 1
        while f"{prefix}{n}" in existing:
            n += 1
        return f"{prefix}{n}"

    def block(self, bid, container=None):
        return next((b for b in (container or self.root)["blocks"] if b["id"] == bid), None)

    def add_block(self, kind, cid=None, pos=(0.0, 0.0), name=None, container=None, definition=None) -> dict:
        """A new block in `container` (default: the top level). A subsystem block uses `definition` (a new empty one
        when None); `container` must not be that definition or one it uses (no recursion)."""
        container = container or self.root
        component = self.components[cid] if cid else None
        block = new_block(kind, component, pos=pos)
        if kind == SUBSYSTEM:
            if definition is None:
                definition = self.new_definition(name or "Subsystem")
            here = next((did for did, d in self.definitions.items() if d is container), None)
            if here is not None and self.uses(definition, here):
                raise ValueError("A subsystem cannot contain itself.")
            block["definition"] = definition
        block["id"] = self._new_id("b", {b["id"] for b in container["blocks"]})
        if kind in (INPORT, OUTPORT):
            block["index"] = len([b for b in container["blocks"] if b["type"] == kind]) + 1
        base = name or (component["name"] if component else self.definitions[definition]["name"] if kind == SUBSYSTEM
                        else TYPE_NAMES[kind])
        block["name"] = self.unique_name(base, container)
        container["blocks"].append(block)
        return block

    @staticmethod
    def unique_name(base, container, exclude=None):
        names, n, name = {b["name"] for b in container["blocks"] if b is not exclude}, 1, base
        while name in names:          # D1, D1 2, D1 3, ...
            n += 1
            name = f"{base} {n}"
        return name

    def remove_block(self, bid, container=None):
        container = container or self.root
        container["blocks"] = [b for b in container["blocks"] if b["id"] != bid]
        container["wires"] = [w for w in container["wires"] if w["from"][0] != bid and w["to"][0] != bid]
        self.prune_wires()     # a removed port block changes the ports of its subsystem's instances

    def ports(self, block):
        """(input ports, output ports) of a block as lists of port keys."""
        kind = block["type"]
        if kind in (NEURON, VALVE):
            component = self.component_of(block)
            order = [a["name"] for a in component["axes"]] if component else list(block.get("axes", {}))
            ins = [n for n in order if block["axes"].get(n, {}).get("role") == ROLE_INPUT]
            return ins, list(block.get("outputs") or [])
        if kind == SUBSYSTEM:
            d = self.definition_of(block)
            if d is None:
                return [], []
            return [b["id"] for b in _ordered_ports(d, INPORT)], [b["id"] for b in _ordered_ports(d, OUTPORT)]
        if kind in (INPUT, CONSTANT, INPORT):
            return [], ["out"]
        if kind in (OUTPUT, OUTPORT):
            return ["in"], []
        return TYPES[kind].inputs(block), ["out"]

    def port_label(self, block, port, output=True):
        kind = block["type"]
        if kind in (NEURON, VALVE):
            return short_label(port)
        if kind == SUBSYSTEM:
            d = self.definition_of(block)
            inner = self.block(port, d) if d else None
            return inner["name"] if inner else port
        labels = TYPES[kind].labels
        return labels(block, port) if labels and not output else ""

    def connect(self, src, dst, container=None):
        """Wire output port src (block id, port) to input port dst; an input takes one wire (replaced)."""
        container = container or self.root
        sb, db = self.block(src[0], container), self.block(dst[0], container)
        if sb is None or db is None or src[1] not in self.ports(sb)[1] or dst[1] not in self.ports(db)[0]:
            raise ValueError("Connect an output port to an input port.")
        container["wires"] = [w for w in container["wires"] if w["to"] != list(dst)]
        wire = {"from": list(src), "to": list(dst)}
        container["wires"].append(wire)
        return wire

    def disconnect(self, wire, container=None):
        container = container or self.root
        container["wires"] = [w for w in container["wires"] if w is not wire and w != wire]

    def prune_wires(self):
        """Drop wires whose ports no longer exist (a port changed role, an output or a port block was removed)."""
        removed = 0
        for container in self.containers():
            keep = []
            for w in container["wires"]:
                sb, db = self.block(w["from"][0], container), self.block(w["to"][0], container)
                if sb and db and w["from"][1] in self.ports(sb)[1] and w["to"][1] in self.ports(db)[0]:
                    keep.append(w)
            removed += len(container["wires"]) - len(keep)
            container["wires"] = keep
        return removed

    def unit(self, block, port, output=True):
        """The unit of a port ("" when it takes any)."""
        kind = block["type"]
        if kind in (NEURON, VALVE):
            c = self.component_of(block)
            if c is None:
                return ""
            if output:
                return next((k.get("unit", "") for k in c["keys"] if k["key"] == port), "")
            return next((a.get("unit", "") for a in c["axes"] if a["name"] == port), "")
        if kind in (INPUT, CONSTANT):
            return block.get("unit", "")
        if kind == SUBSYSTEM and output:      # what feeds the out port inside (followed into nested subsystems)
            d = self.definition_of(block)
            w = next((w for w in (d["wires"] if d else []) if w["to"] == [port, "in"]), None)
            src = self.block(w["from"][0], d) if w else None
            if src is not None and src is not block:
                return self.unit(src, w["from"][1])
        return ""

    def unit_warnings(self, container=None):
        container = container or self.root
        out = []
        for w in container["wires"]:
            sb, db = self.block(w["from"][0], container), self.block(w["to"][0], container)
            u, v = self.unit(sb, w["from"][1]), self.unit(db, w["to"][1], output=False)
            if u and v and u != v:
                out.append(f"{sb['name']}.{short_label(w['from'][1])} ({u}) → {db['name']}.{short_label(w['to'][1])} "
                           f"({v}): units differ")
        return out

    def group(self, block_ids, container=None, name="Subsystem", pos=None) -> dict:
        """Move blocks into a new subsystem (Simulink's "create subsystem"): wires inside the selection move with
        them, every outside source feeding the selection becomes an In port and every selected source feeding the
        outside an Out port. Returns the new subsystem block."""
        container = container or self.root
        chosen = [b for b in container["blocks"] if b["id"] in set(block_ids)]
        if not chosen:
            raise ValueError("Select the blocks to put in the subsystem.")
        if any(b["type"] in (INPORT, OUTPORT) for b in chosen):
            raise ValueError("Port blocks cannot be moved into a nested subsystem.")
        ids = {b["id"] for b in chosen}
        inner_wires = [w for w in container["wires"] if w["from"][0] in ids and w["to"][0] in ids]
        incoming = [w for w in container["wires"] if w["from"][0] not in ids and w["to"][0] in ids]
        outgoing = [w for w in container["wires"] if w["from"][0] in ids and w["to"][0] not in ids]
        xs = [b["pos"][0] for b in chosen]
        ys = [b["pos"][1] for b in chosen]
        did = self.new_definition(name, copy.deepcopy(chosen), copy.deepcopy(inner_wires))
        d = self.definitions[did]
        sources_in, sources_out = {}, {}
        for w in incoming:
            src = tuple(w["from"])
            if src not in sources_in:
                sb = self.block(src[0], container)
                label = sb["name"] + ("" if src[1] in ("out",) else f" {short_label(src[1])}")
                port = self.add_block(INPORT, pos=(min(xs) - 200, ys[0] + 70 * len(sources_in)), name=label,
                                      container=d)
                sources_in[src] = port
            self.connect((sources_in[src]["id"], "out"), tuple(w["to"]), d)
        for w in outgoing:
            src = tuple(w["from"])
            if src not in sources_out:
                sb = self.block(src[0], container)
                label = sb["name"] + ("" if src[1] in ("out",) else f" {short_label(src[1])}")
                port = self.add_block(OUTPORT, pos=(max(xs) + 360, ys[0] + 70 * len(sources_out)), name=label,
                                      container=d)
                self.connect(src, (port["id"], "in"), d)
                sources_out[src] = port
        container["blocks"] = [b for b in container["blocks"] if b["id"] not in ids]
        container["wires"] = [w for w in container["wires"] if w["from"][0] not in ids and w["to"][0] not in ids]
        sub = self.add_block(SUBSYSTEM, pos=pos or (sum(xs) / len(xs), sum(ys) / len(ys)), container=container,
                             definition=did, name=name)
        for src, port in sources_in.items():
            self.connect(src, (sub["id"], port["id"]), container)
        for w in outgoing:
            self.connect((sub["id"], sources_out[tuple(w["from"])]["id"]), tuple(w["to"]), container)
        return sub

    # --- evaluation ---

    def flatten(self):
        """The network with every subsystem expanded: (blocks, wires, names). Flat block ids are paths ("b3/b1" =
        block b1 inside subsystem instance b3); port blocks become pass-throughs; names = flat id -> path of names."""
        blocks, wires, names = [], [], {}

        def expand(container, prefix, name_prefix, depth):
            if depth > 32:
                raise ValueError("Subsystems are nested too deep (does one contain itself?).")
            subs = {}
            for b in container["blocks"]:
                fid = prefix + b["id"]
                names[fid] = name_prefix + b["name"]
                if b["type"] == SUBSYSTEM:
                    d = self.definition_of(b)
                    subs[b["id"]] = fid
                    if d is not None:
                        expand(d, fid + "/", names[fid] + "/", depth + 1)
                else:
                    blocks.append((fid, b))

            def source(bid, port):
                return (subs[bid] + "/" + port, "out") if bid in subs else (prefix + bid, port)

            def target(bid, port):
                return (subs[bid] + "/" + port, "in") if bid in subs else (prefix + bid, port)

            for w in container["wires"]:
                wires.append((source(*w["from"]), target(*w["to"])))

        expand(self.root, "", "", 0)
        return blocks, wires, names

    def parameters(self):
        """Every trainable parameter, once per place it is stored (a parameter inside a subsystem is shared by all its
        instances): [{"where" (top level or subsystem name), "block", "name" ("block.variable"), "kind" ("axis" for a
        neuron's or valve's variable, "setting" for a built-in block's), "key", "value", "min", "max", "unit"}].
        Neuron/valve variables are bounded by their characterised range, settings are unbounded (None). Change one
        with `set_parameter`."""
        out = []
        places = [("top level", self.root)] + [(d["name"], d) for d in self.definitions.values()]
        for where, container in places:
            for b in container["blocks"]:
                if b["type"] in (NEURON, VALVE):
                    c = self.component_of(b)
                    for a in (c["axes"] if c else []):
                        s = b["axes"].get(a["name"], {})
                        if s.get("role") == ROLE_PARAMETER:
                            out.append({"where": where, "block": b, "name": f"{b['name']}.{short_label(a['name'])}",
                                        "kind": "axis", "key": a["name"], "value": float(s.get("value", 0.0)),
                                        "min": float(min(a["values"])), "max": float(max(a["values"])),
                                        "unit": a.get("unit", "")})
                elif b["type"] in FUNCTION_TYPES:
                    labels = {q["key"]: q["label"] for q in TYPES[b["type"]].params}
                    for key in b.get("trainable") or []:
                        if key in labels:
                            out.append({"where": where, "block": b, "name": f"{b['name']}.{key}", "kind": "setting",
                                        "key": key, "label": labels[key], "value": float(b.get(key, 0.0)),
                                        "min": None, "max": None, "unit": ""})
        return out

    @staticmethod
    def set_parameter(parameter, value):
        """Set a trainable parameter (from `parameters`), kept within its bounds; returns the value set."""
        v = float(value)
        if parameter["min"] is not None:
            v = min(max(v, parameter["min"]), parameter["max"])
        if parameter["kind"] == "axis":
            parameter["block"]["axes"][parameter["key"]]["value"] = v
        else:
            parameter["block"][parameter["key"]] = v
        parameter["value"] = v
        return v

    def inputs(self):
        """[(path name, block)] of every Input block (inside subsystems too)."""
        blocks, _, names = self.flatten()
        return [(names[fid], b) for fid, b in blocks if b["type"] == INPUT]

    def outputs(self):
        """[(path name, flat id, block)] of every Output block (the probes)."""
        blocks, _, names = self.flatten()
        return [(names[fid], fid, b) for fid, b in blocks if b["type"] == OUTPUT]

    def _evaluate_block(self, fid, block, inputs, overrides, name, flags):
        kind = block["type"]
        if kind == INPUT:
            return {"out": float(overrides.get(name, block.get("value", 0.0)))}
        if kind == CONSTANT:
            return {"out": float(block.get("value", 0.0))}
        if kind in (OUTPUT, OUTPORT, INPORT):
            v = inputs.get("in", math.nan if kind != INPORT else 0.0)
            return {} if kind == OUTPUT else {"out": v}
        if kind in FUNCTION_TYPES:
            u = [inputs.get(p, 0.0) for p in TYPES[kind].inputs(block)]
            try:
                return {"out": float(TYPES[kind].fn(block, u))}
            except Exception as exc:
                flags[fid] = [f"error: {exc}"]
                return {"out": math.nan}
        table = self.table(block["component"])
        x = []
        for axis in table.names:
            setting = block["axes"].get(axis, {})
            x.append(inputs[axis] if setting.get("role") == ROLE_INPUT and axis in inputs
                     else float(setting.get("value", 0.0)))
        values, out = table.evaluate(x, block["outputs"], clamp=block.get("range") == CLAMP)
        if out:
            flags[fid] = out
        return values

    def solve(self, inputs=None, guess=None) -> dict:
        """Every port's value with the network's Input blocks at their values (or `inputs` {path name: value}).
        Returns {"values": {(flat block id, port): value} (a subsystem's ports too, under its own id), "probes":
        {output path name: value}, "flags": {flat id: axes outside range or errors}, "converged", "loop": [flat ids on
        or after algebraic loops], "residual"}."""
        overrides = inputs or {}
        blocks, wires, names = self.flatten()
        ports = {fid: self.ports(b) for fid, b in blocks}
        for fid, b in blocks:
            if b["type"] == INPORT:
                ports[fid] = (["in"], ["out"])
            elif b["type"] == OUTPORT:
                ports[fid] = (["in"], ["out"])
        feeds = {dst: src for src, dst in wires}
        deps = {fid: {feeds[(fid, p)][0] for p in ports[fid][0] if (fid, p) in feeds} for fid, _ in blocks}
        done, order = set(), []
        while True:
            ready = [(fid, b) for fid, b in blocks if fid not in done and deps[fid] <= done]
            if not ready:
                break
            for fid, b in ready:
                done.add(fid)
                order.append((fid, b))
        loop = [(fid, b) for fid, b in blocks if fid not in done]
        values = dict(guess or {})

        def run(items, flags):
            for fid, b in items:
                ins = {p: values.get(feeds[(fid, p)], 0.0) for p in ports[fid][0] if (fid, p) in feeds}
                for p, v in self._evaluate_block(fid, b, ins, overrides, names[fid], flags).items():
                    values[(fid, p)] = v

        flags, converged, residual = {}, True, 0.0
        run(order, flags)
        if loop:
            unknowns = [(fid, p) for fid, _ in loop for p in ports[fid][1]]
            x0 = np.array([values.get(u, 0.0) if np.isfinite(values.get(u, 0.0)) else 0.0 for u in unknowns])

            def residual_of(x):
                values.update(zip(unknowns, x))
                run(loop, {})
                return np.array([values[u] for u in unknowns]) - x

            def small(r, x):
                return np.all(np.isfinite(r)) and np.max(np.abs(r), initial=0.0) <= \
                    LOOP_TOLERANCE * max(1.0, np.max(np.abs(x), initial=0.0))

            from scipy.optimize import root
            x = root(residual_of, x0, method="hybr").x if unknowns else x0
            if not small(residual_of(x), x):
                x = x0.copy()            # damped fixed-point iteration (discontinuous blocks: logic, steps)
                for _ in range(500):
                    r = residual_of(x)
                    if small(r, x):
                        break
                    x = x + 0.5 * r
            values.update(zip(unknowns, x))
            run(loop, flags)
            r = np.array([values[u] for u in unknowns]) - x
            residual = float(np.max(np.abs(r), initial=0.0))
            converged = bool(np.isfinite(residual) and
                             residual <= 1e3 * LOOP_TOLERANCE * max(1.0, np.max(np.abs(x), initial=0.0)))
        # a subsystem's ports under its own id (for display)
        for fid, b in blocks:
            if b["type"] == OUTPORT and (fid, "out") in values:
                sub, _, port = fid.rpartition("/")
                values[(sub, port)] = values[(fid, "out")]
        probes = {}
        for fid, b in blocks:
            if b["type"] == OUTPUT:
                src = feeds.get((fid, "in"))
                v = values.get(src, math.nan) if src else math.nan
                values[(fid, "in")] = v
                probes[names[fid]] = v
        return {"values": values, "probes": probes, "flags": flags, "converged": converged,
                "loop": [fid for fid, _ in loop], "residual": residual}

    def run_sweep(self, x, x_values, y=None, y_values=None, progress=None) -> dict:
        """Solve on a grid of one or two Input blocks (by path name), warm-starting each point from the previous one
        (serpentine order). Returns {"x", "x_values", "y", "y_values", "probes": {name: array (ny, nx)},
        "converged", "flagged" (ny, nx) bool, "ports": {(block, port): array}, "cancelled"}."""
        xs = np.asarray(x_values, float)
        ys = np.asarray(y_values, float) if y else np.array([math.nan])
        shape = (len(ys), len(xs))
        probes = {name: np.full(shape, math.nan) for name, _, _ in self.outputs()}
        converged = np.zeros(shape, bool)
        flagged = np.zeros(shape, bool)
        port_values = {}
        guess = None
        fixed = {name: b.get("value", 0.0) for name, b in self.inputs()}
        result = {"x": x, "x_values": xs, "y": y, "y_values": ys if y else None, "probes": probes,
                  "converged": converged, "flagged": flagged, "ports": port_values, "cancelled": False}
        for j, yv in enumerate(ys):
            cols = range(len(xs)) if j % 2 == 0 else range(len(xs) - 1, -1, -1)
            for count, i in enumerate(cols):
                point = dict(fixed)
                point[x] = float(xs[i])
                if y:
                    point[y] = float(yv)
                r = self.solve(point, guess)
                guess = r["values"]
                for name, v in r["probes"].items():
                    probes[name][j, i] = v
                converged[j, i] = r["converged"]
                flagged[j, i] = bool(r["flags"])
                for key, v in r["values"].items():
                    port_values.setdefault(key, np.full(shape, math.nan))[j, i] = v
                if progress is not None and progress(j * len(xs) + count + 1, shape[0] * shape[1]) is False:
                    result["cancelled"] = True
                    return result
        return result

    # --- persistence ---

    def to_dict(self, path=None):
        components = {}
        for cid, c in self.components.items():
            c = dict(c)
            if c.get("source") and path:
                try:
                    c["source"] = str(Path(c["source"]).resolve().relative_to(Path(path).resolve().parent))
                except ValueError:
                    c["source"] = str(Path(c["source"]).resolve())
            components[cid] = c
        return {"type": FILE_TYPE, "version": 2, "components": components, "definitions": self.definitions,
                "blocks": self.blocks, "wires": self.wires, "sweep": self.sweep}

    def save(self, path):
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(path), indent=1), encoding="utf-8")
        self.path = str(path)

    @classmethod
    def from_dict(cls, data, folder=None):
        if data.get("type") != FILE_TYPE:
            raise ValueError("Not a neuron-network file.")
        net = cls()
        for cid, c in (data.get("components") or {}).items():
            c = dict(c, id=cid)
            if c.get("source") and folder is not None and not Path(c["source"]).is_absolute():
                c["source"] = str((Path(folder) / c["source"]).resolve())
            net.components[cid] = c
        net.definitions = data.get("definitions") or {}
        net.blocks = data.get("blocks") or []
        net.wires = data.get("wires") or []
        net.sweep.update(data.get("sweep") or {})
        return net

    @classmethod
    def load(cls, path):
        path = Path(path)
        net = cls.from_dict(json.loads(path.read_text(encoding="utf-8")), path.parent)
        net.path = str(path)
        return net

    # --- one block on its own (for the Analysis tab) ---

    def port_range(self, block, port, depth=0):
        """A sensible range for an input port: the characterised range of the neuron or valve variable it drives
        (followed into subsystems), a function block's typical range, else None."""
        kind = block["type"]
        if kind in (NEURON, VALVE):
            c = self.component_of(block)
            a = next((a for a in (c["axes"] if c else []) if a["name"] == port), None)
            return (float(min(a["values"])), float(max(a["values"]))) if a else None
        if kind == SUBSYSTEM and depth < 32:
            d = self.definition_of(block)
            for w in (d["wires"] if d else []):
                if w["from"][0] == port:
                    inner = self.block(w["to"][0], d)
                    r = self.port_range(inner, w["to"][1], depth + 1) if inner else None
                    if r:
                        return r
            return None
        t = TYPES.get(kind)
        if t is not None and t.curve and len(t.inputs(block)) == 1:
            try:
                lo, hi = t.curve(block)
                return (float(lo), float(hi)) if math.isfinite(lo) and math.isfinite(hi) and hi > lo else None
            except Exception:
                return None
        return None

    def block_parameters(self, block, container=None):
        """The trainable parameters a block holds: its own, or for a subsystem every one inside it (nested too)."""
        if block["type"] != SUBSYSTEM:
            return [p for p in self.parameters() if p["block"] is block]
        names, todo = set(), [block.get("definition")]
        while todo:
            did = todo.pop()
            if did in names or did not in self.definitions:
                continue
            names.add(did)
            todo += [b.get("definition") for b in self.definitions[did]["blocks"] if b["type"] == SUBSYSTEM]
        wanted = {self.definitions[did]["name"] for did in names}
        return [p for p in self.parameters() if p["where"] in wanted]

    def analyse_block(self, block, sweeps, held=None, progress=None) -> dict:
        """A block's response on its own, as a dataset in the stored-characterisation format (see
        full_neuron.Characterisation) so the Analysis tab can show it: every combination of `sweeps` ([{"kind":
        "input" | "parameter", "port" (an input) or "parameter" (an entry of block_parameters), "label", "unit",
        "values"}]) is evaluated; inputs not swept are held at `held` {port: value} (default 0). Keys = the block's
        output ports; "extrapolated" = some block outside its characterised range. The network is not changed."""
        if block["type"] in (INPUT, CONSTANT, OUTPUT, INPORT, OUTPORT):
            raise ValueError("Sources, probes and ports have nothing to analyse.")
        tmp = Network()
        tmp.components, tmp._tables = self.components, self._tables
        tmp.definitions = copy.deepcopy(self.definitions)
        b = copy.deepcopy(block)
        tmp.root["blocks"].append(b)
        ins, outs = tmp.ports(b)
        if not outs:
            raise ValueError("The block has no output ports.")
        feeds = {}
        for k, port in enumerate(ins):
            i = tmp.add_block(INPUT, name=f"in{k}")
            i["value"] = float((held or {}).get(port, 0.0))
            tmp.connect((i["id"], "out"), (b["id"], port))
            feeds[port] = i["name"]
        probes = []
        for k, port in enumerate(outs):
            o = tmp.add_block(OUTPUT, name=f"out{k}")
            tmp.connect((b["id"], port), (o["id"], "in"))
            probes.append((port, o["name"]))
        # the swept parameters in the copy
        copies = tmp.parameters()

        def twin(prm):
            if prm["block"] is block:
                return next(q for q in copies if q["block"] is b and q["key"] == prm["key"])
            return next(q for q in copies if q["where"] == prm["where"] and q["block"]["id"] == prm["block"]["id"]
                        and q["key"] == prm["key"])
        targets = [twin(s_["parameter"]) if s_["kind"] == "parameter" else feeds[s_["port"]] for s_ in sweeps]
        grids = [np.asarray(s_["values"], float) for s_ in sweeps]
        shape = tuple(len(g) for g in grids)
        n = int(np.prod(shape)) if shape else 1
        values = {port: [None] * n for port, _ in probes}
        converged, flagged = [None] * n, [False] * n
        guess = None
        for flat, idx in enumerate(np.ndindex(*shape) if shape else [()]):
            overrides = {}
            for target, g, i in zip(targets, grids, idx):
                if isinstance(target, str):
                    overrides[target] = float(g[i])
                else:
                    tmp.set_parameter(target, float(g[i]))
            r = tmp.solve(overrides, guess)
            guess = r["values"]
            for port, name in probes:
                v = r["probes"][name]
                values[port][flat] = float(v) if math.isfinite(v) else None
            converged[flat], flagged[flat] = bool(r["converged"]), bool(r["flags"])
            if progress is not None and progress(flat + 1, n) is False:
                break
        units = {port: self.unit(block, port) for port, _ in probes}
        name = block["name"]
        return {"version": 1, "fingerprint": None, "complete": all(c is not None for c in converged),
                "axes": [{"part": s_["label"], "field": "input" if s_["kind"] == "input" else "parameter",
                          "label": s_["label"] + ("" if s_["kind"] == "input" else " (parameter)"),
                          "unit": s_.get("unit", ""), "values": [float(x) for x in g]} for s_, g in zip(sweeps, grids)],
                "keys": [{"key": port, "label": f"{name} {self.port_label(block, port) or port}".strip(),
                          "unit": units[port]} for port, _ in probes],
                "values": values, "converged": converged, "extrapolated": flagged, "block": name}

    # --- library blocks (*.mnb): a subsystem with what it needs, to use in other networks ---

    def _needs(self, did, defs=None, comps=None):
        defs, comps = defs if defs is not None else set(), comps if comps is not None else set()
        defs.add(did)
        for b in self.definitions[did]["blocks"]:
            if b.get("component"):
                comps.add(b["component"])
            if b["type"] == SUBSYSTEM and b.get("definition") and b["definition"] not in defs:
                self._needs(b["definition"], defs, comps)
        return defs, comps

    def export_definition(self, did, path):
        defs, comps = self._needs(did)
        data = {"type": BLOCK_FILE_TYPE, "version": 1, "definition": did,
                "definitions": {d: self.definitions[d] for d in defs},
                "components": {c: self.components[c] for c in comps}}
        Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")

    def import_definition(self, path) -> str:
        """Add a library block (*.mnb) and what it needs; returns the id of its definition here."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != BLOCK_FILE_TYPE:
            raise ValueError(f"{Path(path).name} is not a network library block (*{BLOCK_SUFFIX}).")
        cmap = {cid: self.add_component(dict(c)) for cid, c in data["components"].items()}
        dmap = {}
        for old in data["definitions"]:
            dmap[old] = self._new_id("d", set(self.definitions) | set(dmap.values()))
        names = {d["name"] for d in self.definitions.values()}
        for old, d in data["definitions"].items():
            d = copy.deepcopy(d)
            for b in d["blocks"]:
                if b.get("component"):
                    b["component"] = cmap.get(b["component"], b["component"])
                if b["type"] == SUBSYSTEM:
                    b["definition"] = dmap.get(b.get("definition"), b.get("definition"))
            base, n = d["name"], 2
            while d["name"] in names:
                d["name"] = f"{base} ({n})"
                n += 1
            names.add(d["name"])
            self.definitions[dmap[old]] = d
        return dmap[data["definition"]]


BLOCK_SUFFIX = ".mnb"
BLOCK_FILE_TYPE = "neuron_network_block"
BLOCK_FILTER = f"Network library block (*{BLOCK_SUFFIX})"


def sweep_csv(sweep) -> str:
    """The sweep's probes as CSV (one row per point)."""
    names = list(sweep["probes"])
    head = [sweep["x"]] + ([sweep["y"]] if sweep["y"] else []) + names + ["converged", "outside range"]
    lines = [",".join(head)]
    ys = sweep["y_values"] if sweep["y"] else [None]
    for j, yv in enumerate(ys):
        for i, xv in enumerate(sweep["x_values"]):
            row = [f"{xv:.10g}"] + ([f"{yv:.10g}"] if sweep["y"] else [])
            row += [f"{sweep['probes'][n][j, i]:.10g}" for n in names]
            row += [str(bool(sweep["converged"][j, i])), str(bool(sweep["flagged"][j, i]))]
            lines.append(",".join(row))
    return "\n".join(lines) + "\n"
