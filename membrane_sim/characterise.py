"""
Mechanical weights of a solved state, for back-inferring the neuron equation up to the
activation pressure p_a:

    p_a = (sum_j W_j p_j + W_0 p_0) / (sum_j W_j + W_0).

A weight belongs to an *input path*, not to a membrane: everything between input chamber j and
the activation chamber (one membrane, or e.g. membrane - weight chamber - membrane for bulk
modulus tuning) is lumped into

    W_j = dV_j / (p_j - p_a),   dV_j = volume the path's membranes push into the activation chamber.

Paths are found from the geometry: every shell bounding the activation chamber is followed back
through closed intermediate chambers to the constant-pressure chamber(s) driving it. A shell side
with no chamber is the ambient (0 gauge). Several shells reaching the same input form one path.

The activation chamber's own fluid adds W_0 = -dV_a / (p_a - p_0), with p_0 the pressure of its law
at the rest volume (a compliant chamber; W_0 = 0 for a rigid, incompressible one).

Only W is identified: pressure/volume data can not separate an effective area from a stiffness.
The tangent W_tan = d(dV_j)/d(p_j) is the local sensitivity with p_a and every other input fixed
(intermediate closed chambers follow their own law).
"""
import math

import numpy as np
import torch

from .fluid import cone_volume_grad, shell_cone_volume
from .solver import _Tangent

AMBIENT = "ambient"


def chamber_compliance(volume) -> float:
    """Tangent compliance -d(dV)/dP of a chamber's pressure law (inf for a constant-pressure chamber)."""
    slope = volume.pressure_slope(volume.delta_volume)
    return math.inf if slope == 0 else -1.0 / slope


def _sides(shell, volumes):
    """{+1: [volumes behind the shell normal], -1: [volumes in front]}."""
    out = {1: [], -1: []}
    for v in volumes:
        for s, side, _ in v.boundaries:
            if s is shell:
                out[side].append(v)
    return out


def input_paths(activation, volumes):
    """
    [(input label, input volume or None for the ambient, [(shell, side of the activation chamber)])],
    one entry per input driving the activation chamber. A shell whose far side reaches several
    inputs through closed chambers gets the label of all of them joined with ' + ' (and volume None).
    """
    paths = {}
    for shell, act_side, _ in activation.boundaries:
        reached, seen, stack = set(), {id(activation)}, [(shell, -act_side)]
        while stack:
            s, far = stack.pop()
            chambers = _sides(s, volumes)[far]
            if not chambers:
                reached.add((AMBIENT, None))
            for v in chambers:
                if id(v) in seen:
                    continue
                seen.add(id(v))
                if not v.is_closed:
                    reached.add((v.name, v))
                    continue
                for s2, side2, _ in v.boundaries:  # through an intermediate closed chamber
                    if s2 is not s:
                        stack.append((s2, -side2))
        label = " + ".join(sorted(name for name, _ in reached)) or AMBIENT
        volume = next(iter(reached))[1] if len(reached) == 1 else None
        paths.setdefault(label, (volume, []))[1].append((shell, act_side))
    return [(label, volume, shells) for label, (volume, shells) in paths.items()]


def _side_gradient(solver, shell, side):
    """Free-dof gradient of the volume on `side` of one shell (the load of a unit pressure there)."""
    g = torch.zeros(solver.n_dof, dtype=torch.float64, device=solver.device)
    dofs = shell.face_dofs + solver.dof_offset[id(shell)]
    g.index_add_(0, dofs.reshape(-1), side * cone_volume_grad(shell.x[shell.faces] - shell.volume_origin).reshape(-1))
    return g[solver._free_t].cpu().numpy()


def _volume_gradient(solver, volume):
    return sum(_side_gradient(solver, shell, side) for shell, side, _ in volume.boundaries)


def input_weights(env, activation, load_factor: float = 1.0) -> dict:
    """
    Weights of every input path into `activation` (a FluidVolume of env) at the current solved
    state, in model units. Needs env.solve() to have run (its solver gives the tangent).
    Returns {"p_a", "inputs": [{"input", "p", "dp", "dV", "W", "W_tan", "shells"}], "chamber": {...}}.
    """
    solver = env.solver
    volumes = solver.fluid_volumes
    p_a = activation.P

    # tangent with the activation pressure held fixed; other closed chambers keep their law
    state = solver.evaluate(load_factor, tangent=True)
    keep = [k for k, (v, _) in enumerate(state["closed"]) if v is not activation]
    try:
        tangent = _Tangent(state["K"], state["U"][:, keep], state["c"][keep])
    except (RuntimeError, np.linalg.LinAlgError):  # singular (a limit point)
        tangent = None
    g_act = _volume_gradient(solver, activation)

    inputs = []
    for label, volume, shells in input_paths(activation, volumes):
        # volume pushed into the activation chamber = minus its contribution to the chamber's dV
        dV = -sum(side * (shell_cone_volume(s, s.x).item() - shell_cone_volume(s, s.X).item())
                  for s, side in shells)
        p = 0.0 if label == AMBIENT else (volume.P if volume is not None else math.nan)
        dp = p - p_a
        w_tan = math.nan
        direct_ambient = label == AMBIENT and all(not _sides(s, volumes)[-side] for s, side in shells)
        if tangent is not None and (volume is not None or direct_ambient):
            if volume is not None:
                load = _volume_gradient(solver, volume)
            else:  # ambient: a unit pressure on the far side of the path's shells
                load = sum(_side_gradient(solver, s, -side) for s, side in shells)
            try:
                w_tan = float(-g_act @ tangent.solve(load))
            except np.linalg.LinAlgError:
                pass
        inputs.append({"input": label, "p": p, "dp": dp, "dV": dV,
                       "W": dV / dp if dp != 0 and math.isfinite(dp) else math.nan,
                       "W_tan": w_tan, "shells": [s.name for s, _ in shells]})

    p0 = activation.pressure(0.0)
    dV_a = activation.delta_volume
    chamber = {"input": activation.name + " (fluid)", "p": p0, "dp": p_a - p0, "dV": -dV_a,
               "W": -dV_a / (p_a - p0) if p_a != p0 else math.nan,
               "W_tan": chamber_compliance(activation)}
    return {"p_a": p_a, "inputs": inputs, "chamber": chamber}


def rebuild_activation_pressure(weights) -> float:
    """p_a = (sum W_j p_j + W_0 p_0) / (sum W_j + W_0) from input_weights() output."""
    terms = [(w["W"], w["p"]) for w in weights["inputs"]]
    ch = weights["chamber"]
    if math.isfinite(ch["W"]):
        terms.append((ch["W"], ch["p"]))
    terms = [(W, p) for W, p in terms if math.isfinite(W)]
    if not terms or sum(W for W, _ in terms) == 0:
        return math.nan  # every dp is 0 (e.g. all inputs at the rest pressure): no weight is defined
    return sum(W * p for W, p in terms) / sum(W for W, _ in terms)


def polyfit_weight(dp, W, degree: int) -> dict:
    """Least-squares polynomial W(dp) = c0 + c1 dp + ... of the given degree (capped at the number of
    distinct points - 1). Points with (almost) no pressure difference are left out (W = 0/0 there).

    The residuals are weighted by |dp|, i.e. the fit minimises the error in the displaced volume
    dV = W dp, which is what moves p_a. Unweighted, the few points just off dp = 0, where W = dV/dp
    is large (a slack membrane: W ~ dp^(-2/3)), dominate the fit although they barely matter."""
    dp, W = np.asarray(dp, float), np.asarray(W, float)
    ok = np.isfinite(dp) & np.isfinite(W)
    if ok.any():
        ok &= np.abs(dp) > 1e-6 * np.abs(dp[ok]).max()
    left_out = [float(x) for x in dp[~ok & np.isfinite(dp)]]
    dp, W = dp[ok], W[ok]
    if not len(W):
        return {"kind": "none", "points": 0, "degree": 0, "max_degree": -1, "coefficients": [0.0],
                "dp_range": (math.nan, math.nan), "exact": True, "left_out": left_out}
    top = min(len(np.unique(dp)) - 1, MAX_DEGREE)
    degree = min(degree, top)
    scale = np.abs(dp).max()  # fit in dp / scale for conditioning
    c = np.polynomial.polynomial.polyfit(dp / scale, W, degree, w=np.abs(dp) / scale)
    c[np.abs(c) < 1e-12 * np.abs(c).max()] = 0.0  # round-off terms of a symmetric fit
    fitted = np.polynomial.polynomial.polyval(dp / scale, c)
    residual = np.abs((fitted - W) * dp).max() / np.abs(W * dp).max()  # relative error in dV
    return {"kind": "polynomial", "points": len(W), "degree": degree, "max_degree": top,
            "coefficients": (c / scale ** np.arange(degree + 1)).tolist(),
            "dp_range": (float(dp.min()), float(dp.max())), "exact": bool(residual < 1e-9),
            "left_out": left_out}


MAX_DEGREE = 10


def solve_activation(terms, p_hint=None):
    """
    The activation pressure the equation gives for one input set: the root of
        sum_k W_k(x_k) (p_k - p_a) = 0,   x_k = p_k - p_a (inputs) or p_a - p_k (the chamber term),
    i.e. p_a = sum W_k p_k / sum W_k with the weights evaluated at p_a itself.
    terms: [(p_k, coefficients, is_chamber)], coefficients either one polynomial (a list) or a pair
    (for x >= 0, for x < 0) of them. The root is searched between the lowest and highest p_k;
    without a sign change the p_a with the smallest residual (nearest p_hint on ties) is used.
    """
    from scipy.optimize import brentq

    def residual(pa):
        total = 0.0
        for p, c, chamber in terms:
            x = pa - p if chamber else p - pa
            total += evaluate_weight(c, x) * (p - pa)
        return total

    ps = [p for p, _, _ in terms]
    lo, hi = min(ps), max(ps)
    if hi - lo < 1e-300:
        return lo
    grid = np.linspace(lo, hi, 121)
    r = np.asarray(residual(grid), float) * np.ones_like(grid)
    roots = [brentq(residual, grid[k], grid[k + 1]) for k in range(len(grid) - 1)
             if np.sign(r[k]) != np.sign(r[k + 1]) and np.isfinite(r[k]) and np.isfinite(r[k + 1])]
    roots += [grid[k] for k in range(len(grid)) if r[k] == 0]
    if not roots:
        roots = [grid[int(np.nanargmin(np.abs(r)))]]
    hint = 0.5 * (lo + hi) if p_hint is None else p_hint
    return float(min(roots, key=lambda x: abs(x - hint)))


def evaluate_weight(coefficients, x):
    """W(x) for one polynomial (a list of coefficients) or a (x >= 0, x < 0) pair of them."""
    if isinstance(coefficients, tuple):
        positive, negative = coefficients
        return np.where(np.asarray(x) >= 0, np.polynomial.polynomial.polyval(x, positive),
                        np.polynomial.polynomial.polyval(x, negative))
    return np.polynomial.polynomial.polyval(x, coefficients)


SIDES = ("+", "-")  # dp > 0, dp < 0


def weight_coefficients(fit):
    """(positive side, negative side) coefficients of a piecewise weight; a side without samples
    uses the other side's polynomial."""
    positive, negative = fit["sides"]["+"], fit["sides"]["-"]
    positive, negative = positive or negative, negative or positive
    return (positive["coefficients"], negative["coefficients"])


def weight_degree(fit):
    """Total order of a piecewise weight (the sum over its sampled sides)."""
    return sum(f["degree"] for f in fit["sides"].values() if f is not None)


LOWEST_TOTAL = "lowest total order"
BIGGEST_ERROR = "biggest own error first"


def fit_neuron_equation(samples, tolerance: float, method: str = LOWEST_TOTAL) -> dict:
    """
    Fit every weight W_k(dp_k) with least-squares polynomials, one for dp_k > 0 and one for dp_k < 0,
    so that the neuron equation, solved for p_a, reproduces the simulated activation pressure of
    every sample within `tolerance` (absolute, in the samples' pressure unit), with degrees as low as
    possible. Each side of each weight is a piece with its own degree; all start constant (degree 0).
    A weight sampled on one side only uses that side's polynomial for both signs. Two ways to raise
    the degrees:

    LOWEST_TOTAL   exhaustive: try every combination of piece degrees with total order 0, 1, 2, ...
                   and stop at the first total order where one meets the tolerance (the one with the
                   smallest error). Guaranteed to give the lowest possible total order.
    BIGGEST_ERROR  greedy: while the tolerance is not met, the piece with the biggest error of its
                   own gets one degree more. A piece's own error is the worst |p_a error| when only
                   that piece uses its polynomial and everything else takes its sampled values. Fast,
                   but can end above the lowest total order (errors of different pieces can cancel).

    A piece is never raised past the degree that already fits its points exactly, or past the
    number of distinct points - 1. If the tolerance can not be met, the best combination found is
    returned with "met" False.

    samples: [{"p_a": simulated activation pressure,
               "terms": {key: (p_k, dp_k, W_k)}}]  with dp_k = p_k - p_a for an input and
             p_a - p_k for the key "W0" (the activation chamber's own compliance about p_0 = p_k).
    Returns {"fits": {key: {"kind": "piecewise", "sides": {"+": polyfit_weight() result or None,
                                                           "-": ...}}},
             "error": worst |p_a error|, "errors": per sample, "predicted": per sample,
             "weight_errors": {key: {side: own error}}, "tolerance", "met", "method",
             "evaluated": number of degree combinations tried}.
    """
    keys = list(dict.fromkeys(k for s in samples for k in s["terms"]))
    data = {}
    for k in keys:
        dp = np.array([s["terms"][k][1] for s in samples if k in s["terms"]], float)
        W = np.array([s["terms"][k][2] for s in samples if k in s["terms"]], float)
        for side, mask in (("+", dp > 0), ("-", dp < 0)):
            data[k, side] = (dp[mask], W[mask])
    # pieces with usable points; a weight with none at all (e.g. a rigid, incompressible activation
    # fluid) has W = 0 and is left out of the equation
    pieces = [(k, side) for k in keys for side in SIDES if polyfit_weight(*data[k, side], 0)["kind"] != "none"]
    keys = [k for k in keys if any(piece[0] == k for piece in pieces)]

    cache = {}

    def fit(piece, degree):
        if (piece, degree) not in cache:
            cache[piece, degree] = polyfit_weight(*data[piece], degree)
        return cache[piece, degree]

    # highest useful degree per piece: the first that is exact, else what the number of points allows
    caps = {}
    for piece in pieces:
        d = 0
        while d < fit(piece, 0)["max_degree"] and not fit(piece, d)["exact"]:
            d += 1
        caps[piece] = d

    def weights_for(degrees):
        return {k: {"kind": "piecewise",
                    "sides": {side: fit((k, side), degrees[k, side]) if (k, side) in degrees else None
                              for side in SIDES}} for k in keys}

    def predict(fits, only=None):
        """p_a per sample from the equation. With `only` (a piece), only that piece uses its polynomial:
        the other side of its weight and every other weight take the sample's value (where a sample
        value is missing, dp ~ 0 and the fitted value is used: it barely matters)."""
        out = []
        for s in samples:
            terms = []
            for k in keys:
                if k not in s["terms"] or not math.isfinite(s["terms"][k][0]):
                    continue
                p, _, W = s["terms"][k]
                c = weight_coefficients(fits[k])
                if only is not None and math.isfinite(W):
                    if k != only[0]:
                        c = [W]
                    else:
                        c = (c[0], [W]) if only[1] == "+" else ([W], c[1])
                terms.append((p, c, k == "W0"))
            out.append(solve_activation(terms, s["p_a"]) if terms else math.nan)
        return out

    def worst(predicted):
        errors = [abs(p - s["p_a"]) for p, s in zip(predicted, samples)]
        return errors, max(errors) if errors else 0.0

    evaluated = 0

    def evaluate(degrees):
        nonlocal evaluated
        evaluated += 1
        fits = weights_for(degrees)
        predicted = predict(fits)
        errors, error = worst(predicted)
        return {"fits": fits, "predicted": predicted, "errors": errors, "error": error}

    if method == LOWEST_TOTAL:
        best = None
        for total in range(sum(caps.values()) + 1):
            meeting = None
            for degrees in _compositions(total, [caps[piece] for piece in pieces]):
                result = evaluate(dict(zip(pieces, degrees)))
                if best is None or result["error"] < best["error"]:
                    best = result
                if result["error"] <= tolerance and (meeting is None or result["error"] < meeting["error"]):
                    meeting = result
            if meeting is not None:
                best = meeting
                break
    elif method == BIGGEST_ERROR:
        degrees = {piece: 0 for piece in pieces}
        while True:
            best = evaluate(degrees)
            if best["error"] <= tolerance:
                break
            open_pieces = [piece for piece in pieces if degrees[piece] < caps[piece]]
            if not open_pieces:
                break
            own = {piece: worst(predict(best["fits"], only=piece))[1] for piece in open_pieces}
            degrees[max(open_pieces, key=lambda piece: own[piece])] += 1
    else:
        raise ValueError(f"unknown method {method!r}")

    own = {k: {} for k in keys}
    for piece in pieces:
        own[piece[0]][piece[1]] = worst(predict(best["fits"], only=piece))[1]
    return {"fits": best["fits"], "error": best["error"], "errors": best["errors"],
            "predicted": best["predicted"], "weight_errors": own, "tolerance": tolerance,
            "met": best["error"] <= tolerance, "method": method, "evaluated": evaluated}


def _compositions(total, caps):
    """Every tuple of non-negative degrees d with sum(d) == total and d[i] <= caps[i]."""
    if not caps:
        if total == 0:
            yield ()
        return
    for d in range(min(total, caps[0]) + 1):
        for rest in _compositions(total - d, caps[1:]):
            yield (d,) + rest


def polynomial_text(coefficients, variable="dp", precision=5):
    terms = [f"{c:+.{precision}g}" + ("" if n == 0 else f" {variable}" if n == 1 else f" {variable}^{n}")
             for n, c in enumerate(coefficients)]
    return " ".join(terms).lstrip("+")


# -----------------------------
# The neuron equation as LaTeX
# -----------------------------

def latex_number(x, precision=4):
    """1.234e-05 -> 1.234 \times 10^{-5}."""
    text = f"{x:.{precision}g}"
    if "e" not in text:
        return text
    mantissa, exponent = text.split("e")
    return rf"{mantissa} \times 10^{{{int(exponent)}}}"


def latex_name(label):
    return r"\mathrm{" + label.replace("_", r"\_").replace(" ", r"\ ") + "}"


def latex_polynomial(coefficients, variable, precision=4):
    out = ""
    for n, c in enumerate(coefficients):
        if c == 0 and len(coefficients) > 1:
            continue
        term = latex_number(abs(c), precision) + ("" if n == 0 else rf"\,{variable}" if n == 1
                                                   else rf"\,{variable}^{{{n}}}")
        out += ("-" if c < 0 else "+" if out else "") + term
    return out or "0"


def neuron_equation_latex(inputs, chamber=None, activation="a", precision=6, p_unit="kPa", w_unit="mm^3/kPa"):
    """
    The neuron equation up to the activation pressure, as LaTeX pieces.
      inputs  : [(label, fit)] with fit a piecewise weight from fit_neuron_equation() for W_j against
                dp_j = p_j - p_a
      chamber : optional (fit, p0) for the activation chamber's own compliance W_0(p_a - p_0)
    Returns {"main", "weights": [{"symbol", "variable", "pieces": [(polynomial, condition)],
    "definition"}], "note"}; see equation_align() and equation_lines() to put it together. The
    weights depend on p_a through dp_j, so the main equation is implicit in p_a.
    """
    pa = rf"p_{{{latex_name(activation)}}}"
    terms, weights = [], []
    for j, (label, fit) in enumerate(inputs, start=1):
        W, p, dp = rf"W_{{{j}}}", rf"p_{{{latex_name(label)}}}", rf"\Delta p_{{{j}}}"
        terms.append((W, p))
        weights.append({"symbol": W, "variable": dp, "pieces": _latex_pieces(fit, dp, precision),
                        "definition": rf"{dp} = {p} - {pa}"})
    if chamber is not None:
        fit, p0 = chamber
        W, dp = r"W_{0}", r"\Delta p_{0}"
        terms.append((W, "p_{0}"))
        weights.append({"symbol": W, "variable": dp, "pieces": _latex_pieces(fit, dp, precision),
                        "definition": rf"{dp} = {pa} - p_{{0}}, \quad p_{{0}} = {latex_number(p0, precision)}"
                                      rf"\ \mathrm{{{p_unit}}}"})
    numerator = " + ".join(rf"{W}\,{p}" for W, p in terms)
    denominator = " + ".join(W for W, _ in terms)
    main = rf"{pa} = \frac{{{numerator}}}{{{denominator}}}"
    note = rf"\mathrm{{W\ in\ {w_unit.replace('^3', '^{3}')},\ pressures\ in\ {p_unit}}}"
    return {"main": main, "weights": weights, "note": note}


def _latex_pieces(fit, variable, precision):
    """[(polynomial, condition)] of a piecewise weight: one per sampled side."""
    sides = fit.get("sides", {}) if fit.get("kind") == "piecewise" else {}
    sampled = [side for side in SIDES if sides.get(side) is not None]
    if not sampled:
        return [(r"\mathrm{undetermined}", "")]
    if len(sampled) == 1:
        sign = ">" if sampled[0] == "+" else "<"
        only = rf"\mathrm{{all}}\ {variable}\ (\mathrm{{sampled\ only}}\ {variable} {sign} 0)"
        return [(latex_polynomial(sides[sampled[0]]["coefficients"], variable, precision), only)]
    return [(latex_polynomial(sides[side]["coefficients"], variable, precision),
             rf"{variable} {'>' if side == '+' else '<'} 0") for side in SIDES]


def equation_align(equation):
    """A LaTeX align block of the equation (piecewise weights as cases)."""
    rows = [equation["main"].replace(" = ", " &= ", 1)]
    for w in equation["weights"]:
        if len(w["pieces"]) == 1:
            poly, condition = w["pieces"][0]
            rhs = poly + (rf", \quad {condition}" if condition else "")
        else:
            rhs = (r"\begin{cases} " + r" \\ ".join(rf"{poly}, & {condition}" for poly, condition in w["pieces"])
                   + r" \end{cases}")
        rows.append(rf"{w['symbol']}({w['variable']}) &= {rhs}, \quad {w['definition']}")
    return "\\begin{align}\n" + " \\\\\n".join("  " + row for row in rows) + "\n\\end{align}\n"


def equation_lines(equation):
    """Lines for matplotlib mathtext (which has no cases environment): [(latex, weight index or None,
    level)] with level 0 for the main equation, 1 for a weight piece, 2 for a definition."""
    lines = [(equation["main"], None, 0)]
    for n, w in enumerate(equation["weights"]):
        for poly, condition in w["pieces"]:
            lines.append((rf"{w['symbol']}({w['variable']}) = {poly}" + (rf", \quad {condition}" if condition else ""),
                          n, 1))
        lines.append((w["definition"], n, 2))
    lines.append((equation["note"], None, 2))
    return lines
