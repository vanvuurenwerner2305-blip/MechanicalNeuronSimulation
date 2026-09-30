"""
Mechanical weights of a solved state, for back-inferring the neuron equation up to the
pre-activation pressure p_a:

    p_a = (sum_j W_j p_j + W_0 p_0) / (sum_j W_j + W_0).

A weight belongs to an *input path*, not to a membrane: everything between input chamber j and
the pre-activation chamber (one membrane, or e.g. membrane - weight chamber - membrane for bulk
modulus tuning) is lumped into

    W_j = dV_j / (p_j - p_a),   dV_j = volume the path's membranes push into the pre-activation chamber.

Paths are found from the geometry: every shell bounding the pre-activation chamber is followed back
through closed intermediate chambers to the constant-pressure chamber(s) driving it. A shell side
with no chamber is the ambient (0 gauge). Several shells reaching the same input form one path.

The pre-activation chamber's own fluid adds W_0 = -dV_a / (p_a - p_0), with p_0 the pressure of its law
at the rest volume (a compliant chamber; W_0 = 0 for a rigid, incompressible one).

A path through a closed chamber that is not neutral at rest (a gas weight chamber filled above ambient,
a liquid one filled with more or less liquid than its volume) pushes volume into the pre-activation
chamber even when all pressures are equal: a bias. Such a path is *biased*; its volume is modelled as
dV_j = b_j + W_j(dp_j) dp_j, and the equation gets the bias B = sum b_j:

    p_a = (sum_j W_j p_j + W_0 p_0 + B) / (sum_j W_j + W_0).

b_j is measured, not fitted (fitted, it trades off against the W polynomials and is not identifiable
from a sweep): at the neutral state, every input at the pre-activation chamber's rest pressure p_0, an
unbiased path pushes nothing, so what a biased path pushes there is its bias (bias_volumes()).

Only W is identified: pressure/volume data can not separate an effective area from a stiffness.
The tangent W_tan = d(dV_j)/d(p_j) is the local sensitivity with p_a and every other input fixed
(intermediate closed chambers follow their own law).
"""
import math

import numpy as np
import torch

from .fluid import wall_volume, wall_volume_terms
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


def is_neutral(volume) -> bool:
    """A closed chamber whose pressure law gives 0 (gauge) at its rest volume: it pushes nothing at rest."""
    return abs(volume.pressure(0.0)) <= 1e-12


def input_paths(activation, volumes):
    """
    [(input label, input volume or None for the ambient, [(shell, side of the pre-activation chamber)],
      [closed intermediate chambers on the path])], one entry per input driving the pre-activation chamber.
    A shell whose far side reaches several inputs through closed chambers gets the label of all of them
    joined with ' + ' (and volume None).
    """
    paths = {}
    for shell, act_side, _ in activation.boundaries:
        reached, seen, stack, between = set(), {id(activation)}, [(shell, -act_side)], []
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
                between.append(v)
                for s2, side2, _ in v.boundaries:  # through an intermediate closed chamber
                    if s2 is not s:
                        stack.append((s2, -side2))
        label = " + ".join(sorted(name for name, _ in reached)) or AMBIENT
        volume = next(iter(reached))[1] if len(reached) == 1 else None
        entry = paths.setdefault(label, (volume, [], []))
        entry[1].append((shell, act_side))
        entry[2].extend(v for v in between if all(v is not u for u in entry[2]))
    return [(label, volume, shells, between) for label, (volume, shells, between) in paths.items()]


def _side_gradient(solver, shell, side):
    """Free-dof gradient of the volume on `side` of one shell (the load of a unit pressure there)."""
    g = torch.zeros(solver.n_dof, dtype=torch.float64, device=solver.device)
    dofs, grad, _ = wall_volume_terms(shell, side, tangent=False)
    g.index_add_(0, (dofs + solver.dof_offset[id(shell)]).reshape(-1), grad.reshape(-1))
    return g[solver._free_t].cpu().numpy()


def _volume_gradient(solver, volume):
    return sum(_side_gradient(solver, shell, side) for shell, side, _ in volume.boundaries)


def input_weights(env, activation, load_factor: float = 1.0) -> dict:
    """
    Weights of every input path into `activation` (a FluidVolume of env) at the current solved
    state, in model units. Needs env.solve() to have run (its solver gives the tangent).
    Returns {"p_a", "inputs": [{"input", "p", "dp", "dV", "W", "W_tan", "shells", "biased",
    "bias_chambers"}], "chamber": {...}}. W is the secant dV / dp; for a biased path it includes the bias.
    """
    solver = env.solver
    volumes = solver.fluid_volumes
    p_a = activation.P

    # tangent with the pre-activation pressure held fixed; other closed chambers keep their law
    state = solver.evaluate(load_factor, tangent=True)
    keep = [k for k, (v, _) in enumerate(state["closed"]) if v is not activation]
    try:
        tangent = _Tangent(state["K"], state["U"][:, keep], state["c"][keep])
    except (RuntimeError, np.linalg.LinAlgError):  # singular (a limit point)
        tangent = None
    g_act = _volume_gradient(solver, activation)

    inputs = []
    for label, volume, shells, between in input_paths(activation, volumes):
        # volume pushed into the pre-activation chamber = minus its contribution to the chamber's dV
        dV = -sum(side * (wall_volume(s) - wall_volume(s, rest=True)) for s, side in shells)
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
        biasing = [v.name for v in between if not is_neutral(v)]
        inputs.append({"input": label, "p": p, "dp": dp, "dV": dV,
                       "W": dV / dp if dp != 0 and math.isfinite(dp) else math.nan,
                       "W_tan": w_tan, "shells": [s.name for s, _ in shells],
                       "biased": bool(biasing), "bias_chambers": biasing})

    p0 = activation.pressure(0.0)
    dV_a = activation.delta_volume
    chamber = {"input": activation.name + " (fluid)", "p": p0, "dp": p_a - p0, "dV": -dV_a,
               "W": -dV_a / (p_a - p0) if p_a != p0 else math.nan,
               "W_tan": chamber_compliance(activation)}
    return {"p_a": p_a, "inputs": inputs, "chamber": chamber}


def bias_volumes(weights) -> dict:
    """{path label: b_j} from input_weights() at the neutral state (every input at the pre-activation
    chamber's rest pressure): the volume a biased path pushes in at zero pressure difference. The small
    pressure difference left there is corrected to first order with the tangent weight."""
    out = {}
    for w in weights["inputs"]:
        if w.get("biased"):
            slope = w["W_tan"] if math.isfinite(w["W_tan"]) else 0.0
            dp = w["dp"] if math.isfinite(w["dp"]) else 0.0
            out[w["input"]] = w["dV"] - slope * dp
    return out


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


SENSITIVITY_CANCELLATION = 1e-3  # sum W below this share of sum |W|: the weights cancel, dp_a/dW is meaningless


def activation_sensitivities(terms) -> dict:
    """dp_a/dW_k of every weight at one solved state (Article 2, Eq. 4.10):
        dp_a/dW_k = (p_k - p_a) / sum_j W_j,
    terms: {key: (dp_k, W_k)} with dp_k = p_k - p_a for an input and p_a - p_k for the key "W0" (the
    pre-activation chamber's own compliance), whose sign is therefore flipped. Non-finite W are left out
    of the sum.

    The sum must be clearly positive. A chamber pre-pressurised inside a path (e.g. a gas weight
    chamber) acts as a hidden bias: where all pressures around the pre-activation chamber are (nearly)
    equal, p_a still differs from them, so the secant weights must cancel (NeuronTest2 at Input1 = 0:
    -152.6 + 109.7 + 42.6 + 0.3 = 0) and dp_a/dW -> infinity. A state whose sum is not positive, or below
    SENSITIVITY_CANCELLATION of the weights' total size, gets NaN for every weight (the fit leaves it out)."""
    total = weight_total(W for _, W in terms.values())
    return {k: (-dp if k == "W0" else dp) / total for k, (dp, _) in terms.items()}


def weight_total(weights) -> float:
    """sum W over the finite weights, or NaN where they (nearly) cancel (see activation_sensitivities)."""
    finite = [W for W in weights if math.isfinite(W)]
    total = sum(finite)
    if not finite or total <= SENSITIVITY_CANCELLATION * sum(abs(W) for W in finite):
        return math.nan
    return total


def polyfit_weight(dp, W, degree: int, sensitivity=None) -> dict:
    """Least-squares polynomial W(dp) = c0 + c1 dp + ... of the given degree (capped at the number of
    distinct points - 1). Points with (almost) no pressure difference are left out (W = 0/0 there).

    The residuals are weighted by the sensitivity of p_a to W at each point, |dp_a/dW| = |dp| / sum W
    (Eq. 4.10, see activation_sensitivities), so an error in W costs what it moves p_a by. Without
    `sensitivity` the weight is |dp| (the error in the displaced volume dV = W dp). Unweighted, the few
    points just off dp = 0, where W = dV/dp is large (a slack membrane: W ~ dp^(-2/3)), dominate the
    fit although they barely move p_a."""
    dp, W = np.asarray(dp, float), np.asarray(W, float)
    sensitivity = np.abs(dp) if sensitivity is None else np.abs(np.asarray(sensitivity, float))
    ok = np.isfinite(dp) & np.isfinite(W) & np.isfinite(sensitivity)
    if ok.any():
        ok &= np.abs(dp) > 1e-6 * np.abs(dp[ok]).max()
    left_out = [float(x) for x in dp[~ok & np.isfinite(dp)]]
    dp, W, sensitivity = dp[ok], W[ok], sensitivity[ok]
    if not len(W):
        return {"kind": "none", "points": 0, "degree": 0, "max_degree": -1, "coefficients": [0.0],
                "dp_range": (math.nan, math.nan), "exact": True, "left_out": left_out}
    top = min(len(np.unique(dp)) - 1, MAX_DEGREE)
    degree = min(degree, top)
    scale = np.abs(dp).max()  # fit in dp / scale for conditioning
    weight = sensitivity / sensitivity.max()  # only relative weights matter
    c = np.polynomial.polynomial.polyfit(dp / scale, W, degree, w=weight)
    c[np.abs(c) < 1e-12 * np.abs(c).max()] = 0.0  # round-off terms of a symmetric fit
    fitted = np.polynomial.polynomial.polyval(dp / scale, c)
    residual = np.abs((fitted - W) * weight).max() / np.abs(W * weight).max()  # relative weighted error
    return {"kind": "polynomial", "points": len(W), "degree": degree, "max_degree": top,
            "coefficients": (c / scale ** np.arange(degree + 1)).tolist(),
            "dp_range": (float(dp.min()), float(dp.max())), "exact": bool(residual < 1e-9),
            "left_out": left_out}


MAX_DEGREE = 10


def solve_activation(terms, p_hint=None, bias=0.0, all_roots=False):
    """
    The pre-activation pressure the equation gives for one input set: the root of
        sum_k W_k(x_k) (p_k - p_a) + B = 0,   x_k = p_k - p_a (inputs) or p_a - p_k (the chamber term),
    i.e. p_a = (sum W_k p_k + B) / sum W_k with the weights evaluated at p_a itself.
    terms: [(p_k, coefficients, is_chamber)], coefficients either one polynomial (a list) or a pair
    (for x >= 0, for x < 0) of them; bias: B (a volume). Without a bias the root is searched between
    the lowest and highest p_k (a weighted average); a bias can move p_a outside them, so the range is
    widened by about B / sum W (and includes p_hint). Without a sign change the p_a with the smallest
    residual (nearest p_hint on ties) is used. Polynomial weights can make the equation multivalued;
    all_roots=True returns every root found (sorted), else the one nearest p_hint (or the middle).
    """
    from scipy.optimize import brentq

    def residual(pa):
        total = bias
        for p, c, chamber in terms:
            x = pa - p if chamber else p - pa
            total += evaluate_weight(c, x) * (p - pa)
        return total

    ps = [p for p, _, _ in terms]
    lo, hi = min(ps), max(ps)
    if bias:
        if p_hint is not None and math.isfinite(p_hint):
            lo, hi = min(lo, p_hint), max(hi, p_hint)
        slope = sum(float(evaluate_weight(c, 0.0)) for _, c, _ in terms)
        shift = abs(bias) / slope if slope > 0 else 0.0
        pad = max(0.5 * (hi - lo), 2.0 * shift) + 1e-9 * max(1.0, abs(lo), abs(hi))
        lo, hi = lo - pad, hi + pad
    if hi - lo < 1e-300:
        return [lo] if all_roots else lo
    grid = np.linspace(lo, hi, 121)
    r = np.asarray(residual(grid), float) * np.ones_like(grid)
    roots = [brentq(residual, grid[k], grid[k + 1]) for k in range(len(grid) - 1)
             if np.sign(r[k]) != np.sign(r[k + 1]) and np.isfinite(r[k]) and np.isfinite(r[k + 1])]
    roots += [grid[k] for k in range(len(grid)) if r[k] == 0]
    if not roots:
        roots = [grid[int(np.nanargmin(np.abs(r)))]]
    if all_roots:
        roots = sorted(float(x) for x in roots)
        return [x for n, x in enumerate(roots) if n == 0 or x - roots[n - 1] > 1e-9 * (hi - lo)]
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
# Budget of degree combinations for one fit. An unreachable tolerance would otherwise make the exhaustive
# search try every combination up to the degree caps: millions of combinations, hours of work.
MAX_EVALUATIONS = 2000
BIGGEST_ERROR = "biggest own error first"


def fit_neuron_equation(samples, tolerance: float, method: str = LOWEST_TOTAL, check=None,
                        max_evaluations: int = MAX_EVALUATIONS, bias=None) -> dict:
    """
    Fit every weight W_k(dp_k) with least-squares polynomials, one for dp_k > 0 and one for dp_k < 0,
    so that the neuron equation, solved for p_a, reproduces the simulated pre-activation pressure of
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
    returned with "met" False. The search also stops after `max_evaluations` degree combinations
    ("stopped" True), with the best combination so far. `check()`, if given, is called before every
    combination; it can raise to cancel the fit.

    samples: [{"p_a": simulated pre-activation pressure,
               "terms": {key: (p_k, dp_k, W_k)},     with dp_k = p_k - p_a for an input and
                                                     p_a - p_k for the key "W0" (the pre-activation
                                                     chamber's own compliance about p_0 = p_k),
               "volumes": {key: dV_k}}]   (needed for the keys in `bias`)
             with bias = {key: b_k}: measured bias volumes of the biased paths (bias_volumes()). Their
             weights are fitted on (dV_k - b_k) / dp_k instead of the secant, and the equation's bias
             B = sum b_k is returned as "bias".
    Returns {"fits": {key: {"kind": "piecewise", "sides": {"+": polyfit_weight() result or None,
                                                           "-": ...}}},
             "error": worst |p_a error|, "errors": per sample, "predicted": per sample,
             "weight_errors": {key: {side: own error}}, "tolerance", "met", "method",
             "evaluated": number of degree combinations tried, "stopped": the budget ran out first}.
    """
    keys = list(dict.fromkeys(k for s in samples for k in s["terms"]))
    bias = {k: float(b) for k, b in (bias or {}).items() if b and math.isfinite(b)}

    def sample_weight(s, k):
        """W of key k at sample s: the secant, or for a biased path (dV - b) / dp."""
        _, dp, W = s["terms"][k]
        if k in bias:
            dV = s.get("volumes", {}).get(k)
            return (dV - bias[k]) / dp if dV is not None and dp != 0 else math.nan
        return W

    # each point is weighted in its polynomial fit by how much p_a responds to that W there
    sensitivities = [activation_sensitivities({k: (s["terms"][k][1], sample_weight(s, k)) for k in s["terms"]})
                     for s in samples]
    data = {}
    for k in keys:
        dp = np.array([s["terms"][k][1] for s in samples if k in s["terms"]], float)
        W = np.array([sample_weight(s, k) for s in samples if k in s["terms"]], float)
        S = np.array([sens[k] for s, sens in zip(samples, sensitivities) if k in s["terms"]], float)
        for side, mask in (("+", dp > 0), ("-", dp < 0)):
            data[k, side] = (dp[mask], W[mask], S[mask])
    # pieces with usable points; a weight with none at all (e.g. a rigid, incompressible activation
    # fluid) has W = 0 and is left out of the equation
    pieces = [(k, side) for k in keys for side in SIDES if _fit_piece(data[k, side], 0)["kind"] != "none"]
    keys = [k for k in keys if any(piece[0] == k for piece in pieces)]

    cache = {}

    def fit(piece, degree):
        if (piece, degree) not in cache:
            cache[piece, degree] = _fit_piece(data[piece], degree)
        return cache[piece, degree]

    # highest useful degree per piece: the first that is exact, else what the number of points allows
    caps = {}
    for piece in pieces:
        d = 0
        while d < fit(piece, 0)["max_degree"] and not fit(piece, d)["exact"]:
            d += 1
        caps[piece] = d

    def weights_for(degrees):
        out = {}
        for k in keys:
            out[k] = {"kind": "piecewise",
                      "sides": {side: fit((k, side), degrees[k, side]) if (k, side) in degrees else None
                                for side in SIDES}}
            if k in bias:
                out[k]["bias"] = bias[k]
        return out

    def predict(fits, only=None):
        """p_a per sample from the equation. With `only` (a piece), only that piece uses its polynomial:
        the other side of its weight and every other weight take the sample's value (where a sample
        value is missing, dp ~ 0 and the fitted value is used: it barely matters)."""
        out = []
        total_bias = sum(bias.get(k, 0.0) for k in keys)
        for s in samples:
            terms = []
            for k in keys:
                if k not in s["terms"] or not math.isfinite(s["terms"][k][0]):
                    continue
                p, _, W = s["terms"][k]
                W = sample_weight(s, k)
                c = weight_coefficients(fits[k])
                if only is not None and math.isfinite(W):
                    if k != only[0]:
                        c = [W]
                    else:
                        c = (c[0], [W]) if only[1] == "+" else ([W], c[1])
                terms.append((p, c, k == "W0"))
            out.append(solve_activation(terms, s["p_a"], bias=total_bias, all_roots=True) if terms else [math.nan])
        return out

    def worst(roots):
        """Per sample the error of its worst root: a multivalued equation only meets the tolerance if
        every p_a it allows does (used on its own, nothing picks the root nearest the simulation)."""
        errors = [max(abs(x - s["p_a"]) for x in rs) for rs, s in zip(roots, samples)]
        return errors, max(errors) if errors else 0.0

    def nearest(roots):
        return [min(rs, key=lambda x: abs(x - s["p_a"])) for rs, s in zip(roots, samples)]

    evaluated = 0

    class OutOfBudget(Exception):
        pass

    def evaluate(degrees):
        nonlocal evaluated
        if check is not None:
            check()
        if evaluated >= max_evaluations:
            raise OutOfBudget()
        evaluated += 1
        fits = weights_for(degrees)
        roots = predict(fits)
        errors, error = worst(roots)
        return {"fits": fits, "predicted": nearest(roots), "errors": errors, "error": error,
                "ambiguous": sum(len(rs) > 1 for rs in roots)}

    if method not in (LOWEST_TOTAL, BIGGEST_ERROR):
        raise ValueError(f"unknown method {method!r}")
    max_evaluations = max(1, int(max_evaluations))  # the constant weights are always tried
    best, meeting, stopped = None, None, False
    try:
        if method == LOWEST_TOTAL:
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
        else:
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
    except OutOfBudget:
        stopped = True
        if meeting is not None:  # met the tolerance in the total order that was cut short
            best = meeting

    own = {k: {} for k in keys}
    for piece in pieces:
        own[piece[0]][piece[1]] = worst(predict(best["fits"], only=piece))[1]
    return {"fits": best["fits"], "error": best["error"], "errors": best["errors"],
            "predicted": best["predicted"], "weight_errors": own, "tolerance": tolerance,
            "met": best["error"] <= tolerance, "method": method, "evaluated": evaluated, "stopped": stopped,
            "ambiguous": best["ambiguous"],
            "bias": sum(f.get("bias", 0.0) for f in best["fits"].values())}


def _fit_piece(points, degree):
    dp, W, sensitivity = points
    return polyfit_weight(dp, W, degree, sensitivity=sensitivity)


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
    The neuron equation up to the pre-activation pressure, as LaTeX pieces.
      inputs  : [(label, fit)] with fit a piecewise weight from fit_neuron_equation() for W_j against
                dp_j = p_j - p_a
      chamber : optional (fit, p0) for the pre-activation chamber's own compliance W_0(p_a - p_0)
    Returns {"main", "weights": [{"symbol", "variable", "pieces": [(polynomial, condition)],
    "definition"}], "bias": None or {"symbol", "value", "parts": [(symbol, label, value)], "definition"},
    "note"}; see equation_align() and equation_lines() to put it together. The weights depend on p_a
    through dp_j, so the main equation is implicit in p_a. A fit with "bias" (a biased path, see
    fit_neuron_equation) adds B = sum b_j to the numerator.
    """
    pa = rf"p_{{{latex_name(activation)}}}"
    terms, weights, parts = [], [], []
    for j, (label, fit) in enumerate(inputs, start=1):
        W, p, dp = rf"W_{{{j}}}", rf"p_{{{latex_name(label)}}}", rf"\Delta p_{{{j}}}"
        terms.append((W, p))
        definition = rf"{dp} = {p} - {pa}"
        if fit.get("bias"):
            b = rf"b_{{{j}}}"
            parts.append((b, label, fit["bias"]))
            definition += rf", \quad {W} = (\Delta V_{{{j}}} - {b}) / {dp}"
        weights.append({"symbol": W, "variable": dp, "pieces": _latex_pieces(fit, dp, precision),
                        "definition": definition})
    if chamber is not None:
        fit, p0 = chamber
        W, dp = r"W_{0}", r"\Delta p_{0}"
        terms.append((W, "p_{0}"))
        weights.append({"symbol": W, "variable": dp, "pieces": _latex_pieces(fit, dp, precision),
                        "definition": rf"{dp} = {pa} - p_{{0}}, \quad p_{{0}} = {latex_number(p0, precision)}"
                                      rf"\ \mathrm{{{p_unit}}}"})
    numerator = " + ".join(rf"{W}\,{p}" for W, p in terms)
    denominator = " + ".join(W for W, _ in terms)
    bias = None
    if parts:
        numerator += " + B"
        value = sum(v for _, _, v in parts)
        volume = w_unit.split("/")[0].replace("^3", "^{3}")
        sum_text = " + ".join(symbol for symbol, _, _ in parts)
        bias = {"symbol": "B", "value": value, "parts": parts,
                "definition": rf"B = {sum_text} = {latex_number(value, precision)}\ \mathrm{{{volume}}}"}
    main = rf"{pa} = \frac{{{numerator}}}{{{denominator}}}"
    note = rf"\mathrm{{W\ in\ {w_unit.replace('^3', '^{3}')},\ pressures\ in\ {p_unit}}}"
    return {"main": main, "weights": weights, "bias": bias, "note": note}


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
    if equation.get("bias"):
        rows.append(equation["bias"]["definition"].replace(" = ", " &= ", 1))
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
    if equation.get("bias"):
        lines.append((equation["bias"]["definition"], None, 1))
    lines.append((equation["note"], None, 2))
    return lines
