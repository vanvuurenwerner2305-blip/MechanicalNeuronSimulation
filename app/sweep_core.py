"""Parameter sweep over one or two input chamber pressures (the neuron's response surface).

Besides the chamber pressures, every point records the mechanical weight of every input path into
the chosen pre-activation chamber: W_j = dV_j / (p_j - p_a), with dV_j the volume the path pushes into
the pre-activation chamber (see membrane_sim.characterise; a path lumps its membranes and intermediate
chambers), plus the pre-activation chamber's own compliance W_0. After the sweep each W is fitted as
the lowest-degree polynomial W(dp) that passes within the tolerance of every point (degree 0: W is
constant; points with dp = 0 are left out), and the neuron equation is written out in LaTeX.
"""
import csv
import math
import time

from membrane_sim.characterise import (LOWEST_TOTAL, activation_sensitivities, bias_volumes, fit_neuron_equation,
                                       input_paths, input_weights, is_neutral, polynomial_text,
                                       rebuild_activation_pressure, weight_coefficients, weight_degree,
                                       equation_align, neuron_equation_latex, solve_activation)

from .builder import KPA, build_environment, generate_mesh
from .project import CONSTANT

FLUID = "W0"  # key of the pre-activation chamber's own compliance among the weights
# per-path quantities: key -> (label, unit, factor from model units: mm, MPa)
WEIGHT_FIELDS = {
    "p": ("p", "kPa", 1.0 / KPA),
    "dp": ("dp", "kPa", 1.0 / KPA),
    "dV": ("dV", "mm3", 1.0),
    "W": ("W", "mm3/kPa", KPA),
    "W_tan": ("dV/dp tangent", "mm3/kPa", KPA),
}


# quantities of an activation membrane (a linked activation-function design) per sweep point, besides its named
# outputs (keys "out:<name>", kPa)
ACTIVATION_QUANTITIES = {
    "dp": ("pre-activation Δp", "kPa"),
    "area": ("tube area", "mm2"),
    "mdot": ("mass flow", "kg/s"),
}


def activation_row(out) -> dict:
    """A design's outputs at one point (ActivationDesign.outputs) as flat sweep values: named outputs first."""
    row = {f"out:{name}": v for name, v in out["outputs"].items()}
    row.update({k: out[k] for k in ACTIVATION_QUANTITIES})
    return row


def activation_label(key):
    """(name, unit) of a flat activation value."""
    return (key[4:], "kPa") if key.startswith("out:") else ACTIVATION_QUANTITIES[key]


def point_weights(build, activation_index, load_factor):
    """Weights of every input path (label -> {key: value in WEIGHT_FIELDS units, "shells": names}),
    the pre-activation chamber's own term under FLUID, and the pre-activation pressure rebuilt from them."""
    raw = input_weights(build.env, build.volumes[activation_index], load_factor)
    scale = lambda w: {k: w[k] * f for k, (_, _, f) in WEIGHT_FIELDS.items()}
    weights = {w["input"]: {**scale(w), "shells": w["shells"], "biased": w.get("biased", False),
                            "bias_chambers": w.get("bias_chambers", [])} for w in raw["inputs"]}
    weights[FLUID] = {**scale(raw["chamber"]), "shells": []}
    return weights, rebuild_activation_pressure(raw) / KPA


def activation_value(row, index, field):
    """An activation membrane's output at a sweep point (nan when it was not built)."""
    return row.get("act", {}).get(index, {}).get(field, math.nan)


def sample_weight(row, key):
    """W of a weight at a sweep point: the secant dV / dp, or for a biased path (dV - b) / dp with its
    measured bias b (row["bias"], see run_sweep)."""
    w = row["W"][key]
    b = row.get("bias", {}).get(key)
    if b is None:
        return w["W"]
    return (w["dV"] - b) / w["dp"] if w["dp"] != 0 and math.isfinite(w["dp"]) else math.nan


def weight_sensitivity(row, key):
    """dp_a/dW of weight `key` at a sweep point (Article 2, Eq. 4.10): (p_k - p_a) / sum of all W, in
    kPa per mm3/kPa. For the pre-activation fluid's own W_0 the pressure difference is p_0 - p_a = -dp."""
    return activation_sensitivities({k: (w["dp"], sample_weight(row, k)) for k, w in row["W"].items()})[key]


def equation_fit(rows, activation_index, tolerance, method=LOWEST_TOTAL, check=None):
    """Fit the weights of the converged rows so the equation meets `tolerance` (kPa) on p_a.
    check() is called between degree combinations and may raise to cancel (Worker.check)."""
    rows = [r for r in rows if r["converged"] and r["W"]]
    samples = [{"p_a": r["P"][activation_index],
                "terms": {k: (w["p"], w["dp"], w["W"]) for k, w in r["W"].items()},
                "volumes": {k: w.get("dV", math.nan) for k, w in r["W"].items()}} for r in rows]
    bias = rows[0].get("bias", {}) if rows else {}
    return fit_neuron_equation(samples, tolerance, method, check=check, bias=bias)


def measure_bias(build, project, activation_index, callback, log):
    """{path label: b_j (mm3)} of the biased paths into the pre-activation chamber, from one solve at the
    neutral state: every constant-pressure input at the pre-activation chamber's rest pressure p_0. There an
    unbiased path pushes nothing; a path through a chamber that is not neutral at rest (e.g. a gas weight
    chamber filled above ambient) pushes its bias. {} when no path is biased (no extra solve)."""
    activation = build.volumes[activation_index]
    volumes = list(build.volumes.values())
    biased = [label for label, _, _, between in input_paths(activation, volumes)
              if any(not is_neutral(v) for v in between)]
    if not biased:
        return {}
    inputs = [v for i, v in build.volumes.items() if project.parts[i].props.get("model") == CONSTANT]  # not vents
    saved = {id(v): v.P0 for v in inputs}
    p0 = activation.pressure(0.0)
    log(f"Measuring the bias of {', '.join(biased)} (all inputs at the pre-activation chamber's rest pressure)…")
    try:
        for v in inputs:
            v.P0 = p0
        result = build.solve(project.solver, callback, fixed_contact=True)
        if not result.converged:
            log("The bias could not be measured (the neutral state did not converge); fitting without it.")
            return {}
        return bias_volumes(input_weights(build.env, activation, result.load_factor))
    finally:
        for v in inputs:
            v.P0 = saved[id(v)]


NO_FIT = {"kind": "none", "sides": {"+": None, "-": None}}
SIDE_NAME = {"+": "dp > 0", "-": "dp < 0"}
SIDE_COLOR = {"+": "#d9480f", "-": "#7048e8"}


def describe_side(fit):
    if fit["degree"] == 0:
        return f"W = {fit['coefficients'][0]:.6g} mm3/kPa, constant ({fit['points']} points)"
    return (f"W(dp) = {polynomial_text(fit['coefficients'], precision=6)}, degree {fit['degree']} "
            f"({fit['points']} points)")


def describe_fit(fit):
    """One text per side of a piecewise weight."""
    sides = [side for side in ("+", "-") if fit["sides"][side] is not None]
    if not sides:
        return "no points with a pressure difference (W = 0)"
    if len(sides) == 1:
        return f"{describe_side(fit['sides'][sides[0]])}, for all dp (sampled only {SIDE_NAME[sides[0]]})"
    return "; ".join(f"{SIDE_NAME[side]}: {describe_side(fit['sides'][side])}" for side in sides)


def sweep_points(a_values, b_values):
    """Grid points in serpentine order, so every solve warm-starts from a close neighbour."""
    points = []
    for i, a in enumerate(a_values):
        js = range(len(b_values)) if i % 2 == 0 else reversed(range(len(b_values)))
        points += [(i, j, a, b_values[j]) for j in js]
    return points


def run_sweep(worker, cad, project, mesh_data, a_index, a_values, b_index, b_values, activation_index=None):
    if mesh_data is None:
        worker.log("Generating mesh…")
        mesh_data = generate_mesh(cad, project)
    build = build_environment(cad, mesh_data, project)
    chambers = list(build.volumes.items())
    if activation_index is not None and not build.volumes[activation_index].is_closed:
        raise ValueError(f"{project.parts[activation_index].name} is not a closed chamber: "
                         "it can not be the pre-activation chamber.")
    # one contact stiffness for the whole sweep, sized for its highest pressure
    p_max = max([abs(v.P0) for v in build.volumes.values()] + [abs(x) * KPA for x in a_values]
                + ([abs(x) * KPA for x in b_values] if b_index is not None else []))
    build.set_contact_stiffness(p_max)
    points = sweep_points(list(a_values), list(b_values) if b_index is not None else [None])
    bias = {}
    if activation_index is not None:
        bias = measure_bias(build, project, activation_index, lambda lam, it, r: worker.check(), worker.log)
    first = True
    for k, (i, j, a, b) in enumerate(points):
        worker.check()
        build.volumes[a_index].P0 = a * KPA
        if b_index is not None:
            build.volumes[b_index].P0 = b * KPA
        callback = lambda lam, it, r: worker.check()
        start = time.time()
        result = None if first else build.solve(project.solver, callback, warm_start=True, load_steps=2,
                                                 fixed_contact=True)
        if result is None or not result.converged:
            result = build.solve(project.solver, callback, fixed_contact=True)
        first = False
        row = {"i": i, "j": j, "a": a, "b": b, "converged": result.converged, "time": time.time() - start,
               "P": {c: v.P / KPA for c, v in chambers}, "dV": {c: v.delta_volume for c, v in chambers},
               "W": {}, "p_a_rebuilt": math.nan,
               "act": {i: activation_row(out) for i, out in build.activation_outputs().items()},
               "extrapolated": build.range_warnings(project.parts)}
        for w in row["extrapolated"]:
            worker.log("Warning: " + w)
        if activation_index is not None:
            row["W"], row["p_a_rebuilt"] = point_weights(build, activation_index, result.load_factor)
            row["bias"] = bias
        if getattr(worker, "wants_coords", False):  # the deformed sheets (live view of the API), build.shells order
            row["coords"] = [s.x.detach().cpu().numpy().copy() for s in build.shells.values()]
        worker.item.emit(row)
        worker.report((k + 1) / len(points), f"Sweep point {k + 1}/{len(points)}")
    return mesh_data


# -----------------------------
# After a sweep: the equation, its LaTeX and the exported files (shared by the dialog and the API)
# -----------------------------

def weight_keys_of(rows):
    """The weights of a sweep with a pre-activation chamber: the input paths, then W0."""
    row = next((r for r in rows if r.get("W")), None)
    if row is None:
        return []
    return [k for k in row["W"] if k != FLUID] + ([FLUID] if FLUID in row["W"] else [])


def weight_label(key, activation_name):
    return f"{activation_name} fluid (W0)" if key == FLUID else key


def fits_of(fit, weight_keys):
    return {k: fit["fits"].get(k, NO_FIT) for k in weight_keys} if fit else {}


def equation_keys(fit, weight_keys):
    """Weights in the order of the equation's lines: the inputs, then W0 when it is in the equation."""
    fits = fits_of(fit, weight_keys)
    keys = [k for k in weight_keys if k != FLUID]
    return keys + ([FLUID] if FLUID in fits and fits[FLUID]["kind"] != "none" else [])


def path_shells(rows, key):
    """The membranes on the path of weight `key`."""
    return next((r["W"][key]["shells"] for r in rows if key in r["W"]), [])


def equation_latex(rows, fit, weight_keys, activation_name, precision=6):
    """The neuron equation (neuron_equation_latex() pieces), or None without weights."""
    if not weight_keys or not rows or fit is None:
        return None
    fits = fits_of(fit, weight_keys)
    inputs = [(k, fits[k]) for k in weight_keys if k != FLUID]
    chamber = None
    if FLUID in fits and fits[FLUID]["kind"] != "none":
        p0 = next(r["W"][FLUID]["p"] for r in rows if FLUID in r["W"])
        chamber = (fits[FLUID], p0)
    return neuron_equation_latex(inputs, chamber, activation_name, precision=precision)


def latex_source(rows, fit, weight_keys, activation_name):
    """The equation as an align block, with how well it reproduces the sweep and every weight as comments."""
    equation, fits = equation_latex(rows, fit, weight_keys, activation_name), fits_of(fit, weight_keys)
    if equation is None:
        return ""
    verdict = "within" if fit["met"] else "NOT within (every weight is at its highest useful degree)"
    notes = [f"% Solved for p_a, the equation reproduces all {len(fit['errors'])} sweep points to "
             f"{fit['error']:.3g} kPa, {verdict} the tolerance of {fit['tolerance']:.4g} kPa.",
             f"% Degrees by {fit['method']}, one polynomial per sign of dp: total order "
             f"{sum(weight_degree(f) for f in fit['fits'].values())}, {fit['evaluated']} combinations tried."]

    def own(k):
        errors = fit["weight_errors"].get(k, {})
        return ("; alone it puts p_a off by " + ", ".join(f"{e:.3g} kPa ({SIDE_NAME[side]})"
                                                         for side, e in errors.items())) if errors else ""
    notes += [f"% W_{j} ({k}, path through {', '.join(path_shells(rows, k))}): {describe_fit(fits[k])}{own(k)}"
              for j, k in enumerate([k for k in weight_keys if k != FLUID], start=1)]
    if FLUID in fits:
        notes.append(f"% W_0 ({weight_label(FLUID, activation_name)}): {describe_fit(fits[FLUID])}{own(FLUID)}")
    if fit.get("bias"):
        notes.append(f"% B = {fit['bias']:.6g} mm3: measured bias of the biased paths (every input at the "
                     "pre-activation chamber's rest pressure); their W = (dV - b) / dp.")
    return "\n".join(notes) + "\n" + equation_align(equation) + f"% {equation['note']}\n"


def equation_activation(rows, fit, weight_keys, swept):
    """Every p_a the fitted equation allows with the swept inputs at `swept` {input name: kPa}, sorted; every
    other pressure (inputs not swept, the ambient, p_0) as in the sweep."""
    row = next(r for r in rows if r["converged"] and r["W"])
    terms = []
    for k in equation_keys(fit, weight_keys):
        p = swept.get(k, row["W"][k]["p"]) if k != FLUID else row["W"][k]["p"]
        if math.isfinite(p):
            terms.append((p, weight_coefficients(fit["fits"][k]), k == FLUID))
    return solve_activation(terms, bias=fit.get("bias", 0.0), all_roots=True) if terms else [math.nan]


def activation_keys(rows, index):
    """The values of linked part `index`, from the first sweep point (its design's outputs are known then)."""
    first = rows[0].get("act", {}).get(index) if rows else None
    return list(first) if first else list(ACTIVATION_QUANTITIES)


def write_sweep_csv(path, rows, parts, a_index, b_index, chambers, linked, weight_keys, activation_name=""):
    """Every sweep point: inputs, chamber pressures and volume changes, linked designs' outputs, weights."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [f"{parts[a_index].name} [kPa]"]
        if b_index is not None:
            header.append(f"{parts[b_index].name} [kPa]")
        header += [f"P {parts[c].name} [kPa]" for c in chambers]
        header += [f"dV {parts[c].name} [mm3]" for c in chambers]
        header += [f"{parts[i].name} {'{} [{}]'.format(*activation_label(k))}" for i in linked
                   for k in activation_keys(rows, i)]
        if weight_keys:
            header.append("p_a from weights [kPa]")
        header += [f"{name} {weight_label(k, activation_name)} [{unit}]"
                   for k in weight_keys for name, unit, _ in WEIGHT_FIELDS.values()]
        writer.writerow(header + ["converged"])
        for r in sorted(rows, key=lambda r: (r["a"], r["b"] if r["b"] is not None else 0)):
            writer.writerow([r["a"]] + ([r["b"]] if b_index is not None else [])
                            + [r["P"][c] for c in chambers] + [r["dV"][c] for c in chambers]
                            + [activation_value(r, i, k) for i in linked for k in activation_keys(rows, i)]
                            + ([r["p_a_rebuilt"]] if weight_keys else [])
                            + [r["W"].get(k, {}).get(field, math.nan) for k in weight_keys for field in WEIGHT_FIELDS]
                            + [r["converged"]])


def write_weights_csv(path, rows, fit, weight_keys, activation_name):
    """One line per weight and side of dp: its W(dp) polynomial, and the equation's error on p_a."""
    fits = fits_of(fit, weight_keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["weight", "input", "dp", "side", "membranes on the path", "degree",
                         "W(dp) coefficients c0 c1 c2 ... [mm3/kPa, dp in kPa]", "points",
                         "dp min [kPa]", "dp max [kPa]", "equation error on p_a [kPa]",
                         "equation tolerance [kPa]"])
        inputs = [k for k in weight_keys if k != FLUID]
        for name, k in [(f"W_{j}", k) for j, k in enumerate(inputs, start=1)] + \
                       ([("W_0", FLUID)] if FLUID in fits else []):
            dp = f"p_a - p_0 ({activation_name})" if k == FLUID else f"p({k}) - p({activation_name})"
            sides = {side: w for side, w in fits[k]["sides"].items() if w is not None}
            for side, w in sides.items():
                applies = SIDE_NAME[side] if len(sides) == 2 else f"all dp (sampled only {SIDE_NAME[side]})"
                writer.writerow([name, weight_label(k, activation_name), dp, applies, " ".join(path_shells(rows, k)),
                                 w["degree"], " ".join(f"{c:.10g}" for c in w["coefficients"]), w["points"],
                                 *w["dp_range"], fit["error"], fit["tolerance"]])
        for j, k in enumerate(inputs, start=1):
            b = fits[k].get("bias")
            if b:
                writer.writerow([f"b_{j}", weight_label(k, activation_name), "", "bias volume [mm3]",
                                 " ".join(path_shells(rows, k)), "", f"{b:.10g}", "", "", "", fit["error"],
                                 fit["tolerance"]])
