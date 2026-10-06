"""Parameter sweep over one or two input chamber pressures (the neuron's response surface): every point records the
chamber pressures and volume changes and the outputs of any linked activation design. Shared by the sweep dialog
and the command-line API.
"""
import csv
import math
import time

from .builder import KPA, build_environment, generate_mesh

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


def activation_value(row, index, field):
    """An activation membrane's output at a sweep point (nan when it was not built)."""
    return row.get("act", {}).get(index, {}).get(field, math.nan)


def sweep_points(a_values, b_values):
    """Grid points in serpentine order, so every solve warm-starts from a close neighbour."""
    points = []
    for i, a in enumerate(a_values):
        js = range(len(b_values)) if i % 2 == 0 else reversed(range(len(b_values)))
        points += [(i, j, a, b_values[j]) for j in js]
    return points


def run_sweep(worker, cad, project, mesh_data, a_index, a_values, b_index, b_values):
    if mesh_data is None:
        worker.log("Generating mesh…")
        mesh_data = generate_mesh(cad, project)
    build = build_environment(cad, mesh_data, project)
    chambers = list(build.volumes.items())
    # one contact stiffness for the whole sweep, sized for its highest pressure
    p_max = max([abs(v.P0) for v in build.volumes.values()] + [abs(x) * KPA for x in a_values]
                + ([abs(x) * KPA for x in b_values] if b_index is not None else []))
    build.set_contact_stiffness(p_max)
    points = sweep_points(list(a_values), list(b_values) if b_index is not None else [None])
    first = True
    for k, (i, j, a, b) in enumerate(points):
        worker.check()
        build.volumes[a_index].P0 = a * KPA
        if b_index is not None:
            build.volumes[b_index].P0 = b * KPA
        callback = lambda lam, it, r: worker.check()  # noqa: E731
        start = time.time()
        result = None if first else build.solve(project.solver, callback, warm_start=True, load_steps=2,
                                                 fixed_contact=True)
        if result is None or not result.converged:
            result = build.solve(project.solver, callback, fixed_contact=True)
        first = False
        row = {"i": i, "j": j, "a": a, "b": b, "converged": result.converged, "time": time.time() - start,
               "P": {c: v.P / KPA for c, v in chambers}, "dV": {c: v.delta_volume for c, v in chambers},
               "act": {i: activation_row(out) for i, out in build.activation_outputs().items()},
               "extrapolated": build.range_warnings(project.parts)}
        for w in row["extrapolated"]:
            worker.log("Warning: " + w)
        if getattr(worker, "wants_coords", False):  # the deformed sheets (live view of the API), build.shells order
            row["coords"] = [s.x.detach().cpu().numpy().copy() for s in build.shells.values()]
        worker.item.emit(row)
        worker.report((k + 1) / len(points), f"Sweep point {k + 1}/{len(points)}")
    return mesh_data


def activation_keys(rows, index):
    """The values of linked part `index`, from the first sweep point (its design's outputs are known then)."""
    first = rows[0].get("act", {}).get(index) if rows else None
    return list(first) if first else list(ACTIVATION_QUANTITIES)


def write_sweep_csv(path, rows, parts, a_index, b_index, chambers, linked):
    """Every sweep point: inputs, chamber pressures and volume changes, linked designs' outputs."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [f"{parts[a_index].name} [kPa]"]
        if b_index is not None:
            header.append(f"{parts[b_index].name} [kPa]")
        header += [f"P {parts[c].name} [kPa]" for c in chambers]
        header += [f"dV {parts[c].name} [mm3]" for c in chambers]
        header += [f"{parts[i].name} {'{} [{}]'.format(*activation_label(k))}" for i in linked
                   for k in activation_keys(rows, i)]
        writer.writerow(header + ["converged"])
        for r in sorted(rows, key=lambda r: (r["a"], r["b"] if r["b"] is not None else 0)):
            writer.writerow([r["a"]] + ([r["b"]] if b_index is not None else [])
                            + [r["P"][c] for c in chambers] + [r["dV"][c] for c in chambers]
                            + [activation_value(r, i, k) for i in linked for k in activation_keys(rows, i)]
                            + [r["converged"]])
