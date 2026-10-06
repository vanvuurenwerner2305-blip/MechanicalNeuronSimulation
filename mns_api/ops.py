"""
The simulations behind the commands, per space:

  neuron      check (mesh + model: which chamber loads which membrane), solve (one static solve), sweep (one or
              two input pressures)
  activation  check (tube, fluids, flow connections, bonds), study (the Δp sweep that makes the design usable)
  full        check, solve (once), characterise (a grid over chamber parameters, stored in the .mfn)

Every run writes its full results under the design's results/ folder, figures under results/ and renders/,
records a short summary and key metrics in state.json, and reports progress and shapes to the live view.
Command output is the short summary.
"""
import csv
import math
import time
from pathlib import Path

import numpy as np

from app.builder import KPA, build_environment, generate_mesh
from app.meshview import body_kind
from app.project import (ACTIVATION_MEMBRANE, CHAMBER, CONSTANT, DEFORMABLE, IDEAL_GAS, INCOMPRESSIBLE, VENT)

from . import plots
from .headless import HeadlessWorker
from .model import activation_metrics, apply_sets, load_full, load_model
from .util import ApiError, rounded, write_json

MODEL_SHORT = {CONSTANT: "input", IDEAL_GAS: "gas", INCOMPRESSIBLE: "liquid", VENT: "vent"}


def _deadline(max_minutes):
    return time.time() + 60.0 * max_minutes if max_minutes else None


def _shell_frames(shells, parts, coords=None):
    """[(name, kind, faces, x, |u|)] of simulated bodies for the live view."""
    out = []
    for k, (i, s) in enumerate(shells.items()):
        faces = s.faces_np if hasattr(s, "faces_np") else s.faces.cpu().numpy()
        x = coords[k] if coords is not None else s.x.detach().cpu().numpy()
        X = s.X.detach().cpu().numpy()
        out.append((parts[i].name, body_kind(s), faces, x, np.linalg.norm(np.asarray(x) - X, axis=1)))
    return out


# -----------------------------
# Neuron space
# -----------------------------

def default_preactivation(project):
    closed = [i for i, p in enumerate(project.parts)
              if p.role == CHAMBER and p.props.get("model") in (IDEAL_GAS, INCOMPRESSIBLE)]
    named = [i for i in closed if "activ" in project.parts[i].name.lower()]
    return (named or closed or [None])[0]


def chamber_index(project, name, what="chamber"):
    for i, p in enumerate(project.parts):
        if p.name == name:
            if p.role != CHAMBER:
                raise ApiError(f"{name} is not a fluid chamber ({p.role})")
            return i
    chambers = [p.name for p in project.parts if p.role == CHAMBER]
    raise ApiError(f"No {what} {name!r}", f"Chambers: {', '.join(chambers)}")


def neuron_check(design):
    cad, project = load_model(design)
    parts = project.parts
    start = time.time()
    mesh = generate_mesh(cad, project)
    build = build_environment(cad, mesh, project)
    couplings = {parts[c].name: [{"sheet": parts[k.shell_index].name, "side": k.side,
                                  "coverage": round(k.coverage, 3)} for k in cs]
                 for c, cs in build.couplings.items()}
    sheets = {parts[i].name: {"role": parts[i].role, "thickness_mm": round(t, 5),
                              "elements": len(mesh.midsurfaces[i].faces),
                              "fixed_nodes": int(build.shells[i].fixed.sum())}
              for i, t in build.thickness.items()}
    chambers = {parts[c].name: {"model": MODEL_SHORT.get(parts[c].props.get("model"), parts[c].props.get("model")),
                                "pressure_kPa": parts[c].props.get("pressure"), "closed": v.is_closed}
                for c, v in build.volumes.items()}
    pre = default_preactivation(project)
    links = {}
    for i, link in build.activation.items():
        names = {c: parts[c].name for c in link.sides}
        links[parts[i].name] = {"design": Path(link.path).parent.name or Path(link.path).name,
                                "simulated_dp_range_kPa": list(link.design.dp_range),
                                "driving_side": [names[c] for c, s in link.sides.items() if s > 0],
                                "tube_side": [names[c] for c, s in link.sides.items() if s < 0],
                                "outputs": link.design.output_names}
    out = {"design": design.id, "seconds": round(time.time() - start, 1), "chambers": chambers,
           "chamber_loads": couplings, "sheets": sheets,
           "inputs": [n for n, c in chambers.items() if c["model"] == "input"],
           "preactivation_default": parts[pre].name if pre is not None else None,
           "activation_links": links,
           "contact_stiffness_MPa_per_mm": rounded(build.contact_stiffness), "warnings": build.warnings}
    write_json(design.path("results", "check.json"), out)
    design.set_state(checked=time.strftime("%Y-%m-%dT%H:%M:%S"), status=_status(design, "checked"))
    return out


def _status(design, new):
    order = ["created", "built", "checked", "solved", "swept", "studied", "characterised", "reported"]
    old = design.state().get("status", "created")
    return new if order.index(new) >= order.index(old if old in order else "created") else old


def neuron_solve(design, sets=(), render=True, max_minutes=None):
    cad, project = load_model(design)
    apply_sets(project, sets)
    parts = project.parts
    events = design.study.events
    mesh = generate_mesh(cad, project)
    build = build_environment(cad, mesh, project)
    deadline = _deadline(max_minutes)
    seen = [0]

    def callback(lam, iteration, residual):
        if deadline and time.time() > deadline:
            raise TimeoutError("the run reached its time limit (--max-minutes)")
        events.progress(design.id, lam, f"load {lam:.0%}, Newton iteration {iteration}, |R| = {residual:.2e}")
        if len(build.env.history) > seen[0]:
            seen[0] = len(build.env.history)
            events.frame(design.id, _shell_frames(build.shells, parts), f"load {lam:.0%}")

    start = time.time()
    events.progress(design.id, 0.0, "solving", force=True)
    result = build.solve(project.solver, callback=callback)
    events.frame(design.id, _shell_frames(build.shells, parts), "solved" if result.converged else "last converged",
                 force=True)
    events.clear_progress()
    chambers = {parts[c].name: {"P_kPa": v.P / KPA, "dV_mm3": v.delta_volume} for c, v in build.volumes.items()}
    sheets = {}
    for i, s in build.shells.items():
        x, X = s.x.detach().cpu().numpy(), s.X.detach().cpu().numpy()
        F = s.faces.cpu().numpy()
        area = 0.5 * np.linalg.norm(np.cross(x[F[:, 1]] - x[F[:, 0]], x[F[:, 2]] - x[F[:, 0]]), axis=1)
        stretch = area / s.rest_area.cpu().numpy()
        sheets[parts[i].name] = {"max_displacement_mm": float(np.linalg.norm(x - X, axis=1).max()),
                                 "max_area_stretch": float(stretch.max())}
    activation = {}
    for i, out in build.activation_outputs().items():
        activation[parts[i].name] = {"dp_kPa": out["dp"], "outputs_kPa": out["outputs"], "area_mm2": out["area"],
                                     "mdot_kg_s": out["mdot"], "extrapolated": out["extrapolated"]}
    summary = {"design": design.id, "converged": bool(result.converged), "message": result.message,
               "load_reached": round(result.load_factor, 4), "newton_iterations": int(sum(result.iterations)),
               "seconds": round(time.time() - start, 1), "set": dict(sets), "chambers": rounded(chambers),
               "sheets": rounded(sheets), "activation": rounded(activation),
               "warnings": build.warnings + build.range_warnings(parts)}
    files = {"results": str(design.path("results", "solve.json"))}
    if render:
        from .render import render_deformed
        coords = {i: s.x.detach().cpu().numpy() for i, s in build.shells.items()}
        files["render"] = render_deformed(mesh.surfaces, parts, build.shells, coords,
                                          design.path("renders", "solve.png"),
                                          title=f"{design.id} solve ({'converged' if result.converged else 'NOT converged'})")
    summary["files"] = files
    write_json(files["results"], summary)
    metrics = {f"solve:P_{n}_kPa": c["P_kPa"] for n, c in summary["chambers"].items()}
    metrics.update({f"solve:max_u_{n}_mm": s["max_displacement_mm"] for n, s in summary["sheets"].items()})
    design.record_run("solve", {k: summary[k] for k in ("converged", "seconds", "set")}, metrics)
    design.set_state(status=_status(design, "solved"))
    return summary


def neuron_sweep(design, inputs, sets=(), max_minutes=None, render=True):
    """inputs: [(chamber name, values)] (one or two)."""
    from app.sweep_core import run_sweep
    if not 1 <= len(inputs) <= 2:
        raise ApiError("A neuron sweep takes one or two --input chambers.",
                       "For more parameters use a full neuron (F###) and mns characterise.")
    cad, project = load_model(design)
    apply_sets(project, sets)
    parts = project.parts
    idx = []
    for name, values in inputs:
        i = chamber_index(project, name, "input chamber")
        if parts[i].props.get("model") != CONSTANT:
            raise ApiError(f"{name} is not a constant-pressure input chamber",
                           f"Inputs: {', '.join(p.name for p in parts if p.props.get('model') == CONSTANT)}")
        idx.append((i, list(values)))
    events = design.study.events
    mesh = generate_mesh(cad, project)
    shells_idx = [i for i, p in enumerate(parts) if p.role in DEFORMABLE]
    faces = [mesh.midsurfaces[i].faces for i in shells_idx]
    rests = [mesh.midsurfaces[i].vertices for i in shells_idx]
    n_points = len(idx[0][1]) * (len(idx[1][1]) if len(idx) > 1 else 1)

    def on_item(row):
        k = len(worker.rows)
        where = ", ".join(f"{parts[i].name} = {v:.4g}" for (i, _), v in zip(idx, (row["a"], row["b"])))
        events.progress(design.id, k / n_points, f"point {k}/{n_points}: {where}", force=True)
        coords = row.pop("coords", None)
        if coords is not None:
            events.frame(design.id, [(parts[i].name, "shell", f, x, np.linalg.norm(x - r, axis=1))
                                     for i, f, x, r in zip(shells_idx, faces, coords, rests)], where)

    worker = HeadlessWorker(events, design.id, on_item, _deadline(max_minutes))
    worker.wants_coords = True
    start = time.time()
    a_index, a_values = idx[0]
    b_index, b_values = idx[1] if len(idx) > 1 else (None, [math.nan])
    stopped = None
    try:
        run_sweep(worker, cad, project, mesh, a_index, np.asarray(a_values), b_index, np.asarray(b_values))
    except TimeoutError as exc:
        stopped = str(exc)
    events.clear_progress()
    rows = worker.rows
    if not rows:
        raise ApiError("The sweep produced no points" + (f" ({stopped})" if stopped else ""))
    seconds = time.time() - start
    sweep = {"inputs": [{"chamber": parts[i].name, "values": v} for i, v in idx], "a_index": a_index,
             "b_index": b_index, "set": dict(sets), "seconds": seconds, "stopped": stopped,
             "chambers": [c for c, p in enumerate(parts) if p.role == CHAMBER],
             "linked": [i for i, p in enumerate(parts) if p.role == ACTIVATION_MEMBRANE],
             "rows": [_row_json(r) for r in rows]}
    write_json(design.path("results", "sweep_rows.json"), sweep)
    summary = sweep_summary(design, project, sweep, rows, render=render)
    design.set_state(status=_status(design, "swept"))
    return summary


def _row_json(row):
    out = dict(row)
    for key in ("P", "dV", "act"):
        if key in out:
            out[key] = {str(k): v for k, v in out[key].items()}
    return out


def _row_back(row):
    out = dict(row)
    for key in ("P", "dV", "act"):
        if key in out:
            out[key] = {int(k): v for k, v in out[key].items()}
    return out


def load_sweep(design):
    from .util import read_json
    path = design.folder / "results" / "sweep_rows.json"
    if not path.exists():
        raise ApiError(f"{design.id} has no sweep yet.", f"Run mns sweep {design.id} --input <chamber>=0:20:5 first.")
    sweep = read_json(path)
    return sweep, [_row_back(r) for r in sweep["rows"]]


def sweep_summary(design, project, sweep, rows, render=True):
    from app.sweep_core import write_sweep_csv
    parts = project.parts
    a_index, b_index = sweep["a_index"], sweep["b_index"]
    csv_path = design.path("results", "sweep.csv")
    write_sweep_csv(csv_path, rows, parts, a_index, b_index, sweep["chambers"], sweep["linked"])
    files = {"csv": str(csv_path)}
    ok = [r for r in rows if r["converged"]]
    out = {"design": design.id, "points": len(rows), "converged": len(ok), "seconds": round(sweep["seconds"], 1),
           "inputs": {d["chamber"]: [min(d["values"]), max(d["values"]), len(d["values"])] for d in sweep["inputs"]}}
    if sweep.get("stopped"):
        out["stopped"] = sweep["stopped"]
    extrapolated = sum(bool(r.get("extrapolated")) for r in rows)
    if extrapolated:
        out["extrapolated_points"] = extrapolated
        out["warning"] = ("Some points put the pre-activation outside the activation design's simulated Δp range: "
                          "there the design is EXTRAPOLATED.")
    closed = [c for c in sweep["chambers"] if parts[c].props.get("model") in (IDEAL_GAS, INCOMPRESSIBLE)]
    if ok:
        out["closed_chamber_pressure_ranges_kPa"] = {
            parts[c].name: rounded([min(r["P"][c] for r in ok), max(r["P"][c] for r in ok)]) for c in closed}
        for i in sweep["linked"]:
            for key in rows[0].get("act", {}).get(i, {}):
                vals = [r["act"][i][key] for r in ok if i in r.get("act", {})]
                if vals and key.startswith("out:") or key == "dp":
                    out.setdefault("activation_ranges", {})[f"{parts[i].name}:{key}"] = rounded([min(vals), max(vals)])
    # figures: the pre-activation (and the linked designs' outputs) over the inputs
    a_name = parts[a_index].name
    a_vals = sorted({r["a"] for r in rows})
    targets = [(f"P {parts[c].name} [kPa]", (lambda r, c=c: r["P"][c])) for c in closed[:2]]
    for i in sweep["linked"]:
        for key in rows[0].get("act", {}).get(i, {}):
            if key.startswith("out:"):
                targets.append((f"{parts[i].name} {key[4:]} [kPa]", (lambda r, i=i, key=key: r["act"][i][key])))
    if targets and ok:
        path = design.path("results", "sweep_response.png")
        if b_index is None:
            rs = sorted(ok, key=lambda r: r["a"])
            plots.panels(path, [r["a"] for r in rs], [(label, [("", [f(r) for r in rs])]) for label, f in targets],
                         f"{a_name} [kPa]", title=f"{design.id}: response to {a_name}")
        else:
            b_vals = sorted({r["b"] for r in rows})
            panels = []
            for label, f in targets:
                series = []
                for b in b_vals[:8]:
                    rs = sorted([r for r in ok if r["b"] == b], key=lambda r: r["a"])
                    series.append((f"{parts[b_index].name} = {b:.3g}", [f(r) for r in rs] if len(rs) == len(a_vals)
                                   else np.interp(a_vals, [r["a"] for r in rs], [f(r) for r in rs]) if rs else
                                   [math.nan] * len(a_vals)))
                panels.append((label, series))
            plots.panels(path, a_vals, panels, f"{a_name} [kPa]", title=f"{design.id}: response")
        files["response_plot"] = str(path)
    out["files"] = files
    design.record_run("sweep", {k: out[k] for k in ("points", "converged", "seconds", "inputs")})
    return out



# -----------------------------
# Activation-function space
# -----------------------------

def activation_check(design):
    from app.activation import build_activation, generate_activation_mesh, output_definitions
    from membrane_sim.flow import compile_flow_law, ORIFICE_LAW, SEGMENT_LAW
    cad, project = load_model(design)
    parts = project.parts
    start = time.time()
    problems = []
    for key, st in project.connections.items():
        try:
            compile_flow_law(st.get("law"), ORIFICE_LAW)
        except Exception as exc:
            problems.append(f"connection {key}: {exc}")
    for p in parts:
        if p.props.get("segment_law"):
            try:
                compile_flow_law(p.props["segment_law"], SEGMENT_LAW)
            except Exception as exc:
                problems.append(f"{p.name} segment law: {exc}")
    mesh = generate_activation_mesh(cad, project)
    build = build_activation(cad, mesh, project)
    f = build.flow
    out = {"design": design.id, "seconds": round(time.time() - start, 1),
           "tube": parts[mesh.channel].name, "tube_fluid": parts[f.lumen].name, "segments": f.segments,
           "A0_mm2": rounded(build.A0), "inside_height_mm": rounded(build.channel_height),
           "membrane_area_mm2": rounded(build.membrane_area), "fixed_tube_nodes": build.fixed_tube_nodes,
           "connections": [{"key": c.key.replace("↔", "<->"), "type": st["type"].split(" ")[0].lower(),
                            "area_mm2": rounded(c.area)} for c, st in f.connections],
           "constant_fluids_kPa": {f.names[j]: v for j, v in f.constant.items()},
           "bonds": [{"membrane": parts[i].name, "to": parts[j].name, "nodes": n} for i, j, n in build.bonds],
           "outputs": output_definitions(parts, {"lumen": parts[f.lumen].name,
                                                 "pressures": [[0.0] * (f.segments + 1)]}),
           "elements": {"tube_tets": len(mesh.volume.tets),
                        "membrane_triangles": sum(len(m.faces) for m in mesh.midsurfaces.values())},
           "warnings": build.warnings + problems}
    write_json(design.path("results", "check.json"), out)
    design.set_state(checked=time.strftime("%Y-%m-%dT%H:%M:%S"), status=_status(design, "checked"))
    return out


def activation_study(design, dp=None, sets=(), max_minutes=None, render=True):
    from app.activation import (DesignFrames, build_activation, generate_activation_mesh, output_definitions,
                                output_pressures, run_study)
    cad, project = load_model(design)
    apply_sets(project, sets)
    if dp is not None:
        project.study.dp_min, project.study.dp_max, project.study.points = float(dp[0]), float(dp[-1]), len(dp)
    parts = project.parts
    events = design.study.events
    mesh = generate_activation_mesh(cad, project)
    build = build_activation(cad, mesh, project)
    deadline = _deadline(max_minutes)

    def callback(k, total, dp_, lam, iteration, residual):
        if deadline and time.time() > deadline:
            raise TimeoutError("the run reached its time limit (--max-minutes)")
        events.progress(design.id, (k + lam) / total, f"Δp = {dp_:.4g} kPa ({k + 1}/{total}), load {lam:.0%}, "
                                                      f"iteration {iteration}")

    def point(row):
        events.frame(design.id, _shell_frames(build.shells, parts),
                     f"Δp = {row['dp']:.4g} kPa: A = {row['area']:.4g} mm²", force=True)

    start = time.time()
    try:
        results, _ = run_study(build, project, callback=callback, point_callback=point)
    finally:
        events.clear_progress()
    seconds = time.time() - start
    project.results = results
    project.save(design.project_path)
    metrics = activation_metrics(results, parts)
    outputs = output_definitions(parts, results)
    out_p = output_pressures(results, outputs)
    folder = design.path("results")
    with open(folder / "study.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dp [kPa]", "area [mm2]", "mdot [kg/s]", "p_end [kPa]"] + [f"{k} [kPa]" for k in out_p]
                   + ["swept volume [mm3]", "travel [mm]", "converged", "flow iterations"])
        for k in range(len(results["dp"])):
            w.writerow([results["dp"][k], results["area"][k], results["mdot"][k], results["p_end"][k]]
                       + [float(v[k]) for v in out_p.values()]
                       + [results["volume"][k], results["travel"][k], results["converged"][k],
                          results["iterations"][k]])
    dps = np.asarray(results["dp"], float)
    panels = [("tube area [mm²]", [("min. section", results["area"])]),
              ("mass flow [g/s]", [("", [1000 * m for m in results["mdot"]])]),
              (f"{next(iter(out_p))} pressure [kPa]" if len(out_p) == 1 else "output pressure [kPa]",
               [(name, list(v)) for name, v in out_p.items()]),
              ("swept volume [mm³]", [("", results["volume"])])]
    files = {"csv": str(folder / "study.csv"),
             "plot": plots.panels(folder / "activation.png", dps, panels, "Δp across the membrane [kPa]",
                                  title=f"{design.id}: activation function")}
    stations = np.asarray(results["stations"], float)
    picks = sorted({0, len(dps) // 3, (2 * len(dps)) // 3, len(dps) - 1})
    files["profiles"] = plots.panels(folder / "profiles.png", stations,
                                     [("area along the tube [mm²]",
                                       [(f"Δp = {dps[k]:.3g}", results["profiles"][k]) for k in picks])],
                                     "position along the tube [mm]", title=f"{design.id}: tube cross-section")
    if render and results.get("fem"):
        from .render import render_frames
        ok = np.nonzero(np.asarray(results["converged"], bool))[0]
        if len(ok):
            frames = DesignFrames(results["fem"], results["dp"], ok[np.argsort(dps[ok])])
            top = float(dps[ok].max())
            files["render"] = render_frames(frames.at(top), design.path("renders", "study.png"),
                                            title=f"{design.id} at Δp = {top:.3g} kPa")
    summary = {"design": design.id, "seconds": round(seconds, 1), "dp_kPa": [float(dps[0]), float(dps[-1]), len(dps)],
               "metrics": rounded(metrics), "outputs": [o["name"] for o in outputs],
               "not_converged_dp": [float(d) for d, c in zip(dps, results["converged"]) if not c], "files": files}
    design.record_run("study", {"seconds": summary["seconds"], "dp": summary["dp_kPa"],
                                "converged": metrics.get("converged")}, metrics)
    design.set_state(status=_status(design, "studied"))
    return summary


# -----------------------------
# Full neuron
# -----------------------------

def _full_model(design):
    from app.cad import CadModel
    full = load_full(design)
    linked = full.linked_project()
    cad = CadModel()
    cad.load_step(linked.step_path)
    linked.match_bodies(cad.bodies)
    return full, linked, cad


def full_check(design):
    full, linked, cad = _full_model(design)
    start = time.time()
    mesh = generate_mesh(cad, linked)
    build = build_environment(cad, mesh, linked)
    parts = linked.parts
    links = {}
    for i, link in build.activation.items():
        names = {c: parts[c].name for c in link.sides}
        links[parts[i].name] = {"simulated_dp_range_kPa": list(link.design.dp_range),
                                "driving_side": [names[c] for c, s in link.sides.items() if s > 0],
                                "tube_side": [names[c] for c, s in link.sides.items() if s < 0]}
    out = {"design": design.id, "seconds": round(time.time() - start, 1), "link": full.link,
           "activation_link": links, "parameters": [f"{p}.{f}" for p, f, _, _ in full.parameters()],
           "recordable": [k for k, _, _ in full.catalogue()], "recorded": full.recorded(),
           "warnings": build.warnings}
    write_json(design.path("results", "check.json"), out)
    design.set_state(checked=time.strftime("%Y-%m-%dT%H:%M:%S"), status=_status(design, "checked"))
    return out


def full_run(design, axes=(), record=None, sets=(), max_minutes=None, render=True):
    """axes: [(part, field, values)]; no axes = one solve at the set values (sets: Part.field=value)."""
    from app.full_neuron import make_dataset, run_grid, shell_bodies, values_or_nan
    full, linked, cad = _full_model(design)
    if record:
        keys = [k for k, _, _ in full.catalogue()]
        bad = [k for k in record if k not in keys]
        if bad:
            raise ApiError(f"--record: unknown key(s) {', '.join(bad)}", f"Keys: {', '.join(keys)}")
        full.record = list(record)
    for key, value in sets:
        name, _, field = key.partition(".")
        part = next((p for p in linked.parts if p.name == name and p.role == CHAMBER), None)
        if part is None:
            raise ApiError(f"--set {key}: no chamber {name!r}")
        apply_sets(linked, [(key, value)])
    known = {(p, f) for p, f, _, _ in full.parameters()}
    for p, f, _ in axes:
        if (p, f) not in known:
            raise ApiError(f"--axis {p}.{f}: not a parameter of this neuron",
                           f"Parameters: {', '.join(f'{a}.{b}' for a, b in sorted(known))}")
    parts = linked.parts
    events = design.study.events
    mesh = generate_mesh(cad, linked)
    shells_idx = [i for i, p in enumerate(parts) if p.role in DEFORMABLE]
    faces = [mesh.midsurfaces[i].faces for i in shells_idx]
    rests = [mesh.midsurfaces[i].vertices for i in shells_idx]
    n_points = int(np.prod([len(v) for _, _, v in axes])) if axes else 1

    def on_item(row):
        k = len(worker.rows)
        where = ", ".join(f"{p}.{f} = {v:.4g}" for (p, f, _), v in zip(axes, row["params"]))
        events.progress(design.id, k / n_points, f"point {k}/{n_points}" + (f": {where}" if where else ""), force=True)
        if row.get("coords") is not None:
            events.frame(design.id, [(parts[i].name, "shell", fc, x, np.linalg.norm(x - r, axis=1))
                                     for i, fc, x, r in zip(shells_idx, faces, row["coords"], rests)], where)

    worker = HeadlessWorker(events, design.id, on_item, _deadline(max_minutes))
    start = time.time()
    build, stopped = None, None
    try:
        build = run_grid(worker, cad, linked, mesh, list(axes))
    except TimeoutError as exc:
        stopped = str(exc)
    events.clear_progress()
    rows = worker.rows
    seconds = time.time() - start
    if not rows:
        raise ApiError("No point was solved" + (f" ({stopped})" if stopped else ""))
    keys = full.recorded()
    folder = design.path("results")
    name = "characterisation" if axes else "solve"
    with open(folder / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([f"{p}.{f}" for p, f, _ in axes] + keys + ["converged", "extrapolated"])
        for r in rows:
            w.writerow(list(r["params"]) + [values_or_nan(r, k) for k in keys]
                       + [r["converged"], bool(r["values"].get("warnings"))])
    files = {"csv": str(folder / f"{name}.csv")}
    ok = [r for r in rows if r["converged"]]
    summary = {"design": design.id, "points": len(rows), "converged": len(ok), "seconds": round(seconds, 1),
               "extrapolated_points": sum(bool(r["values"].get("warnings")) for r in rows)}
    if stopped:
        summary["stopped"] = stopped
    if summary["extrapolated_points"]:
        summary["warning"] = ("The pre-activation Δp left the activation design's simulated range at some points: "
                              "there the design is EXTRAPOLATED. Simulate the design over a wider Δp range.")
    if not axes:
        summary["values"] = rounded({k: values_or_nan(rows[0], k) for k in keys})
        summary["set"] = dict(sets)
    else:
        summary["ranges"] = rounded({k: [min(values_or_nan(r, k) for r in ok), max(values_or_nan(r, k) for r in ok)]
                                     for k in keys} if ok else {})
        bodies = shell_bodies(build, parts) if build is not None else None
        full.characterisation = make_dataset(full, list(axes), rows, bodies, True, seconds)
        full.save(design.project_path)
        files["plot"] = _characterisation_plot(design, axes, rows, keys, full)
        summary["axes"] = {f"{p}.{f}": [min(v), max(v), len(v)] for p, f, v in axes}
        summary["complete"] = full.characterisation["complete"]
    if render and ok:
        from .render import render_deformed
        last = ok[-1]
        if build is not None:
            coords = dict(zip(build.shells.keys(), last["coords"]))
            files["render"] = render_deformed(mesh.surfaces, parts, build.shells, coords,
                                              design.path("renders", f"{name}.png"), title=f"{design.id} {name}")
    summary["files"] = files
    write_json(folder / f"{name}.json", {"summary": summary, "rows": [
        {"params": r["params"], "converged": r["converged"],
         "values": {k: v for k, v in r["values"].items() if k != "warnings"}} for r in rows]})
    metrics = {}
    if not axes:
        metrics = {f"full:{k}": v for k, v in summary["values"].items()}
    else:
        metrics = {f"full:{k}_range": v for k, v in summary["ranges"].items()}
    design.record_run(name, {k: summary[k] for k in ("points", "converged", "seconds")}, metrics)
    design.set_state(status=_status(design, "characterised" if axes else "solved"))
    return summary


def _characterisation_plot(design, axes, rows, keys, full):
    from app.full_neuron import values_or_nan
    labels = {k: (label, unit) for k, label, unit in full.catalogue()}
    first = axes[0]
    x = list(first[2])
    ok = [r for r in rows if r["converged"]]
    panels = []
    for key in keys[:6]:
        series = []
        if len(axes) == 1:
            series.append(("", [next((values_or_nan(r, key) for r in ok if r["params"][0] == v), math.nan)
                                for v in x]))
        else:
            second = axes[1]
            for v2 in list(second[2])[:8]:
                pts = []
                for v in x:  # other axes (third and up) at their first value
                    match = [r for r in ok if r["params"][0] == v and r["params"][1] == v2
                             and all(r["params"][k] == axes[k][2][0] for k in range(2, len(axes)))]
                    pts.append(values_or_nan(match[0], key) if match else math.nan)
                series.append((f"{second[0]}.{second[1]} = {v2:.3g}", pts))
        label, unit = labels.get(key, (key, ""))
        panels.append((f"{label} [{unit}]" if unit else label, series))
    return plots.panels(design.path("results", "characterisation.png"), x, panels, f"{first[0]}.{first[1]}",
                        title=f"{design.id}: characterisation")
