"""
From a design file to the simulator's project: the CAD is built (mns_api.cadspec), every body gets its role and
properties (validated, with short aliases such as `role: chamber`, `model: input`), and the project is saved
next to it (model.mns / model.mad / model.mfn) so the GUI can open it. Also: loading a built design for a
simulation, and importing existing .mns / .mad / .mfn files as designs.
"""
import copy
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.activation import ActivationProject, StudySettings, connection_key
from app.builder import mesh_sizes, measure_thickness
from app.project import (ACTIVATION_MEMBRANE, ACTIVATION_SPACE, AUTOMATIC, CHAMBER, CHANNEL, CLOSED, CONSTANT,
                         CONSTANT_FLUID, DYNAMIC_FLUID, FLUID, IDEAL_GAS, IGNORE, INCOMPRESSIBLE, MEMBRANE, OPENING,
                         ORIFICE, RIGID, SHEETS, SHELL, SOLID, SPACE_ROLE_FIELDS, SPACE_ROLES, UNASSIGNED, VENT,
                         Project, SolverSettings, part_color, part_opacity)

from .util import ApiError, as_number, evaluate_parameters

SPACE_KEY = {"neuron": "neuron", "activation": ACTIVATION_SPACE}

ROLE_ALIASES = {
    "neuron": {"membrane": MEMBRANE, "shell": SHELL, "rigid": RIGID, "rigid body": RIGID, "obstacle": RIGID,
               "chamber": CHAMBER, "fluid chamber": CHAMBER, "fluid": CHAMBER, "ignore": IGNORE,
               "unassigned": UNASSIGNED, "activation membrane": ACTIVATION_MEMBRANE},
    "activation": {"membrane": MEMBRANE, "shell": SHELL, "rigid": RIGID, "rigid body": RIGID, "channel": CHANNEL,
                   "tube": CHANNEL, "channel (tube)": CHANNEL, "fluid": FLUID, "solid": SOLID, "ignore": IGNORE,
                   "unassigned": UNASSIGNED},
}
# short names for choices: (alias, value); an alias is used only where the value is one of the field's choices
CHOICE_ALIASES = [
    ("input", CONSTANT), ("constant", CONSTANT), ("constant pressure", CONSTANT),
    ("gas", IDEAL_GAS), ("ideal gas", IDEAL_GAS), ("air", IDEAL_GAS),
    ("incompressible", INCOMPRESSIBLE), ("liquid", INCOMPRESSIBLE), ("water", INCOMPRESSIBLE),
    ("vent", VENT), ("open", VENT),
    ("constant", CONSTANT_FLUID), ("supply", CONSTANT_FLUID), ("sink", CONSTANT_FLUID), ("dynamic", DYNAMIC_FLUID),
    ("neo-hookean", "Neo-Hookean (incompressible rubber)"), ("neo_hookean", "Neo-Hookean (incompressible rubber)"),
    ("neohookean", "Neo-Hookean (incompressible rubber)"), ("svk", "St. Venant-Kirchhoff"),
    ("all", "All boundary edges"), ("touching rigid", "Edges touching rigid bodies"),
    ("clamped", "Clamped"), ("pinned", "Pinned"),
    ("fixed", "Fixed"), ("free", "Free (moves as a rigid body)"),
    ("free", "Free (held by what it is bonded to)"), ("fixed", "Fixed where it touches fixed rigid bodies"),
    ("quadratic", "Quadratic (10-node)"), ("linear", "Linear (4-node)"),
]
CONNECTION_TYPES = {"opening": OPENING, "open": OPENING, "orifice": ORIFICE, "closed": CLOSED}


def resolve_role(role, space):
    if role is None:
        return UNASSIGNED
    text = str(role).strip()
    roles = SPACE_ROLES[SPACE_KEY[space]]
    if text in roles:
        return text
    alias = ROLE_ALIASES[space].get(text.lower())
    if alias:
        return alias
    raise ApiError(f"Unknown role {role!r} in the {space} space",
                   f"Roles: {', '.join(sorted(ROLE_ALIASES[space]))}.")


def resolve_choice(value, choices, what):
    if isinstance(choices, str):  # filled at run time (e.g. chamber names)
        return value
    text = str(value).strip()
    if text in choices:
        return text
    low = text.lower()
    for alias, target in CHOICE_ALIASES:
        if alias == low and target in choices:
            return target
    matches = [c for c in choices if c.lower().startswith(low)] or [c for c in choices if low in c.lower()]
    if len(matches) == 1:
        return matches[0]
    short = sorted({a for a, t in CHOICE_ALIASES if t in choices})
    raise ApiError(f"{what}: unknown choice {value!r}", f"Choices: {', '.join(choices)} (short: {', '.join(short)}).")


def resolve_props(part_name, role, props, space, names, study=None):
    """The properties of a body as the simulator stores them: known keys only, numbers evaluated, choices
    resolved, design references (A###) turned into file paths."""
    fields = {f.key: f for f in SPACE_ROLE_FIELDS[SPACE_KEY[space]][role]}
    out = {}
    for key, value in (props or {}).items():
        if key not in fields:
            raise ApiError(f"{part_name}: {role} has no property {key!r}",
                           f"Its properties: {', '.join(fields) or 'none'}.")
        f = fields[key]
        what = f"{part_name}.{key}"
        if f.kind == "float":
            out[key] = as_number(value, names, what)
        elif f.kind == "int":
            out[key] = int(round(as_number(value, names, what)))
        elif f.kind == "choice":
            out[key] = resolve_choice(value, f.choices, what) if f.choices != "__chambers__" else str(value)
        elif f.kind == "file":
            out[key] = design_reference(value, study, what)
        elif f.kind == "outputs":
            if not isinstance(value, list):
                raise ApiError(f"{what}: expected a list of {{name, segment}}")
            out[key] = [{"name": str(o.get("name", "")), "segment": int(as_number(o.get("segment", 1), names, what))}
                        for o in value]
        else:
            out[key] = value
    return out


def design_reference(value, study, what):
    """A .mad file from a design ID (A001) or a path."""
    text = str(value).strip()
    if study is not None and len(text) >= 4 and text[0] == "A" and text[1:4].isdigit():
        design = study.design(text[:4] if len(text) == 4 else text)
        if not design.project_path.exists():
            raise ApiError(f"{what}: {design.id} has not been built yet", f"Run mns cad build {design.id}, then "
                                                                          f"mns study {design.id}.")
        return str(design.project_path)
    if not Path(text).exists():
        raise ApiError(f"{what}: activation design not found: {text}", "Use the ID of an activation design (A###).")
    return str(Path(text).resolve())


# -----------------------------
# Building a design
# -----------------------------

def build_design(design, render=True):
    """CAD + project from the design file; returns the summary printed by `mns cad build`."""
    if design.space == "full":
        return build_full(design)
    from .cadspec import CadBuilder, geometry_report, write_step
    spec = design.spec(reload=True)
    names = evaluate_parameters(spec.get("parameters") or {})
    bodies = [dict(b) for b in spec.get("bodies") or []]
    if not bodies:
        raise ApiError(f"{design.id} has no bodies.", "List the bodies under 'bodies:' (name, shape, role, props).")
    base = spec.get("base_step")
    for b in bodies:
        if "shape" not in b and base:
            b["shape"] = {"import": b.get("name")}
    builder = CadBuilder(bodies, names, base, design.folder)
    built = builder.build()
    cad = write_step(design.step_path, built, model_name=design.name)

    roles = {b["name"]: resolve_role(b.get("role"), design.space) for b in bodies}
    project = project_from_spec(design, spec, cad, names, roles)
    surfaces = cad.mesh(mesh_sizes(cad, project))
    fill_measured(cad, project, surfaces)
    project.step_path = str(design.step_path)
    if design.space == "activation":
        project.results = None
    project.save(design.project_path)
    report = geometry_report(cad, roles)
    write_preview(design, project, surfaces)
    files = {"step": str(design.step_path), "project": str(design.project_path)}
    if render:
        from .render import render_model
        files["render"] = render_model(surfaces, project.parts, design.path("renders", "model.png"),
                                       title=f"{design.id} {design.name}")
    summary = summarise_geometry(design, project, report)
    summary["files"] = files
    design.set_state(status="built", built=summary["built_at"], geometry=summary, checked=None)
    design.study.events.emit("design", f"{design.id} built: {len(project.parts)} bodies", design=design.id,
                             files=list(files.values()))
    return summary


def project_from_spec(design, spec, cad, names, roles):
    cls = ActivationProject if design.space == "activation" else Project
    project = cls(str(design.step_path), [b.name for b in cad.bodies])
    by_name = {b["name"]: b for b in spec.get("bodies")}
    for part in project.parts:
        body = by_name.get(part.name, {})
        project.set_role(part, roles.get(part.name, UNASSIGNED))
        part.props.update(resolve_props(part.name, part.role, body.get("props"), design.space, names, design.study))
    solver = spec.get("solver") or {}
    known = SolverSettings.__dataclass_fields__
    unknown = [k for k in solver if k not in known]
    if unknown:
        raise ApiError(f"Unknown solver setting(s): {', '.join(unknown)}", f"Settings: {', '.join(known)}.")
    for k, v in solver.items():
        setattr(project.solver, k, type(getattr(project.solver, k))(as_number(v, names, f"solver.{k}")))
    if design.space == "activation":
        study = spec.get("study") or {}
        known = StudySettings.__dataclass_fields__
        unknown = [k for k in study if k not in known]
        if unknown:
            raise ApiError(f"Unknown study setting(s): {', '.join(unknown)}", f"Settings: {', '.join(known)}.")
        for k, v in study.items():
            setattr(project.study, k, type(getattr(project.study, k))(as_number(v, names, f"study.{k}")))
        for key, value in (spec.get("connections") or {}).items():
            a, b = [s.strip() for s in str(key).replace("<->", "↔").split("↔")]
            if not isinstance(value, dict):
                value = {"type": value}
            text = str(value.get("type", "opening")).strip()
            kind = text if text in (OPENING, ORIFICE, CLOSED) else CONNECTION_TYPES.get(text.lower().split(" ")[0])
            if kind is None:
                raise ApiError(f"connection {key}: unknown type {value.get('type')!r}",
                               "Types: opening, orifice, closed.")
            entry = {"type": kind}
            if value.get("law"):
                entry["law"] = str(value["law"])
            project.connections[connection_key(a, b)] = project.connection(connection_key(a, b), b == "Outside")
            project.connections[connection_key(a, b)].update(entry)
    return project


def fill_measured(cad, project, surfaces):
    """As the GUI does when a role is assigned: sheets without a thickness get the measured one, chambers
    without a fluid volume their body volume."""
    for i, part in enumerate(project.parts):
        if part.role == CHAMBER and not float(part.props.get("fluid_volume", 0.0) or 0.0):
            part.props["fluid_volume"] = round(cad.bodies[i].volume, 6)
        if part.role in SHEETS and not float(part.props.get("thickness", 0.0) or 0.0):
            try:
                part.props["thickness"] = round(measure_thickness(cad, surfaces, i), 6)
            except Exception:
                pass


def summarise_geometry(design, project, report):
    from datetime import datetime
    roles = {p.name: p for p in project.parts}
    touching = {}
    for a, b in report["touching"]:
        touching.setdefault(a, []).append(b)
        touching.setdefault(b, []).append(a)
    bodies = []
    for b in report["bodies"]:
        part = roles[b["name"]]
        entry = {"name": b["name"], "role": part.role, "volume_mm3": round(b["volume_mm3"], 4),
                 "size_mm": [round(s, 4) for s in b["size"]]}
        if part.role in SHEETS:
            entry["thickness_mm"] = part.props.get("thickness")
        if part.props.get("model"):
            entry["model"] = part.props["model"]
        if "pressure" in part.props and part.props.get("model") not in (VENT, DYNAMIC_FLUID):
            entry["pressure_kPa"] = part.props["pressure"]
        bodies.append(entry)
    warnings = []
    for o in report["overlaps"]:
        warnings.append(f"OVERLAP: {o['bodies'][0]} and {o['bodies'][1]} share {o['shared_volume_mm3']:.4g} mm3 "
                        f"({o['share_of_smaller']:.1%} of the smaller). Bodies must touch, not overlap.")
    sheets = [p.name for p in project.parts if p.role in SHEETS]
    fluids = [p.name for p in project.parts if p.role in (CHAMBER, FLUID)]
    for name in fluids:
        if roles[name].role == CHAMBER and not any(s in touching.get(name, []) for s in sheets):
            warnings.append(f"{name} touches no membrane or shell: its pressure acts on nothing.")
    for name in [s for s in sheets if roles[s].role != ACTIVATION_MEMBRANE]:  # that one only marks the place
        size = np.sort(next(b["size"] for b in report["bodies"] if b["name"] == name))
        if size[0] > 0.2 * size[1]:
            warnings.append(f"{name} is not thin ({size[0]:.3g} x {size[1]:.3g} x {size[2]:.3g} mm): a membrane/shell "
                            "should be a thin plate (its thickness well below its width).")
    unassigned = [p.name for p in project.parts if p.role == UNASSIGNED]
    if unassigned:
        warnings.append(f"No role (ignored): {', '.join(unassigned)}")
    return {"design": design.id, "built_at": datetime.now().isoformat(timespec="seconds"), "bodies": bodies,
            "touching": {k: sorted(v) for k, v in touching.items() if k in fluids or k in sheets},
            "overlaps": report["overlaps"], "warnings": warnings}


def write_preview(design, project, surfaces):
    """The bodies' surface meshes with their colours, for the GUI's Study tab (no gmsh needed there)."""
    arrays = {"names": np.array([p.name for p in project.parts]),
              "roles": np.array([p.role for p in project.parts]),
              "colors": np.array([part_color(p) for p in project.parts]),
              "opacity": np.array([part_opacity(p) for p in project.parts], float)}
    for i, m in surfaces.items():
        arrays[f"v{i}"] = np.asarray(m.vertices, np.float32)
        arrays[f"f{i}"] = np.asarray(m.faces, np.int32)
    tmp = design.folder / f"preview.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, design.folder / "preview.npz")


# -----------------------------
# A built design, ready to simulate
# -----------------------------

def load_model(design):
    """(cad, project) of a built neuron or activation design."""
    from app.cad import CadModel
    if not design.project_path.exists() or not design.step_path.exists():
        raise ApiError(f"{design.id} has not been built.", f"Run mns cad build {design.id} first.")
    stale_check(design)
    cls = ActivationProject if design.space == "activation" else Project
    project = cls.load(design.project_path)
    cad = CadModel()
    cad.load_step(str(design.step_path))
    project.match_bodies(cad.bodies)
    return cad, project


def stale_check(design):
    built = design.state().get("built")
    if built and design.spec_path.stat().st_mtime > _timestamp(built) + 1:
        raise ApiError(f"{design.id}: design.yaml changed after the last build.",
                       f"Run mns cad build {design.id} again (or derive a new design instead of editing a simulated one).")


def _timestamp(iso):
    from datetime import datetime
    return datetime.fromisoformat(iso).timestamp()


def apply_sets(project, sets):
    """--set Part.field=value on the loaded project (chamber pressures and other properties, for one run)."""
    by_name = {p.name: p for p in project.parts}
    for key, value in sets:
        if "." not in key:
            raise ApiError(f"--set {key}: use Part.field=value (e.g. Weight1.pressure=10)")
        name, field = key.split(".", 1)
        if name in ("solver", "study"):
            settings = getattr(project, name, None)
            if settings is None or not hasattr(settings, field):
                raise ApiError(f"--set {key}: unknown setting")
            setattr(settings, field, type(getattr(settings, field))(value))
            continue
        part = by_name.get(name)
        if part is None:
            raise ApiError(f"--set: no part {name!r}", f"Parts: {', '.join(by_name)}")
        space = "activation" if project.space == ACTIVATION_SPACE else "neuron"
        part.props.update(resolve_props(name, part.role, {field: value}, space, {}))


# -----------------------------
# Full neuron designs
# -----------------------------

def build_full(design):
    """model.mfn from the design file: neuron: N###, activation: A###, link: {part, driving}, set: {Part.field: v}."""
    from app.full_neuron import LOCKED_FIELDS, FullNeuronProject
    spec = design.spec(reload=True)
    study = design.study
    if not spec.get("neuron") or not spec.get("activation"):
        raise ApiError(f"{design.id}: a full neuron needs neuron: N### and activation: A###")
    neuron, activation = study.design(spec["neuron"]), study.design(spec["activation"])
    if neuron.space != "neuron" or activation.space != "activation":
        raise ApiError("neuron must be an N### design and activation an A### design")
    for d in (neuron, activation):
        if not d.project_path.exists():
            raise ApiError(f"{d.id} is not built.", f"Run mns cad build {d.id}.")
    project = FullNeuronProject()
    if design.project_path.exists():  # keep the stored characterisation and display settings
        try:
            old = FullNeuronProject.load(design.project_path)
            project.characterisation, project.transforms = old.characterisation, old.transforms
        except Exception:
            pass
    project.import_neuron(str(neuron.project_path))
    try:
        project.import_design(str(activation.project_path))
    except ValueError as exc:
        raise ApiError(str(exc), f"Run mns study {activation.id} first (the activation design needs its results).")
    link = dict(spec.get("link") or {})
    part = link.get("part") or project.link.get("part")
    if part is None:
        raise ApiError(f"{design.id}: say which neuron membrane the activation design replaces",
                       f"link: {{part: <one of {', '.join(project.link_candidates())}>}}")
    if part not in project.link_candidates():
        raise ApiError(f"link part {part!r} is not a membrane/shell of {neuron.id}",
                       f"Candidates: {', '.join(project.link_candidates())}")
    project.link = {"part": part, "driving": link.get("driving") or AUTOMATIC}
    names = evaluate_parameters(spec.get("parameters") or {})
    for key, value in (spec.get("set") or {}).items():
        name, _, field = str(key).partition(".")
        p = project.part(name)
        if p is None or p.role != CHAMBER:
            raise ApiError(f"set {key}: {name!r} is not a chamber of {neuron.id}",
                           f"Chambers: {', '.join(c.name for c in project.chambers())}")
        if field in LOCKED_FIELDS:
            raise ApiError(f"set {key}: the chamber model can not be changed in a full neuron",
                           f"Change it in {neuron.id} (derive a new neuron design).")
        p.props.update(resolve_props(name, CHAMBER, {field: value}, "neuron", names))
    if spec.get("record"):
        keys = [k for k, _, _ in project.catalogue()]
        bad = [k for k in spec["record"] if k not in keys]
        if bad:
            raise ApiError(f"record: unknown key(s) {', '.join(bad)}", f"Keys: {', '.join(keys)}")
        project.record = list(spec["record"])
    project.save(design.project_path)
    summary = {"design": design.id, "neuron": neuron.id, "activation": activation.id, "link": project.link,
               "inputs": project.inputs(), "chambers": [c.name for c in project.chambers()],
               "parameters": [f"{p}.{f}" for p, f, _, _ in project.parameters()],
               "recordable": [k for k, _, _ in project.catalogue()], "recorded": project.recorded(),
               "built_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
               "files": {"project": str(design.project_path)}}
    design.set_state(status="built", built=summary["built_at"], geometry=summary)
    study.events.emit("design", f"{design.id} built: {neuron.id} + {activation.id}", design=design.id)
    return summary


def load_full(design):
    from app.full_neuron import FullNeuronProject
    if not design.project_path.exists():
        raise ApiError(f"{design.id} has not been built.", f"Run mns cad build {design.id} first.")
    stale_check(design)
    return FullNeuronProject.load(design.project_path)


# -----------------------------
# Importing existing files
# -----------------------------

def import_file(study, path, name=None, why=""):
    """A design (or several, for a .mfn) from an existing .mns / .mad / .mfn / .step file."""
    path = Path(path).resolve()
    if not path.exists():
        raise ApiError(f"File not found: {path}")
    copied = study.root / "inputs" / path.name
    if path.parent != copied.parent and not copied.exists():
        shutil.copy2(path, copied)
    suffix = path.suffix.lower()
    if suffix == ".mns":
        return [import_project(study, Project.load(path), "neuron", name or path.stem, why, source=path)]
    if suffix == ".mad":
        return [import_project(study, ActivationProject.load(path), "activation", name or path.stem, why, source=path)]
    if suffix in (".step", ".stp"):
        from app.cad import CadModel
        cad = CadModel()
        bodies = cad.load_step(str(path))
        project = Project(str(path), [b.name for b in bodies])
        project.auto_assign_from_names()
        return [import_project(study, project, "neuron", name or path.stem, why, source=path)]
    if suffix == ".mfn":
        return import_full(study, path, name or path.stem, why)
    raise ApiError(f"Can not import {path.name}", "Import .mns, .mad, .mfn or .step files.")


def _already_imported(study, source):
    for d in study.designs():
        if d.state().get("source") == str(source):
            return d
    return None


def import_project(study, project, space, name, why="", source=None):
    if source is not None:
        existing = _already_imported(study, source)
        if existing is not None:
            return existing.describe() | {"note": "already imported"}
    step = Path(project.step_path)
    if not step.exists():
        raise ApiError(f"The project's STEP file is missing: {step}")
    bodies = []
    for part in project.parts:
        props = copy.deepcopy(part.props)
        if part.role == ACTIVATION_MEMBRANE and props.get("design"):
            mad = Path(props["design"])
            if not mad.is_absolute() and project.path:
                mad = Path(project.path).parent / mad
            if mad.exists():
                linked = import_project(study, ActivationProject.load(mad), "activation", mad.stem,
                                        f"imported with {name}", source=mad.resolve())
                props["design"] = linked["id"]
            else:
                props.pop("design")
        body = {"name": part.name, "role": part.role}
        if props:
            body["props"] = props
        bodies.append(body)
    spec = {"why": why or f"imported from {source.name if source else step.name}",
            "parameters": {}, "base_step": "base.step", "bodies": bodies, "solver": asdict(project.solver)}
    if space == "activation":
        spec["study"] = asdict(project.study)
        spec["connections"] = {k.replace("↔", "<->"): {kk: vv for kk, vv in v.items()} for k, v in
                               project.connections.items()}
    design = study.new_design(space, name, spec, why=spec["why"])
    shutil.copy2(step, design.folder / "base.step")
    summary = build_design(design)
    if space == "activation" and project.results:
        built = ActivationProject.load(design.project_path)
        built.results = project.results
        built.save(design.project_path)
        design.set_state(status="studied")
        design.record_run("study", {"imported": True, "points": len(project.results.get("dp", []))},
                          activation_metrics(project.results, built.parts))
    design.set_state(source=str(source) if source else None)
    return {"id": design.id, "name": design.name, "space": space, "warnings": summary.get("warnings", [])}


def import_full(study, path, name, why=""):
    from app.full_neuron import FullNeuronProject
    full = FullNeuronProject.load(path)
    out = []
    neuron = import_project(study, full.neuron, "neuron", f"{name}-neuron", why or f"neuron of {path.name}",
                            source=path.with_suffix(".mfn#neuron"))
    out.append(neuron)
    activation = None
    if full.design_path and Path(full.design_path).exists():
        activation = import_project(study, ActivationProject.load(full.design_path), "activation",
                                    Path(full.design_path).stem, why or f"activation design of {path.name}",
                                    source=Path(full.design_path).resolve())
        out.append(activation)
    if activation is None:
        return out
    spec = {"why": why or f"imported from {path.name}", "neuron": neuron["id"], "activation": activation["id"],
            "link": dict(full.link)}
    if full.record:
        spec["record"] = list(full.record)
    if full.sweep.get("axes"):
        spec["characterise"] = {"axes": [f"{a['part']}.{a['field']}={a['from']}:{a['to']}:{a['points']}"
                                         for a in full.sweep["axes"]]}
    design = study.new_design("full", name, spec, why=spec["why"])
    build_full(design)
    if full.characterisation:
        project = FullNeuronProject.load(design.project_path)
        project.characterisation = full.characterisation
        project.save(design.project_path)
        design.set_state(status="characterised", note="characterisation imported (made with the original files)")
    design.set_state(source=str(path))
    out.append({"id": design.id, "name": design.name, "space": "full"})
    return out


def activation_metrics(results, parts):
    """Key numbers of an activation study (for comparisons and reports)."""
    from app.activation import output_definitions, output_pressures
    ok = np.asarray(results["converged"], bool)
    dp = np.asarray(results["dp"], float)
    A = np.asarray(results["area"], float)
    A0 = float(results["A0"])
    m = {"A0_mm2": A0, "points": int(len(dp)), "converged": int(ok.sum())}
    if ok.any():
        m["A_at_max_dp_mm2"] = float(A[ok][-1])
        closed = dp[ok][A[ok] <= 0.05 * A0]
        m["closing_dp_kPa"] = float(closed.min()) if len(closed) else None
        half = dp[ok][A[ok] <= 0.5 * A0]
        m["half_area_dp_kPa"] = float(half.min()) if len(half) else None
        mdot = np.asarray(results["mdot"], float)[ok]
        m["mdot_max_kg_s"] = float(np.abs(mdot).max())
        for name, values in output_pressures(results, output_definitions(parts, results)).items():
            v = np.asarray(values, float)[ok]
            m[f"out:{name}_range_kPa"] = [float(v.min()), float(v.max())]
        if "volume" in results:
            m["swept_volume_at_max_dp_mm3"] = float(np.asarray(results["volume"], float)[ok][-1])
    return m


def property_reference(space=None, role=None):
    """Every role's properties (key, kind, default, unit, choices, when it applies), from the simulator's schema."""
    from app.activation import StudySettings
    from dataclasses import fields
    out = {}
    for sp in ([space] if space else ["neuron", "activation"]):
        roles = {}
        for r, fl in SPACE_ROLE_FIELDS[SPACE_KEY[sp]].items():
            if role and r.lower() != str(role).lower() and ROLE_ALIASES[sp].get(str(role).lower()) != r:
                continue
            entries = {}
            for f in fl:
                e = {"kind": f.kind, "default": f.default if f.kind != "text" else "(laminar segment law)"}
                if f.unit:
                    e["unit"] = f.unit
                if f.kind == "choice" and not isinstance(f.choices, str):
                    e["choices"] = list(f.choices)
                if f.visible_if:
                    e["applies_when"] = f"{f.visible_if[0].strip('_')} in {list(f.visible_if[1])}"
                entries[f.key] = e
            if entries or r not in (UNASSIGNED,):
                roles[r] = entries
        out[sp] = {"roles": roles, "role_aliases": ROLE_ALIASES[sp]}
    out["solver"] = {f.name: f.default for f in fields(SolverSettings)}
    out["study (activation)"] = {f.name: f.default for f in fields(StudySettings)}
    out["choice_aliases"] = sorted({a for a, _ in CHOICE_ALIASES})
    return out
