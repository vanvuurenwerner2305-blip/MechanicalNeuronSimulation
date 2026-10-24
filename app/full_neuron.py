"""
Full-neuron workbench (no GUI): an inputs-to-pre-activation project (*.mns) and a pre-activation-to-activation
design (*.mad) linked into one neuron and solved as in the neuron space, with the design's membrane replaced by
its pre-simulated model (the "Activation membrane" role on the linked neuron part).

Here only the fluids' parameters (pressures, volumes, stiffness, ...) and the material of the membranes and shells
(Young's modulus, thickness, pre-tension, Poisson's ratio) can be changed, not the kinds of anything (roles, chamber
models, material models), not the mesh and not the design (that would need simulating it again). Both components are embedded in
the workbench file (*.mfn) as the versions they were imported (or last updated) at, so the original .mns and .mad
are not changed and later changes to them do not leak in unnoticed.

References (as in a CAD assembly): when a component's source file is saved with changes, `reference_status` says
it is out of date; `update_neuron` / `update_design` take the new version. Updating the neuron is a three-way merge:
`neuron_base` is the source as it was imported, so a value changed in the source since is taken, and a value changed
here (a fluid parameter, a solver setting) is kept (`neuron_changes` lists both, and conflicts). What to record at every solve is chosen from
the chamber pressures and volumes and the design's pre-activation Δp, named outputs, tube area and mass flow.

Both models are shown side by side; each has a position and a rotation (about its own centre) that only
affect the display.

Characterisation: any number of parameters (chamber pressures, ghost volumes, liquid shares, stiffnesses, fluid
volumes; membrane and shell Young's moduli, thicknesses, pre-tensions) are swept together over a grid and any number
of quantities recorded at every point. One axis can drive the same parameter of several parts at once (its "also"
parts, e.g. the thickness of both activation membranes). The dataset is saved in
the .mfn (`characterisation`, see `make_dataset` and `Characterisation`), so that the neuron can later be used
(looked up and interpolated) without simulating it again.
"""
import copy
import hashlib
import os
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from .activation import (ActivationDesign, ActivationProject, Coupling, body_display_coords, build_activation,
                         decode_array, display_bodies, encode_array, gas_of, generate_activation_mesh,
                         output_definitions, output_pressures, solve_point, ShutHold)
from .builder import KPA, build_environment, chamber_model
from .project import (ACTIVATION_MEMBRANE, ACTIVATION_ROLE_FIELDS, AUTOMATIC, CHAMBER, CHANNEL, CONSTANT,
                      CONSTANT_FLUID, CONTACT_ON, DEFORMABLE, FLUID, IDEAL_GAS, INCOMPRESSIBLE, ORIFICE, RIGID,
                      ROLE_FIELDS, SHEETS, SOLID, Field, Project, _relative)

SUFFIX = ".mfn"
FILE_TYPE = "full_neuron"
NEURON, ACTIVATION = "neuron", "activation"   # the two models (transform keys)

# fields of a chamber that may be changed here: everything but its model
LOCKED_FIELDS = {"model"}
# chamber fields that can be swept (numbers)
SWEEPABLE = [f for f in ROLE_FIELDS[CHAMBER] if f.kind == "float" and f.key not in LOCKED_FIELDS]
# membrane / shell fields that may be changed and swept here (the material; not the mesh or the supports)
SHEET_FIELDS = ("youngs_modulus", "thickness", "pretension", "poisson_ratio")

# Valve only (a .mad without a neuron): the valve is simulated again at every point, so its own parameters can be
# changed and swept. The pressure difference across its membrane is the parameter "Pre-activation.dp".
PREACTIVATION = "Pre-activation"
SOLID_FIELDS = ("youngs_modulus", "poisson_ratio")   # tube and solid parts
SOLVER_KEY = "__solver__"                            # design_edits entry of the solver settings
VALVE_RECORDS = [("volume", "membrane swept volume", "mm3"), ("area", "tube area", "mm2"), ("mdot", "mass flow", "kg/s"),
                 ("p_end", "pressure at the tube's downstream end", "kPa"), ("travel", "pusher travel", "mm")]


# a membrane's size (its CAD body stretched about its centre): changing it remeshes and rebuilds the valve
SIZE_FIELDS = tuple(f"size_{a}" for a in "xyz")
SIZE_FIELD_LIST = [Field(f"size_{a}", f"Size along {a}", "float", 0.0, "mm", 0.0, 1e6, 4,
                         tooltip=f"The membrane's extent along {a}: its CAD body is stretched about its centre to this "
                                 "size and the valve is meshed again (0: as in the CAD). The parts around it (e.g. the "
                                 "pusher bonded to it) keep their CAD size.")
                   for a in "xyz"]
# a rigid body's set-up displacement (Move x/y/z): changing it rebuilds the valve (same mesh)
MOVE_FIELDS = tuple(f"move_{a}" for a in "xyz")


def valve_fields(part, extents=None):
    """The fields of a design part that can be changed (and swept) when the valve is simulated on its own.
    extents: the part's CAD bounding-box size, if known: a membrane's size is then offered in its plane only."""
    fields = ACTIVATION_ROLE_FIELDS.get(part.role, [])
    if part.role in DEFORMABLE:
        sizes = SIZE_FIELD_LIST
        if extents is not None:   # not across the sheet (its thickness is a material setting)
            thin = int(np.argmin(extents))
            sizes = [f for k, f in enumerate(SIZE_FIELD_LIST) if k != thin]
        return [f for f in fields if f.key in SHEET_FIELDS] + sizes
    if part.role in (CHANNEL, SOLID):
        return [f for f in fields if f.key in SOLID_FIELDS]
    if part.role == RIGID:
        return [f for f in fields if f.key in MOVE_FIELDS]
    if part.role == FLUID and part.props.get("model") == CONSTANT_FLUID:
        return [f for f in fields if f.key == "pressure"]
    return []


def editable_fields(part):
    """The fields of a neuron part that can be changed in the full neuron (shown or not)."""
    if part.role == CHAMBER:
        return [f for f in ROLE_FIELDS[CHAMBER] if f.key not in LOCKED_FIELDS]
    if part.role in DEFORMABLE:
        return [f for f in ROLE_FIELDS[part.role] if f.key in SHEET_FIELDS]
    return []


def field_visible(part, f):
    """Whether a field applies to the part as set (e.g. Poisson's ratio only for St. Venant-Kirchhoff)."""
    if f.visible_if is None:
        return True
    key, allowed = f.visible_if
    return (part.role if key == "__role__" else part.props.get(key)) in allowed


def axis_parts(axis):
    """The parts one sweep axis {"part", "also", ...} drives: its part and its "also" parts."""
    return [axis.get("part")] + [p for p in axis.get("also") or () if p != axis.get("part")]


# An "also" entry "Part@field" drives another field of a part (e.g. a membrane's size along y with its size along x)
LINK = "@"


def target(name, field):
    """(part, field) an axis entry drives: `name` with the axis' field, or the part and field of a "Part@field"."""
    return tuple(name.split(LINK, 1)) if LINK in name else (name, field)


def parts_of(parts):
    """The parts of an (parts, field, values) axis triple: one name, or a tuple of names."""
    return [parts] if isinstance(parts, str) else list(parts)


def axis_name(parts, field):
    """'Part.field', or 'A+B.field' for an axis that drives several parts."""
    return f"{'+'.join(parts_of(parts))}.{field}"


def default_axis(part=None, field="pressure"):
    return {"part": part, "field": field, "from": 0.0, "to": 20.0, "points": 5, "also": []}


def axis_values(axis):
    """The values of a sweep axis {"from", "to", "points"} (increasing order kept as given)."""
    return np.linspace(float(axis["from"]), float(axis["to"]), max(int(axis["points"]), 1)).tolist()


# -----------------------------
# Display transforms
# -----------------------------

def rotation(degrees) -> np.ndarray:
    """Rotation matrix of x, y, z angles in degrees (applied about x, then y, then z)."""
    return Rotation.from_euler("xyz", np.asarray(degrees, float), degrees=True).as_matrix()


def transform_points(points, center, position, degrees):
    """Points rotated about `center` and then moved by `position` (mm)."""
    points = np.asarray(points, float)
    return (points - center) @ rotation(degrees).T + np.asarray(center, float) + np.asarray(position, float)


def identity():
    return {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}


# -----------------------------
# Project
# -----------------------------

CURRENT, OUTDATED, MISSING = "current", "out of date", "not found"   # reference_status


def neuron_state(project) -> dict:
    """What an update compares: the project's dict with absolute paths and without display settings."""
    data = copy.deepcopy(project.to_dict())
    data["step_path"] = str(Path(data["step_path"]).resolve()) if data.get("step_path") else None
    parts = {}
    for part in data["parts"]:
        part.pop("visible", None)
        if part["props"].get("design"):
            part["props"]["design"] = str(Path(part["props"]["design"]).resolve())
        parts[part["name"]] = part
    return {"step_path": data["step_path"], "parts": parts, "solver": data.get("solver", {})}


class FullNeuronProject:
    def __init__(self):
        self.neuron = None          # Project (embedded copy; its fluid parameters are edited here)
        self.neuron_source = None   # the .mns it was imported from
        self.neuron_base = None     # neuron_state of the source when it was imported / last updated
        self.design_path = None     # the .mad it was imported from
        self.design_data = None     # the design as imported / last updated (ActivationProject dict, embedded)
        self.design_sha1 = None     # sha1 of that .mad file's bytes
        self.link = {"part": None, "driving": AUTOMATIC}   # neuron part replaced by the design's membrane
        self.transforms = {NEURON: identity(), ACTIVATION: identity()}
        self.record = None          # keys of recorded quantities (None: the defaults)
        self.sweep = {"axes": [], "store_shapes": True}   # axes: [{"part", "field", "from", "to", "points"}]
        self.characterisation = None   # the stored dataset (make_dataset) or None
        self.path = None
        # valve only: values changed here on top of the embedded design {part / connection key / PREACTIVATION /
        # SOLVER_KEY: {field: value}} (kept when the design is updated)
        self.design_edits = {}
        self.design_extents = {}    # part name -> CAD bounding-box size (mm), set by whoever loaded the CAD (not saved)

    @property
    def valve_only(self) -> bool:
        """A design without a neuron: the valve on its own, simulated again at every point."""
        return self.neuron is None and self.design_data is not None

    @property
    def ready(self) -> bool:
        return self.neuron is not None or self.valve_only

    # -----------------------------
    # Valve only
    # -----------------------------

    def orifices(self):
        """Keys of the design's orifice connections (their area can be set and swept)."""
        if self.design_data is None:
            return []
        return [k for k, s in (self.design_data.get("connections") or {}).items() if s.get("type") == ORIFICE]

    def valve_value(self, name, field):
        """The value of a valve parameter as set here (edit) or in the design."""
        edit = self.design_edits.get(name, {})
        if field in edit:
            return edit[field]
        if name == PREACTIVATION:
            return 0.0
        if field in SIZE_FIELDS:
            size = self.design_extents.get(name)
            return float(size["xyz".index(field[-1])]) if size is not None else 0.0
        if name in self.orifices():
            return float(self.design_data["connections"][name].get("area") or 0.0)
        part = next((p for p in self.design_data["parts"] if p["name"] == name), None)
        if part and field in MOVE_FIELDS:
            return float(part["props"].get(field) or 0.0)
        return part["props"].get(field) if part else None

    def set_valve_value(self, name, field, value):
        self.design_edits.setdefault(name, {})[field] = value

    def valve_project(self) -> ActivationProject:
        """The embedded design with the values changed here (what the valve-only run builds and solves)."""
        project = self.design_project()
        parts = {p.name: p for p in project.parts}
        for name, edit in self.design_edits.items():
            if name == SOLVER_KEY:
                for key, value in edit.items():
                    setattr(project.solver, key, value)
            elif name in parts:
                parts[name].props.update(edit)
            elif name in project.connections:
                project.connections[name].update({k: v for k, v in edit.items() if k == "area"})
        return project

    def _valve_parameters(self):
        out = [(PREACTIVATION, "dp", "Pre-activation Δp (across the membrane)", "kPa")]
        project = self.valve_project()
        for p in project.parts:
            for f in valve_fields(p, self.design_extents.get(p.name)):
                if field_visible(p, f):
                    out.append((p.name, f.key, f"{p.name} {f.label[0].lower() + f.label[1:]}", f.unit))
        out += [(key, "area", f"{key} orifice area (0 = the contact face)", "mm2") for key in self.orifices()]
        return out

    # -----------------------------

    def import_neuron(self, path):
        project = Project.load(path)
        self.neuron, self.neuron_source = project, str(Path(path).resolve())
        self.neuron_base = neuron_state(project)
        linked = [p.name for p in project.parts if p.role == ACTIVATION_MEMBRANE]
        if linked:  # the project already names the membrane the design replaces
            part = next(p for p in project.parts if p.name == linked[0])
            self.link = {"part": part.name, "driving": part.props.get("driving", AUTOMATIC) or AUTOMATIC}
        elif self.link["part"] not in self.link_candidates():
            self.link = {"part": None, "driving": AUTOMATIC}
        self.record = None
        return project

    @staticmethod
    def _read_design(path):
        """(ActivationProject, sha1 of the file) of a usable design file."""
        raw = Path(path).read_bytes()
        project = ActivationProject.from_dict(json.loads(raw.decode("utf-8")), Path(path).parent)
        if not project.results:
            raise ValueError(f"{Path(path).name} has no simulated results: simulate it in the Pre-activation → "
                             "activation space and save it first.")
        design = ActivationDesign(project.results, Path(path).stem)
        if not design.has_volume:
            raise ValueError(f"{Path(path).name} was simulated before the membrane's swept volume was recorded: "
                             "open it in the Pre-activation → activation space, simulate it again and save it.")
        return project, hashlib.sha1(raw).hexdigest()

    def import_design(self, path):
        project, sha1 = self._read_design(path)
        self.design_path = str(Path(path).resolve())
        self.design_data, self.design_sha1 = project.to_dict(), sha1
        self._design_key = None
        self.record = None
        return project

    def design_project(self) -> ActivationProject:
        """The embedded design (the version imported / last updated, not the file as it is now)."""
        if self.design_data is None:
            raise ValueError("Import a pre-activation-to-activation design (*.mad) first.")
        # a copy: from_dict shares the connections and results with the dict, and a valve-only run changes them
        return ActivationProject.from_dict(copy.deepcopy(self.design_data), Path(self.design_path).parent)

    def design(self) -> ActivationDesign:
        """The linked design (embedded version; cached: decoding its stored FEM solution takes a moment)."""
        if getattr(self, "_design_key", None) != self.design_sha1:
            self._design = ActivationDesign.of(self.design_project(), Path(self.design_path).stem)
            self._design_key = self.design_sha1
        return self._design

    # -----------------------------
    # References to the source files (out of date / update)
    # -----------------------------

    def _source_state(self, kind):
        """neuron_state of the .mns / sha1 of the .mad as the file is now (cached per file version);
        None if the file is gone or unreadable."""
        path = self.neuron_source if kind == NEURON else self.design_path
        try:
            st = os.stat(path)
        except (OSError, TypeError):
            return None
        key = (path, st.st_mtime_ns, st.st_size)
        cache = self.__dict__.setdefault("_source_cache", {})
        if cache.get(kind, (None,))[0] != key:
            try:
                state = (neuron_state(Project.load(path)) if kind == NEURON
                         else hashlib.sha1(Path(path).read_bytes()).hexdigest())
            except Exception:
                state = None
            cache[kind] = (key, state)
        return cache[kind][1]

    def reference_status(self, kind):
        """CURRENT, OUTDATED or MISSING for the imported neuron (NEURON) or design (ACTIVATION); None if there is
        none or it has no source file."""
        if kind == NEURON and (self.neuron is None or not self.neuron_source):
            return None
        if kind == ACTIVATION and self.design_data is None:
            return MISSING if self.design_path else None
        state = self._source_state(kind)
        if state is None:
            return MISSING
        if kind == NEURON:
            if self.neuron_base is None:   # a file saved before references were tracked
                self.neuron_base = self._legacy_base(state)
            return CURRENT if state == self.neuron_base else OUTDATED
        return CURRENT if state == self.design_sha1 else OUTDATED

    def _legacy_base(self, source):
        """For files without neuron_base: if the source differs from the embedded copy only in values that can be
        edited here (fluid and sheet parameters, solver), those were edited here; otherwise the copy is the old
        version."""
        embedded = neuron_state(self.neuron)
        return copy.deepcopy(source) if _editable_free(source) == _editable_free(embedded) else embedded

    def outdated(self):
        """The components whose source file was saved with changes, of [NEURON, ACTIVATION]."""
        return [k for k in (NEURON, ACTIVATION) if self.reference_status(k) == OUTDATED]

    def neuron_changes(self):
        """What updating the neuron would do: {"source": [...], "kept": [...], "conflicts": [...]} of lines; source =
        changed in the .mns since the import, kept = changed here only, conflicts = changed in both."""
        if self.neuron_base is None:
            self.reference_status(NEURON)
        new, base, here = self._source_state(NEURON), self.neuron_base, neuron_state(self.neuron)
        if new is None:
            raise ValueError(f"{self.neuron_source} cannot be read.")
        out = {"source": [], "kept": [], "conflicts": []}
        if new["step_path"] != base["step_path"]:
            out["source"].append(f"CAD file: {Path(base['step_path'] or '').name} → {Path(new['step_path']).name}")
        for name in base["parts"]:
            if name not in new["parts"]:
                out["source"].append(f"{name}: removed")
        for name, part in new["parts"].items():
            old = base["parts"].get(name)
            if old is None:
                out["source"].append(f"{name}: added ({part['role']})")
            elif old["role"] != part["role"]:
                out["source"].append(f"{name}: role {old['role']} → {part['role']}")
            else:
                mine = here["parts"].get(name)
                mine = mine["props"] if mine and mine["role"] == old["role"] else old["props"]
                for key in sorted(set(part["props"]) | set(old["props"])):
                    _compare(out, f"{name} {key}", old["props"].get(key), part["props"].get(key), mine.get(key))
        for key in sorted(set(new["solver"]) | set(base["solver"])):
            _compare(out, f"Solver {key}", base["solver"].get(key), new["solver"].get(key), here["solver"].get(key))
        return out

    def update_neuron(self, keep_conflicts=True):
        """Take the .mns as it is now, keeping the values changed here (fluid parameters, solver settings; also
        where the source changed them too if keep_conflicts). Returns notes for the log."""
        if self.neuron_base is None:
            self.reference_status(NEURON)
        source = Project.load(self.neuron_source)
        new, base, here = neuron_state(source), self.neuron_base, neuron_state(self.neuron)
        old_parts = {p.name: p for p in self.neuron.parts}
        for part in source.parts:
            b, h = base["parts"].get(part.name), here["parts"].get(part.name)
            if b is None or h is None or not (b["role"] == h["role"] == part.role):
                continue
            for key, value in h["props"].items():
                changed_here = value != b["props"].get(key)
                changed_there = new["parts"][part.name]["props"].get(key) != b["props"].get(key)
                if changed_here and (keep_conflicts or not changed_there):
                    part.props[key] = copy.deepcopy(old_parts[part.name].props[key])
            part.visible = old_parts[part.name].visible
        for key, value in here["solver"].items():
            changed_there = new["solver"].get(key) != base["solver"].get(key)
            if value != base["solver"].get(key) and (keep_conflicts or not changed_there):
                setattr(source.solver, key, value)
        self.neuron, self.neuron_base = source, new
        notes = []
        if self.link.get("part") and self.link["part"] not in self.link_candidates():
            notes.append(f"The linked membrane {self.link['part']} is no longer a membrane of the neuron: link again "
                         "(Link tab).")
            self.link = {"part": None, "driving": AUTOMATIC}
        elif self.link.get("driving") not in (None, AUTOMATIC) and \
                self.link["driving"] not in [p.name for p in self.chambers()]:
            notes.append(f"The driving chamber {self.link['driving']} is gone: set to Automatic.")
            self.link["driving"] = AUTOMATIC
        known = {(p, f) for p, f, _, _ in self.parameters()}
        for a in self.sweep["axes"]:
            for name in axis_parts(a):
                if (name, a.get("field")) not in known:
                    notes.append(f"Sweep parameter {name} {a.get('field')} no longer exists; it is left out.")
        return notes

    def update_design(self):
        """Take the .mad as it is now (it must have simulated results). Returns notes on what changed."""
        old = self.design() if self.design_data is not None else None
        project, sha1 = self._read_design(self.design_path)
        self.design_data, self.design_sha1 = project.to_dict(), sha1
        new = self.design()
        notes = []
        if old is not None:
            if tuple(old.dp_range) != tuple(new.dp_range):
                notes.append(f"Δp range {old.dp_range[0]:.4g} … {old.dp_range[1]:.4g} → "
                             f"{new.dp_range[0]:.4g} … {new.dp_range[1]:.4g} kPa")
            if old.output_names != new.output_names:
                notes.append(f"outputs {', '.join(old.output_names)} → {', '.join(new.output_names)}")
            if not math.isclose(old.membrane_area, new.membrane_area, rel_tol=1e-9):
                notes.append(f"membrane area {old.membrane_area:.4g} → {new.membrane_area:.4g} mm²")
        return notes

    def link_candidates(self):
        """Neuron parts the design's membrane can replace: membranes, shells and activation membranes."""
        return [p.name for p in self.neuron.parts if p.role in SHEETS] if self.neuron else []

    def part(self, name):
        return next((p for p in self.neuron.parts if p.name == name), None)

    def chambers(self):
        return [p for p in self.neuron.parts if p.role == CHAMBER] if self.neuron else []

    def inputs(self):
        """Constant-pressure chambers (the neuron's inputs)."""
        return [p.name for p in self.chambers() if p.props.get("model") == CONSTANT]

    def sheets(self):
        """The neuron's simulated membranes and shells (not the one the linked design replaces)."""
        linked = self.link.get("part") if self.design_path else None
        return [p for p in self.neuron.parts if p.role in DEFORMABLE and p.name != linked] if self.neuron else []

    def parameters(self):
        """[(part, field, label, unit)] of every parameter that can be swept: the chamber fields its model uses and
        the material of every simulated membrane and shell. Valve only: the membrane's Δp, the supply and sink
        pressures, the orifice areas and the materials of the membrane, tube and solid parts."""
        if self.valve_only:
            return self._valve_parameters()
        out = []
        for p in self.chambers():
            for f in SWEEPABLE:
                if field_visible(p, f):
                    out.append((p.name, f.key, f"{p.name} {f.label.lower().replace(' (gauge)', '')}", f.unit))
        for p in self.sheets():
            for f in editable_fields(p):
                if f.kind == "float" and field_visible(p, f):
                    out.append((p.name, f.key, f"{p.name} {f.label[0].lower() + f.label[1:]}", f.unit))
        return out

    def axis_label(self, parts, field):
        """(label, unit) of a sweep axis over one or several parts."""
        params = {(p, f): (label, unit) for p, f, label, unit in self.parameters()}
        names = parts_of(parts)
        label, unit = params.get((names[0], field), (f"{names[0]} {field}", ""))
        same = [n for n in names if LINK not in n]
        if len(same) > 1:
            label = "+".join(same) + label[len(names[0]):]
        linked = [params.get(target(n, field), (n,))[0] for n in names if LINK in n]
        if linked:
            label += " (= " + ", ".join(linked) + ")"
        return label, unit

    def also_label(self, name, field):
        """How an "also" entry reads in the UI: a part name, or the label of a "Part@field"."""
        if LINK not in name:
            return name
        params = {(p, f): label for p, f, label, _ in self.parameters()}
        return params.get(target(name, field), name)

    def also_candidates(self, axis):
        """The other parts that have the axis' parameter (one axis can drive them too); for a membrane's size also
        its size along the other in-plane direction ("Part@field": a square membrane, say)."""
        part, field = axis.get("part"), axis.get("field")
        out = [p for p, f, _, _ in self.parameters() if f == field and p != part]
        if field in SIZE_FIELDS:
            out += [f"{part}{LINK}{f}" for p, f, _, _ in self.parameters() if p == part and f in SIZE_FIELDS and f != field]
        return out

    def sweep_axes(self):
        """The valid sweep axes as [(parts, field, values)], parts = one name, or a tuple of names for an axis that
        drives the same field of several parts. Unknown parameters and repeats are left out."""
        known = {(p, f) for p, f, _, _ in self.parameters()}
        out, seen = [], set()
        for a in self.sweep["axes"]:
            field = a.get("field")
            if (a.get("part"), field) not in known or (a.get("part"), field) in seen:
                continue
            names = [n for n in axis_parts(a) if target(n, field) in known and target(n, field) not in seen]
            seen.update(target(n, field) for n in names)
            out.append((names[0] if len(names) == 1 else tuple(names), field, axis_values(a)))
        return out

    def fingerprint(self, swept=()) -> str:
        """What the characterisation depends on: the neuron's parameters (except the swept ones and display
        settings), the solver settings, the link and the design file's contents. swept: [(parts, field)]."""
        if self.valve_only:   # the design and what was changed here, but not the swept values
            swept = {target(n, f) for parts, f in swept for n in parts_of(parts)}
            edits = {name: {f: v for f, v in edit.items() if (name, f) not in swept}
                     for name, edit in self.design_edits.items()}
            text = json.dumps({"valve": self.design_sha1, "edits": edits}, sort_keys=True, default=str)
            return hashlib.sha1(text.encode("utf-8")).hexdigest()
        data = copy.deepcopy(self.neuron.to_dict()) if self.neuron else {}
        data.pop("step_path", None)
        swept = {(n, f) for parts, f in swept for n in parts_of(parts)}
        for part in data.get("parts", []):
            part.pop("visible", None)
            part["props"].pop("design", None)
            for p, f in swept:
                if part["name"] == p:
                    part["props"].pop(f, None)
        text = json.dumps({"neuron": data, "link": self.link, "design": self.design_sha1}, sort_keys=True,
                          default=str)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def dataset(self):
        """The stored characterisation as a Characterisation (None without one)."""
        return Characterisation(self.characterisation) if self.characterisation else None

    def dataset_current(self) -> bool:
        """Whether the stored characterisation still matches the model (same parameters, link and design)."""
        c = self.characterisation
        return bool(c) and c.get("fingerprint") == self.fingerprint([(axis_parts(a), a["field"]) for a in c["axes"]])

    def linked_project(self) -> Project:
        """The neuron with the linked part replaced by the design's membrane (what is built and solved)."""
        if self.neuron is None:
            raise ValueError("Import an inputs-to-pre-activation project (*.mns) first.")
        project = copy.deepcopy(self.neuron)
        if self.design_path and self.link.get("part"):
            part = next((p for p in project.parts if p.name == self.link["part"]), None)
            if part is None:
                raise ValueError(f"The linked part {self.link['part']} is not in the neuron.")
            thickness = part.props.get("thickness", 0.0)
            project.set_role(part, ACTIVATION_MEMBRANE)
            part.props.update(design=self.design_path, driving=self.link.get("driving") or AUTOMATIC,
                              thickness=thickness)
            project.pinned_designs = {part.name: self.design()}   # the embedded version, not the file
        elif self.design_path:
            raise ValueError("Link the design: choose the neuron membrane it replaces (Link tab).")
        return project

    # -----------------------------
    # Recording
    # -----------------------------

    def catalogue(self):
        """[(key, label, unit)] of everything that can be recorded."""
        if self.valve_only:
            try:
                names = self.design().output_names
            except Exception:
                names = []
            return [(f"out:{n}", f"output {n}", "kPa") for n in names] + list(VALVE_RECORDS)
        out = []
        for p in self.chambers():
            out.append((f"P:{p.name}", f"P {p.name}", "kPa"))
        for p in self.chambers():
            out.append((f"dV:{p.name}", f"ΔV {p.name}", "mm3"))
        if self.design_path and self.link.get("part"):
            out.append(("dp", "pre-activation Δp (across the design's membrane)", "kPa"))
            try:
                names = self.design().output_names
            except Exception:
                names = []
            out += [(f"out:{n}", f"output {n}", "kPa") for n in names]
            out += [("area", "tube area", "mm2"), ("mdot", "mass flow", "kg/s")]
        if self.neuron is not None and self.neuron.solver.contact != CONTACT_ON:
            out.append(("penetration", "deepest overlap of the contacts switched off", "mm"))
        return out

    def default_record(self):
        """The pre-activation pressure (closed chambers) and every output of the design."""
        keys = [k for k, _, _ in self.catalogue()]
        if self.valve_only:
            return [k for k in keys if k.startswith("out:") or k in ("area", "mdot", "volume")]
        closed ={f"P:{p.name}" for p in self.chambers() if p.props.get("model") in (IDEAL_GAS, INCOMPRESSIBLE)}
        chosen = [k for k in keys if k in closed or k in ("dp", "penetration") or k.startswith("out:")]
        return chosen or keys[:1]

    def recorded(self):
        keys = [k for k, _, _ in self.catalogue()]
        record = self.default_record() if self.record is None else self.record
        return [k for k in keys if k in record]

    # -----------------------------
    # Persistence
    # -----------------------------

    def to_dict(self, path):
        neuron = self.neuron.to_dict() if self.neuron else None
        if neuron:
            neuron["step_path"] = _relative(self.neuron.step_path, path)
            for part in neuron["parts"]:
                if part["props"].get("design"):
                    part["props"]["design"] = _relative(part["props"]["design"], path)
        base = copy.deepcopy(self.neuron_base)
        if base:
            base["step_path"] = _relative(base["step_path"], path) if base["step_path"] else None
            for part in base["parts"].values():
                if part["props"].get("design"):
                    part["props"]["design"] = _relative(part["props"]["design"], path)
        design = copy.deepcopy(self.design_data)
        if design:
            design["step_path"] = _relative(design["step_path"], path)
        return {"type": FILE_TYPE, "version": 2, "neuron": neuron,
                "neuron_source": _relative(self.neuron_source, path) if self.neuron_source else None,
                "neuron_base": base,
                "design": _relative(self.design_path, path) if self.design_path else None,
                "design_data": design, "design_sha1": self.design_sha1,
                "link": self.link, "transforms": self.transforms, "record": self.record, "sweep": self.sweep,
                "design_edits": self.design_edits, "characterisation": self.characterisation}

    def save(self, path):
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(path), indent=2), encoding="utf-8")
        self.path = str(path)

    @classmethod
    def load(cls, path):
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("type") != FILE_TYPE:
            raise ValueError("Not a full-neuron file.")
        project = cls()
        resolve = lambda p: str((path.parent / p).resolve()) if p and not Path(p).is_absolute() else p  # noqa: E731
        if data.get("neuron"):
            project.neuron = Project.from_dict(data["neuron"], path.parent)
        project.neuron_source = resolve(data.get("neuron_source"))
        base = data.get("neuron_base")
        if base:
            base["step_path"] = resolve(base["step_path"])
            for part in base["parts"].values():
                if part["props"].get("design"):
                    part["props"]["design"] = resolve(part["props"]["design"])
            project.neuron_base = base
        project.design_path = resolve(data.get("design"))
        if data.get("design_data"):
            design = data["design_data"]
            design["step_path"] = resolve(design["step_path"])
            project.design_data, project.design_sha1 = design, data.get("design_sha1")
        elif project.design_path and Path(project.design_path).exists():
            # version 1 only referenced the design: take the file as it is now
            design, project.design_sha1 = cls._read_design(project.design_path)
            project.design_data = design.to_dict()
        project.link = dict({"part": None, "driving": AUTOMATIC}, **(data.get("link") or {}))
        project.transforms = {k: dict(identity(), **(data.get("transforms") or {}).get(k, {}))
                              for k in (NEURON, ACTIVATION)}
        project.record = data.get("record")
        sweep = data.get("sweep") or {}
        if "axes" not in sweep:  # version 1 files: inputs A and B
            sweep = {"axes": [dict(default_axis(sweep[k], "pressure"), **{f: sweep.get(f"{k}_{f}", default_axis()[f])
                                                                          for f in ("from", "to", "points")})
                              for k in ("a", "b") if sweep.get(k)]}
        project.sweep.update(sweep)
        project.design_edits = data.get("design_edits") or {}
        project.characterisation = data.get("characterisation")
        project.path = str(path)
        return project


def _editable_free(state):
    """A neuron_state without the values that can be edited in the full neuron (chamber fields, sheet materials,
    solver)."""
    state = copy.deepcopy(state)
    for part in state["parts"].values():
        if part["role"] == CHAMBER:
            for key in [k for k in part["props"] if k not in LOCKED_FIELDS]:
                del part["props"][key]
        elif part["role"] in DEFORMABLE:
            for key in [k for k in part["props"] if k in SHEET_FIELDS]:
                del part["props"][key]
    state["solver"] = {}
    return state


def _fmt(value):
    return "(none)" if value is None else (f"{value:.6g}" if isinstance(value, float) else str(value))


def _compare(out, label, base, new, here):
    """Sort one value into neuron_changes' lists."""
    if base != new and here != base and here != new:
        out["conflicts"].append(f"{label}: {_fmt(here)} here; {_fmt(base)} → {_fmt(new)} in the source")
    elif base != new:
        out["source"].append(f"{label}: {_fmt(base)} → {_fmt(new)}")
    elif here != base:
        out["kept"].append(f"{label}: {_fmt(here)} here ({_fmt(base)} in the source)")


# -----------------------------
# Solving and recording
# -----------------------------

def record_values(build, parts) -> dict:
    """Every recordable quantity at the current solved state {key: value}."""
    values = {}
    for c, v in build.volumes.items():
        values[f"P:{parts[c].name}"] = v.P / KPA
        values[f"dV:{parts[c].name}"] = v.delta_volume
    for out in build.activation_outputs().values():
        values["dp"] = out["dp"]
        values.update({f"out:{n}": x for n, x in out["outputs"].items()})
        values.update(area=out["area"], mdot=out["mdot"], extrapolated=out["extrapolated"])
    if not math.isnan(penetration := build.penetration()):
        values["penetration"] = penetration
    values["warnings"] = build.range_warnings(parts)
    values["contact"] = build.penetration_warnings()
    return values


def serpentine(shape):
    """Every index of an N-dimensional grid, ordered so that consecutive points differ by one step of one axis
    (each solve warm-starts from a close neighbour)."""
    if not shape:
        return [()]
    inner = serpentine(shape[1:])
    out = []
    for i in range(shape[0]):
        out += [(i,) + t for t in (inner if i % 2 == 0 else inner[::-1])]
    return out


def apply_parameter(build, cad, part, index, field, value):
    """Set a parameter on the part and on the built model (no rebuild): a chamber's on its FluidVolume, a membrane's
    or shell's material on its Shell (the mesh is kept: the mid-surface does not move when the thickness changes)."""
    part.props[field] = float(value)
    if index in build.shells:
        if field not in SHEET_FIELDS:
            raise ValueError(f"{part.name}: {field} cannot be changed without rebuilding the model.")
        build.shells[index].set_material(**{field: float(value)})
        if field == "thickness":
            build.thickness[index] = float(value)
        return
    volume = build.volumes[index]
    kw = chamber_model(part.props, cad.bodies[index].volume)
    volume.P0 = kw["P0"]
    if "gas_volume" in kw:
        volume.gas_volume, volume.initial_volume = kw["gas_volume"], kw["initial_volume"]
    if "bulk_stiffness" in kw:
        volume.bulk_stiffness, volume.liquid_volume = kw["bulk_stiffness"], kw["liquid_volume"]
        volume.excess_volume = kw["initial_volume"] - kw["liquid_volume"]


def get_state(build):
    """Every body's unknowns (shell nodes and rotations, rigid body poses, empirical membranes)."""
    return [b.get_state().detach().clone() for b in build.env.membrane_list]


def set_state(build, state):
    for b, u in zip(build.env.membrane_list, state):
        b.set_state(u.clone())


def predicted_state(states, previous, idx, axes):
    """Starting guess at grid point idx, next to the solved point `previous` along one axis: the straight line
    through the solutions at `previous` and the point before it on that axis, when that one was solved too.
    None when there is no such line (the neighbour's solution is used as it is)."""
    moved = [k for k, (a, b) in enumerate(zip(idx, previous)) if a != b]
    if len(moved) != 1:
        return None
    k = moved[0]
    back = list(previous)
    back[k] -= idx[k] - previous[k]
    back = tuple(back)
    if back not in states or not 0 <= back[k] < len(axes[k][2]):
        return None
    values = axes[k][2]
    step = values[previous[k]] - values[back[k]]
    if step == 0:
        return None
    t = (values[idx[k]] - values[previous[k]]) / step
    return [u1 + t * (u1 - u0) for u0, u1 in zip(states[back], states[previous])]


def run_grid(worker, cad, project, mesh, axes=()):
    """Solve the linked neuron at every point of the grid spanned by `axes` [(part name, field, values)]
    (serpentine order, warm-started), or once at the set values without axes. Emits one row per point through
    worker.item: {"index" (grid index), "params" (the axes' values), "converged", "time", "values", "message",
    "coords" (per neuron sheet, for the 3D view)}. Returns the build (its shells give the rest geometry)."""
    build = build_environment(cad, mesh, project)
    for w in build.warnings:
        worker.log("Warning: " + w)
    parts = project.parts
    index = {p.name: i for i, p in enumerate(parts)}
    axes = [(names, field, [float(x) for x in values]) for names, field, values in axes]
    for names, field, values in axes:
        for name in parts_of(names):
            i = index.get(name)
            if i not in build.volumes and i not in build.shells:
                raise ValueError(f"{name} is not a chamber, membrane or shell of the neuron.")
            if i in build.shells and field not in SHEET_FIELDS:
                raise ValueError(f"{name}: {field} cannot be swept.")
            if i in build.shells and field in ("youngs_modulus", "thickness") and min(values) <= 0:
                raise ValueError(f"{name}: the {field.replace('_', ' ')} must be positive at every point.")
    pressures = [abs(x) * KPA for _, field, values in axes if field == "pressure" for x in values]
    thinnest = [x for names, field, values in axes if field == "thickness"
                and any(index[n] in build.shells for n in parts_of(names)) for x in values]
    build.set_contact_stiffness(max([abs(v.P0) for v in build.volumes.values()] + pressures + [KPA]),
                                min(thinnest + list(build.thickness.values())) if thinnest else None)
    points = serpentine([len(values) for _, _, values in axes])
    states = {}   # grid index -> solved state (for the next points' starting guesses)
    previous = None
    for k, idx in enumerate(points):
        worker.check()
        params = [values[i] for i, (_, _, values) in zip(idx, axes)]
        for (names, field, _), value in zip(axes, params):
            for name in parts_of(names):
                apply_parameter(build, cad, parts[index[name]], index[name], field, value)
        callback = lambda lam, it, r: worker.check()  # noqa: E731
        start = time.time()
        result, predicted = None, False
        if previous is not None:
            # straight to the new value from the neighbour's solution (no load ramp), first from the line through
            # the last two points along this axis, then from the neighbour itself, and only then from rest
            guess = predicted_state(states, previous, idx, axes)
            if guess is not None:
                set_state(build, guess)
                result = build.solve(project.solver, callback, warm_start=True, load_steps=1, fixed_contact=True)
                predicted = result.converged
                if not result.converged:
                    set_state(build, states[previous])
            if result is None or not result.converged:
                result = build.solve(project.solver, callback, warm_start=True, load_steps=1, fixed_contact=True)
        if result is None or not result.converged:
            result = build.solve(project.solver, callback, fixed_contact=True)
        if result.converged:
            states[idx] = get_state(build)
            previous = idx
        row_values = record_values(build, parts)
        for w in row_values["warnings"] + row_values["contact"]:
            worker.log("Warning: " + w)
        worker.item.emit({"index": tuple(idx), "params": params, "converged": result.converged,
                          "time": time.time() - start, "values": row_values, "message": result.message,
                          "iterations": sum(result.iterations), "predicted": predicted,
                          "coords": [s.x.detach().cpu().numpy().copy() for s in build.shells.values()]})
        worker.report((k + 1) / len(points), f"Point {k + 1}/{len(points)}")
    return build


def run_points(worker, cad, project, mesh, a_name=None, a_values=(None,), b_name=None, b_values=(None,)):
    """run_grid over one or two chamber pressures; rows also carry "i", "j", "a", "b"."""
    axes = [(n, "pressure", v) for n, v in ((a_name, a_values), (b_name, b_values)) if n is not None]
    item = worker.item

    def emit(row):
        idx, params = list(row["index"]) + [None, None], list(row["params"]) + [None, None]
        row.update(i=idx[0], j=idx[1], a=params[0], b=params[1])
        item.emit(row)
    relay = SimpleNamespace(check=worker.check, report=worker.report, log=worker.log, item=SimpleNamespace(emit=emit))
    return run_grid(relay, cad, project, mesh, axes)


def apply_valve_parameter(build, project, name, field, value):
    """Set a valve parameter on the project and on the built valve (no rebuild; build None: the project only): a
    supply / sink pressure, an orifice area, the material of a membrane, shell, tube or solid part. A membrane's size
    (SIZE_FIELDS) and a rigid body's displacement (MOVE_FIELDS) are set on the project only: they take effect when
    the valve is built again (run_valve_grid; a size also remeshes it). The membrane's Δp is set by solve_point."""
    value = float(value)
    if name in project.connections:
        if field != "area":
            raise ValueError(f"{name}: {field} cannot be changed.")
        if value < 0:
            raise ValueError(f"{name}: the orifice area must not be negative.")
        project.connections[name]["area"] = value
        for conn, settings in (build.flow.connections if build is not None else ()):
            if conn.key == name:
                settings["area"] = value
        return
    index = next((i for i, p in enumerate(project.parts) if p.name == name), None)
    if index is None:
        raise ValueError(f"{name} is not a part of the valve.")
    part = project.parts[index]
    if field not in [f.key for f in valve_fields(part)]:
        raise ValueError(f"{name}: {field} cannot be swept.")
    if field in SIZE_FIELDS and value < 0:
        raise ValueError(f"{name}: the size must not be negative.")
    part.props[field] = value
    if build is None or field in SIZE_FIELDS + MOVE_FIELDS:
        return
    if field == "pressure":
        build.flow.constant[index] = value
        return
    body = build.shells.get(index)
    if body is None:
        raise ValueError(f"{name} is not simulated.")
    if field in ("youngs_modulus", "thickness") and value <= 0:
        raise ValueError(f"{name}: the {field.replace('_', ' ')} must be positive.")
    body.set_material(**{field: value})
    if field == "thickness":
        build.thickness[index] = value


def valve_values(build, project, row, outputs) -> dict:
    """The recordable quantities of a solve_point row."""
    values = {f"out:{n}": float(v[0]) for n, v in output_pressures({"pressures": [row["pressures"]]}, outputs).items()}
    values.update(dp=row["dp"], area=row["area"], mdot=row["mdot"], p_end=row["p_end"], volume=row["volume"],
                  travel=row["travel"], warnings=[], contact=[])
    return values


def valve_extents(project) -> dict:
    """{body index: (dx, dy, dz)} the membranes are stretched to (None: that side as in the CAD); only resized ones."""
    out = {}
    for i, p in enumerate(project.parts):
        if p.role in DEFORMABLE:
            size = [float(p.props.get(k) or 0.0) for k in SIZE_FIELDS]
            if any(x > 0 for x in size):
                out[i] = tuple(round(x, 9) if x > 0 else None for x in size)
    return out


def valve_moves(project) -> dict:
    """{body index: (x, y, z)} the rigid bodies are displaced by in the set-up (Move x/y/z); only moved ones."""
    return {i: tuple(float(p.props.get(k) or 0.0) for k in MOVE_FIELDS)
            for i, p in enumerate(project.parts) if p.role == RIGID and any(p.props.get(k) for k in MOVE_FIELDS)}


class Snapshot:
    """A solved valve state (structure, fluid volumes, flow) to start a later point from."""

    def __init__(self, build, coupling):
        self.env = build.env
        self.states = [b.get_state().clone() for b in self.env.membrane_list]
        self.volumes = [(v.solved_P0, v.start_P0) for v in self.env.fluid_volume_list]
        self.flow, self.wall = copy.deepcopy(coupling.flow), np.array(coupling.wall, copy=True)

    def restore(self, coupling):
        for b, u in zip(self.env.membrane_list, self.states):
            b.set_state(u.clone())
        for v, (solved, start) in zip(self.env.fluid_volume_list, self.volumes):
            v.solved_P0, v.start_P0 = solved, start
        coupling.flow, coupling.wall, coupling.warm = copy.deepcopy(self.flow), np.array(self.wall, copy=True), True


def run_valve_grid(worker, cad, project, mesh, axes=(), dp=0.0):
    """The valve on its own (no neuron), simulated again at every point of the grid spanned by `axes` [(names,
    field, values)]: the membrane's Δp (PREACTIVATION, "dp"; `dp` when it is not swept) and any valve parameter
    (apply_valve_parameter). The points are solved in serpentine order with Δp as the innermost axis, each
    warm-started from the point before (as along the Δp sweep of the Pre-activation → activation tab), and from rest
    when that fails. With the shut-tube cut-off on (SolverSettings.shut_cutoff), every Δp line runs rising and the
    points above its first shut point are held (activation.ShutHold; row "held", and a warning so the dataset marks
    them extrapolated); each line then starts from the previous line's first point. A membrane's size (SIZE_FIELDS) changes the geometry: its axes are the outermost, and at every
    new size the CAD body is stretched (CadModel.resize), the valve meshed and built again; `mesh` is the mesh of the
    CAD as loaded (None: meshed here). A rigid body's displacement (MOVE_FIELDS) rebuilds the valve on the same mesh
    (the body is placed at CAD + move and pushes into the soft parts before any pressure, as a set-up move); its axes
    come after the size axes. The CAD is put back as it was at the end.
    Emits rows like run_grid ("coords" per display body). Returns the last build, with `display` = the display
    bodies of every build in turn (a row's coords line up with them; None for the other builds' bodies)."""
    axes = [(names, field, [float(x) for x in values]) for names, field, values in axes]
    known = {(p.name, f.key) for p in project.parts for f in valve_fields(p)}
    known |= {(k, "area") for k, s in project.connections.items() if s.get("type") == ORIFICE}
    known.add((PREACTIVATION, "dp"))
    for names, field, values in axes:
        for name in parts_of(names):
            if target(name, field) not in known:
                raise ValueError(f"{name} {field} is not a parameter of the valve.")
    dp_axis = next((k for k, (names, field, _) in enumerate(axes) if field == "dp"), None)
    pressures = [abs(x) for names, field, values in axes if field in ("dp", "pressure") for x in values]
    def rebuilds(fields):
        return [k for k, (names, field, _) in enumerate(axes)
                if any(target(n, field)[1] in fields for n in parts_of(names))]
    geometry = rebuilds(SIZE_FIELDS)
    geometry += [k for k in rebuilds(MOVE_FIELDS) if k not in geometry]
    order = geometry + [k for k in range(len(axes)) if k != dp_axis and k not in geometry] \
        + ([dp_axis] if dp_axis is not None else [])
    cutoff = project.solver.cutoff and dp_axis is not None
    if cutoff:   # every Δp line rising (held above the shut point); each line starts from the last one's first point
        rising = sorted(range(len(axes[dp_axis][2])), key=lambda i: axes[dp_axis][2][i])
        grid = [line + (i,) for line in serpentine([len(axes[k][2]) for k in order[:-1]]) for i in rising]
    else:
        grid = serpentine([len(axes[k][2]) for k in order])
    points = [tuple(idx[order.index(k)] for k in range(len(axes))) for idx in grid]
    loaded = dict(cad.extents)   # the geometry `mesh` belongs to
    build, built, display, offset, outputs, coupling = None, None, [], 0, None, None
    applied = {}
    hold, line, start, shut_coords = ShutHold(project.solver), None, None, None
    try:
        for n, idx in enumerate(points):
            worker.check()
            params = [values[i] for i, (_, _, values) in zip(idx, axes)]
            new_line = tuple(i for k, i in enumerate(idx) if k != dp_axis) != line
            line = tuple(i for k, i in enumerate(idx) if k != dp_axis)
            changed = [k for k, value in enumerate(params) if k != dp_axis and applied.get(k) != value]
            for k in changed:
                names, field, _ = axes[k]
                for name in parts_of(names):
                    apply_valve_parameter(None, project, *target(name, field), params[k])
                applied[k] = params[k]
            extents = valve_extents(project)
            moves = valve_moves(project)
            if build is None or (extents, moves) != built:
                if extents != loaded or mesh is None:
                    sizes = "; ".join(f"{project.parts[i].name} " + " x ".join(f"{x:.4g}" for x in size if x)
                                      + " mm" for i, size in extents.items())
                    worker.log("Meshing the valve" + (f" ({sizes})…" if sizes else "…"))
                    cad.resize(extents)
                    loaded = dict(extents)
                    mesh = generate_activation_mesh(cad, project)
                if moves:
                    worker.log("Building the valve (" + "; ".join(
                        f"{project.parts[i].name} moved " + ", ".join(f"{a} {x:+.4g}" for a, x in zip("xyz", m) if x)
                        + " mm" for i, m in moves.items()) + ")…")
                build = build_activation(cad, mesh, project)
                for w in build.warnings:
                    worker.log("Warning: " + w)
                built = (extents, moves)
                if outputs is None:
                    outputs = output_definitions(project.parts, {"lumen": project.parts[build.flow.lumen].name,
                                                                 "pressures": [[0.0] * (build.flow.segments + 1)]})
                build.set_contact_stiffness(max(pressures + [abs(dp)]
                                                + [abs(v) for v in build.flow.constant.values()]) * KPA)
                offset = len(display)
                display += display_bodies(build, project)
                coupling = Coupling(build, gas_of(project.study))
                start = None
            else:
                for k in changed:
                    names, field, _ = axes[k]
                    for name in parts_of(names):
                        apply_valve_parameter(build, project, *target(name, field), params[k])
            point_dp = params[dp_axis] if dp_axis is not None else float(dp)
            if new_line:
                hold.reset()
                if cutoff and start is not None:   # back to the lowest Δp: from the previous line's first point
                    start.restore(coupling)
            if hold.held(point_dp):
                row = hold.row(point_dp)
                values = valve_values(build, project, row, outputs)
                values["warnings"] = [row["message"]]   # marks the point extrapolated in the dataset
                worker.item.emit({"index": tuple(idx), "params": params, "converged": True, "time": 0.0,
                                  "values": values, "message": row["message"], "iterations": 0, "predicted": False,
                                  "held": True, "coords": shut_coords})
                worker.report((n + 1) / len(points), f"Point {n + 1}/{len(points)} (held)")
                continue
            callback = lambda lam, it, r: worker.check()  # noqa: E731
            began = time.time()
            warm = coupling.warm
            row = solve_point(build, project, point_dp, coupling, callback=callback, check=worker.check)
            if not row["converged"] and warm:   # from rest
                coupling.restart()
                row = solve_point(build, project, point_dp, coupling, callback=callback, check=worker.check)
            values = valve_values(build, project, row, outputs)
            coords = [None] * offset + [body_display_coords(b) for b in build.shells.values()]
            worker.item.emit({"index": tuple(idx), "params": params, "converged": row["converged"],
                              "time": time.time() - began, "values": values, "message": row["message"],
                              "iterations": row["iterations"], "predicted": False, "coords": coords})
            hold.add(row)
            if hold.shut is row:
                shut_coords = coords
                worker.log(f"Tube shut at Δp {point_dp:.4g} kPa (mass flow {row['mdot']:.3g} kg/s): "
                           "the higher Δp of this line are held.")
            if not row["converged"]:
                coupling.restart()
            elif cutoff and new_line:
                start = Snapshot(build, coupling)
            worker.report((n + 1) / len(points), f"Point {n + 1}/{len(points)}")
    finally:
        if cad.extents:   # leave the CAD as imported (the caller's meshes belong to it)
            cad.load_step(cad.path)
    if build is not None:
        build.display = display
    return build


def values_or_nan(row, key):
    v = row["values"].get(key, math.nan)
    return float(v) if isinstance(v, (int, float)) else math.nan


def shell_bodies(build, parts):
    """The neuron's simulated sheets as display bodies (like a design's stored bodies): name, kind, faces, rest,
    thickness, in build.shells order (the order of a row's "coords")."""
    return [{"name": parts[i].name, "kind": "shell", "faces": s.faces.cpu().numpy().astype(np.int64),
             "rest": s.X.detach().cpu().numpy().astype(float), "thickness": float(s.thickness),
             "material": s.material} for i, s in build.shells.items()]


# -----------------------------
# Characterisation dataset
# -----------------------------

def _json_number(x):
    return None if x is None or not math.isfinite(float(x)) else float(x)


def make_dataset(project, axes, rows, bodies=None, store_shapes=True, seconds=0.0) -> dict:
    """The characterisation of the linked neuron as stored in the .mfn: every recorded quantity on the grid of the
    swept parameters (points not solved: null), plus optionally the deformed sheets at every point.
    axes: [(part, field, values)]; rows: run_grid rows."""
    catalogue = {k: (label, unit) for k, label, unit in project.catalogue()}
    shape = [len(v) for _, _, v in axes]
    n = int(np.prod(shape)) if shape else 1
    keys = project.recorded()
    values = {k: [None] * n for k in keys}
    converged, extrapolated = [None] * n, [False] * n
    frames = None
    if store_shapes and bodies:
        frames = [np.full((n,) + b["rest"].shape, np.nan, np.float32) for b in bodies]
    for row in rows:
        flat = int(np.ravel_multi_index(row["index"], shape)) if shape else 0
        converged[flat] = bool(row["converged"])
        extrapolated[flat] = bool(row["values"].get("warnings"))
        for k in keys:
            values[k][flat] = _json_number(values_or_nan(row, k))
        if frames is not None and row.get("coords") is not None:
            for f, x in zip(frames, row["coords"]):
                if x is not None:   # None: a body of another mesh (a valve run that remeshed)
                    f[flat] = x
    data = {"version": 1, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": float(seconds),
            "fingerprint": project.fingerprint([(p, f) for p, f, _ in axes]),
            "axes": [_dataset_axis(project, p, f, v) for p, f, v in axes],
            "keys": [{"key": k, "label": catalogue[k][0], "unit": catalogue[k][1]} for k in keys],
            "values": values, "converged": converged, "extrapolated": extrapolated,
            "complete": all(c is not None for c in converged),
            "link": dict(project.link),
            "design": {"file": Path(project.design_path).name if project.design_path else None,
                       "dp_range": [float(x) for x in project.design().dp_range] if project.design_path else None}}
    if frames is not None:
        data["shapes"] = [{"name": b["name"], "kind": b["kind"], "faces": encode_array(b["faces"]),
                           "rest": encode_array(b["rest"]), "thickness": b["thickness"],
                           "material": b.get("material"), "frames": encode_array(f)} for b, f in zip(bodies, frames)]
    return data


def _dataset_axis(project, parts, field, values):
    """A swept parameter as stored: part (the first), also (the others it drove too), field, label, unit, values."""
    names = parts_of(parts)
    label, unit = project.axis_label(names, field)
    axis = {"part": names[0], "field": field, "label": label, "unit": unit, "values": [float(x) for x in values]}
    if len(names) > 1:
        axis["also"] = names[1:]
    return axis


class Characterisation:
    """A stored characterisation (FullNeuronProject.characterisation): the neuron's response looked up and
    interpolated without simulating it, e.g. by a workbench that connects neurons:

        c = FullNeuronProject.load("neuron.mfn").dataset()
        c.names                    # ["Weight1.pressure", "Weight2.pressure"]; "A+B.thickness" drives two parts
        c.grid("out:Output2")      # the values on the grid, shape c.shape (NaN where not solved)
        c.evaluate({"Weight1.pressure": 12.5, "Weight2.pressure": 3.0})   # {key: value}, linear interpolation
    """

    def __init__(self, data):
        self.data = data
        self.axes = [dict(a, values=np.asarray(a["values"], float)) for a in data["axes"]]
        self.names = [axis_name(axis_parts(a), a["field"]) for a in self.axes]
        self.shape = tuple(len(a["values"]) for a in self.axes)
        self.keys = [k["key"] for k in data["keys"]]
        self.labels = {k["key"]: (k["label"], k["unit"]) for k in data["keys"]}
        self.complete = bool(data.get("complete", True))
        self.fingerprint = data.get("fingerprint")

    @property
    def size(self):
        return int(np.prod(self.shape)) if self.shape else 1

    def grid(self, key):
        return np.asarray([math.nan if v is None else v for v in self.data["values"][key]], float).reshape(self.shape)

    @property
    def solved(self):
        return np.asarray([c is not None for c in self.data["converged"]]).reshape(self.shape)

    @property
    def converged(self):
        return np.asarray([bool(c) for c in self.data["converged"]]).reshape(self.shape)

    @property
    def extrapolated(self):
        return np.asarray(self.data["extrapolated"], bool).reshape(self.shape)

    def params(self, index):
        return [float(a["values"][i]) for a, i in zip(self.axes, index)]

    def out_of_range(self, point):
        """The axes whose value in `point` {name: value} is outside the characterised range."""
        out = []
        for name, a in zip(self.names, self.axes):
            x = float(point.get(name, a["values"][0]))
            lo, hi = a["values"].min(), a["values"].max()
            tol = 1e-9 * max(1.0, abs(lo), abs(hi))
            if x < lo - tol or x > hi + tol:
                out.append(name)
        return out

    def evaluate(self, point, keys=None) -> dict:
        """Every recorded quantity (or `keys`) at `point` {axis name: value}, linear on the grid (and linear
        extrapolation outside it: check out_of_range). Axes missing from `point` take their first value."""
        from scipy.interpolate import RegularGridInterpolator
        live = [k for k, a in enumerate(self.axes) if len(a["values"]) > 1]
        x = [float(point.get(self.names[k], self.axes[k]["values"][0])) for k in live]
        order = [np.argsort(self.axes[k]["values"]) for k in live]
        out = {}
        for key in (keys or self.keys):
            g = self.grid(key)[tuple(slice(None) if k in live else 0 for k in range(len(self.axes)))]
            if not live:
                out[key] = float(g)
                continue
            for d, o in enumerate(order):
                g = np.take(g, o, axis=d)
            f = RegularGridInterpolator([self.axes[k]["values"][o] for k, o in zip(live, order)], g,
                                        bounds_error=False, fill_value=None)
            out[key] = float(f(x)[0])
        return out

    def shapes(self):
        """[{"name", "kind", "faces", "rest", "thickness", "frames" (points, nodes, 3)}] or None."""
        if not self.data.get("shapes"):
            return None
        return [{"name": b["name"], "kind": b["kind"], "faces": decode_array(b["faces"]).astype(np.int64),
                 "rest": decode_array(b["rest"]).astype(float), "thickness": b.get("thickness"),
                 "material": b.get("material"), "frames": decode_array(b["frames"])} for b in self.data["shapes"]]

    def rows(self, shapes=None):
        """The solved points as window rows (like run_grid's), in grid order."""
        rows = []
        for flat, idx in enumerate(np.ndindex(*self.shape) if self.shape else [()]):
            converged = self.data["converged"][flat]
            if converged is None:
                continue
            v = {k: (math.nan if self.data["values"][k][flat] is None else self.data["values"][k][flat])
                 for k in self.keys}
            v["warnings"] = ["The pre-activation Δp is outside the design's simulated range: EXTRAPOLATING, not "
                             "interpolating."] if self.data["extrapolated"][flat] else []
            rows.append({"index": tuple(idx), "params": self.params(idx), "converged": converged, "time": math.nan,
                         "values": v, "message": "",
                         "coords": [b["frames"][flat].astype(float) for b in shapes] if shapes else None})
        return rows
