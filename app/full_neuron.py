"""
Full-neuron workbench (no GUI): an inputs-to-pre-activation project (*.mns) and a pre-activation-to-activation
design (*.mad) linked into one neuron and solved as in the neuron space, with the design's membrane replaced by
its pre-simulated model (the "Activation membrane" role on the linked neuron part).

Here only the fluids' parameters can be changed (pressures, volumes, stiffness, ...), not the kinds of anything
(roles, chamber models) and not the design (that would need simulating it again). The neuron project is embedded
in the workbench file (*.mfn), so the original .mns is not changed. What to record at every solve is chosen from
the chamber pressures and volumes and the design's pre-activation Δp, named outputs, tube area and mass flow.

Both models are shown side by side; each has a position and a rotation (about its own centre) that only
affect the display.

Characterisation: any number of chamber parameters (pressures, ghost volumes, liquid shares, stiffnesses, fluid
volumes) are swept together over a grid and any number of quantities recorded at every point. The dataset is saved in
the .mfn (`characterisation`, see `make_dataset` and `Characterisation`), so that the neuron can later be used
(looked up and interpolated) without simulating it again.
"""
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from .activation import ActivationDesign, ActivationProject, decode_array, encode_array
from .builder import KPA, build_environment, chamber_model
from .project import (ACTIVATION_MEMBRANE, AUTOMATIC, CHAMBER, CONSTANT, IDEAL_GAS, INCOMPRESSIBLE, ROLE_FIELDS,
                      SHEETS,
                      Project, _relative)

SUFFIX = ".mfn"
FILE_TYPE = "full_neuron"
NEURON, ACTIVATION = "neuron", "activation"   # the two models (transform keys)

# fields of a chamber that may be changed here: everything but its model
LOCKED_FIELDS = {"model"}
# chamber fields that can be swept (numbers)
SWEEPABLE = [f for f in ROLE_FIELDS[CHAMBER] if f.kind == "float" and f.key not in LOCKED_FIELDS]


def default_axis(part=None, field="pressure"):
    return {"part": part, "field": field, "from": 0.0, "to": 20.0, "points": 5}


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

class FullNeuronProject:
    def __init__(self):
        self.neuron = None          # Project (embedded copy; its fluid parameters are edited here)
        self.neuron_source = None   # the .mns it was imported from (for information)
        self.design_path = None     # the .mad (not embedded: the design is fixed)
        self.link = {"part": None, "driving": AUTOMATIC}   # neuron part replaced by the design's membrane
        self.transforms = {NEURON: identity(), ACTIVATION: identity()}
        self.record = None          # keys of recorded quantities (None: the defaults)
        self.sweep = {"axes": [], "store_shapes": True}   # axes: [{"part", "field", "from", "to", "points"}]
        self.characterisation = None   # the stored dataset (make_dataset) or None
        self.path = None

    # -----------------------------

    def import_neuron(self, path):
        project = Project.load(path)
        self.neuron, self.neuron_source = project, str(Path(path).resolve())
        linked = [p.name for p in project.parts if p.role == ACTIVATION_MEMBRANE]
        if linked:  # the project already names the membrane the design replaces
            part = next(p for p in project.parts if p.name == linked[0])
            self.link = {"part": part.name, "driving": part.props.get("driving", AUTOMATIC) or AUTOMATIC}
        elif self.link["part"] not in self.link_candidates():
            self.link = {"part": None, "driving": AUTOMATIC}
        self.record = None
        return project

    def import_design(self, path):
        project = ActivationProject.load(path)
        if not project.results:
            raise ValueError(f"{Path(path).name} has no simulated results: simulate it in the Pre-activation → "
                             "activation space and save it first.")
        design = ActivationDesign(project.results, Path(path).stem)
        if not design.has_volume:
            raise ValueError(f"{Path(path).name} was simulated before the membrane's swept volume was recorded: "
                             "open it in the Pre-activation → activation space, simulate it again and save it.")
        self.design_path = str(Path(path).resolve())
        self.record = None
        return project

    def design(self) -> ActivationDesign:
        """The linked design (cached until the file changes: decoding its stored FEM solution takes a moment)."""
        path = Path(self.design_path)
        key = (str(path), path.stat().st_mtime_ns)
        if getattr(self, "_design_key", None) != key:
            self._design, self._design_key = ActivationDesign.load(path), key
        return self._design

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

    def parameters(self):
        """[(part, field, label, unit)] of every chamber parameter that can be swept (those its model uses)."""
        out = []
        for p in self.chambers():
            for f in SWEEPABLE:
                if f.visible_if is None or p.props.get(f.visible_if[0]) in f.visible_if[1]:
                    out.append((p.name, f.key, f"{p.name} {f.label.lower().replace(' (gauge)', '')}", f.unit))
        return out

    def sweep_axes(self):
        """The valid sweep axes as [(part, field, values)] (unknown parameters and repeats left out)."""
        known = {(p, f) for p, f, _, _ in self.parameters()}
        out, seen = [], set()
        for a in self.sweep["axes"]:
            key = (a.get("part"), a.get("field"))
            if key in known and key not in seen:
                seen.add(key)
                out.append((key[0], key[1], axis_values(a)))
        return out

    def fingerprint(self, swept=()) -> str:
        """What the characterisation depends on: the neuron's parameters (except the swept ones and display
        settings), the solver settings, the link and the design file's contents."""
        data = copy.deepcopy(self.neuron.to_dict()) if self.neuron else {}
        data.pop("step_path", None)
        swept = {tuple(s) for s in swept}
        for part in data.get("parts", []):
            part.pop("visible", None)
            part["props"].pop("design", None)
            for p, f in swept:
                if part["name"] == p:
                    part["props"].pop(f, None)
        design = None
        if self.design_path and Path(self.design_path).exists():
            path = Path(self.design_path)
            key = (str(path), path.stat().st_mtime_ns)
            if getattr(self, "_hash_key", None) != key:
                self._hash, self._hash_key = hashlib.sha1(path.read_bytes()).hexdigest(), key
            design = self._hash
        text = json.dumps({"neuron": data, "link": self.link, "design": design}, sort_keys=True, default=str)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def dataset(self):
        """The stored characterisation as a Characterisation (None without one)."""
        return Characterisation(self.characterisation) if self.characterisation else None

    def dataset_current(self) -> bool:
        """Whether the stored characterisation still matches the model (same parameters, link and design)."""
        c = self.characterisation
        return bool(c) and c.get("fingerprint") == self.fingerprint([(a["part"], a["field"]) for a in c["axes"]])

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
        elif self.design_path:
            raise ValueError("Link the design: choose the neuron membrane it replaces (Link tab).")
        return project

    # -----------------------------
    # Recording
    # -----------------------------

    def catalogue(self):
        """[(key, label, unit)] of everything that can be recorded."""
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
        return out

    def default_record(self):
        """The pre-activation pressure (closed chambers) and every output of the design."""
        keys = [k for k, _, _ in self.catalogue()]
        closed = {f"P:{p.name}" for p in self.chambers() if p.props.get("model") in (IDEAL_GAS, INCOMPRESSIBLE)}
        chosen = [k for k in keys if k in closed or k == "dp" or k.startswith("out:")]
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
        return {"type": FILE_TYPE, "version": 1, "neuron": neuron,
                "neuron_source": _relative(self.neuron_source, path) if self.neuron_source else None,
                "design": _relative(self.design_path, path) if self.design_path else None,
                "link": self.link, "transforms": self.transforms, "record": self.record, "sweep": self.sweep,
                "characterisation": self.characterisation}

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
        project.design_path = resolve(data.get("design"))
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
        project.characterisation = data.get("characterisation")
        project.path = str(path)
        return project


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
    values["warnings"] = build.range_warnings(parts)
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
    """Set a chamber parameter on the part and on its FluidVolume in the built model (no rebuild)."""
    part.props[field] = float(value)
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
    axes = [(name, field, [float(x) for x in values]) for name, field, values in axes]
    for name, field, _ in axes:
        if index.get(name) not in build.volumes:
            raise ValueError(f"{name} is not a chamber of the neuron.")
    pressures = [abs(x) * KPA for _, field, values in axes if field == "pressure" for x in values]
    build.set_contact_stiffness(max([abs(v.P0) for v in build.volumes.values()] + pressures + [KPA]))
    points = serpentine([len(values) for _, _, values in axes])
    states = {}   # grid index -> solved state (for the next points' starting guesses)
    previous = None
    for k, idx in enumerate(points):
        worker.check()
        params = [values[i] for i, (_, _, values) in zip(idx, axes)]
        for (name, field, _), value in zip(axes, params):
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
        for w in row_values["warnings"]:
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
    params = {(p, f): (label, unit) for p, f, label, unit in project.parameters()}
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
                f[flat] = x
    data = {"version": 1, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": float(seconds),
            "fingerprint": project.fingerprint([(p, f) for p, f, _ in axes]),
            "axes": [{"part": p, "field": f, "label": params.get((p, f), (f"{p} {f}", ""))[0],
                      "unit": params.get((p, f), ("", ""))[1], "values": [float(x) for x in v]} for p, f, v in axes],
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


class Characterisation:
    """A stored characterisation (FullNeuronProject.characterisation): the neuron's response looked up and
    interpolated without simulating it, e.g. by a workbench that connects neurons:

        c = FullNeuronProject.load("neuron.mfn").dataset()
        c.names                    # ["Weight1.pressure", "Weight2.pressure"]
        c.grid("out:Output2")      # the values on the grid, shape c.shape (NaN where not solved)
        c.evaluate({"Weight1.pressure": 12.5, "Weight2.pressure": 3.0})   # {key: value}, linear interpolation
    """

    def __init__(self, data):
        self.data = data
        self.axes = [dict(a, values=np.asarray(a["values"], float)) for a in data["axes"]]
        self.names = [f"{a['part']}.{a['field']}" for a in self.axes]
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
