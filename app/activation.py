"""
Activation-function space ("pre-activation to activation"): CAD + roles -> a squeezed-tube model with gas
flowing through it, the Δp sweep that maps the membrane's pressure difference (the pre-activation) to
the tube's area, mass flow and pressures, and the saved design file. The outputs (activations) are
named segments of the tube's dynamic fluid (its "outputs" property, [{"name", "segment"}]); each is the
gas pressure of its segment, read from the stored node pressures, so adding or changing outputs needs no
new simulation. Without any, the downstream segment is the one output "activation".

The device: a membrane (loaded by the pressure difference Δp across it) pushes, through a part
bonded to it (a free rigid body or a deformable solid, e.g. a pusher), on a soft tube (the channel)
that rests against a fixed rigid body. Gas flows through the tube between fluid bodies:

  Fluid, constant       -> a supply or sink at a set pressure (e.g. the inlet, or 0 kPa at the far end)
  Fluid, dynamic        -> pressure follows from the flow. The one filling the tube is cut into N
                           segments along the tube, each a flow resistance in series; each segment's
                           pressure acts on its piece of the tube wall
  Flow connection       -> where two fluids touch (or a dynamic fluid touches the outside): an opening
                           (no resistance), an orifice (user's law) or closed
  Membrane / Shell      -> Shell on the mid-surface, clamped along its edges, loaded by Δp
                           (positive Δp pushes the membrane towards the tube); bonded to every free
                           rigid body or solid its face touches
  Channel (tube)        -> Solid (10-node tetrahedra), fixed at its end faces (the planar faces
                           normal to the channel axis)
  Solid                 -> Solid (tetrahedra), free or fixed where it touches fixed rigid bodies;
                           contact with the tube, other solids and free rigid bodies
  Rigid body, fixed     -> obstacle (contact with the tube, solids and membranes)
  Rigid body, free      -> RigidBody (6 dofs), moved by the membrane bond and contact

At every Δp the deformation and the flow are iterated: solve the structure with the current wall
pressures, measure every segment's cross-section, solve the flow network (steady mass balance, ideal
gas density) for the pressures, put them back on the wall, until the pressures stop changing.

The design file (*.mad, JSON) keeps the roles, connections, settings and the computed mapping, so
the device can be used as a part without simulating it again.
"""
import base64
import json
import zlib
from types import SimpleNamespace
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

import membrane_sim as ms
from membrane_sim.contact import ObstacleField, _winding_number
from membrane_sim.fluid import boundary_loops
from membrane_sim.flow import ORIFICE_LAW, SEGMENT_LAW, FlowNetwork, Gas, compile_flow_law
from membrane_sim.lumen import ChannelSections
from membrane_sim.rigid import MovingContact, RigidBody, RigidTie
from membrane_sim.shell_contact import ShellContact
from membrane_sim.solid import Solid
from membrane_sim.solid_contact import SolidContact, SurfaceTie

from .builder import KPA, element_size
from .project import (ACTIVATION_SPACE, CHANNEL, CLAMPED, CLOSED, CONSTANT_FLUID, DEFORMABLE, DYNAMIC_FLUID, FLUID,
                      FREE, FREE_RIGID_COLOR, IGNORE, LEGACY_ROLES, NEO_HOOKEAN, OPENING, ORIFICE, RIGID, ROLE_COLORS,
                      SHELL, SOLID, SOLID_FIXED, TOUCHING_RIGID, UNASSIGNED, Field, PartSettings, Project,
                      SolverSettings)

DESIGN_SUFFIX = ".mad"
DESIGN_TYPE = "activation_function_design"
OUTSIDE = "Outside"


# -----------------------------
# Project and study settings
# -----------------------------

@dataclass
class StudySettings:
    dp_min: float = 0.0          # kPa
    dp_max: float = 20.0         # kPa
    points: int = 21
    coupling_tolerance: float = 0.01   # kPa, change of the wall pressures between flow iterations
    max_coupling_iterations: int = 30
    gas_constant: float = 287.05       # J/(kg K), air
    temperature: float = 293.15        # K
    atmospheric_pressure: float = 101.325  # kPa
    viscosity: float = 1.81e-5         # Pa s, air at 20 °C


STUDY_FIELDS = [
    Field("dp_min", "Δp from", "float", 0.0, "kPa", -1e6, 1e6, 4,
          tooltip="Pressure difference across the membrane (positive pushes it towards the tube)."),
    Field("dp_max", "Δp to", "float", 20.0, "kPa", -1e6, 1e6, 4),
    Field("points", "Points", "int", 21, "", 2, 1000),
    Field("coupling_tolerance", "Pressure tolerance", "float", 0.01, "kPa", 1e-9, 1e6, 6,
          tooltip="The flow sets the pressures on the tube wall, the wall pressures change the tube's shape and so "
                  "the flow: they are iterated until no wall pressure changes by more than this."),
    Field("max_coupling_iterations", "Max flow iterations", "int", 30, "", 1, 1000),
    Field("gas_constant", "Gas constant R", "float", 287.05, "J/(kg K)", 1e-6, 1e9, 4,
          tooltip="Specific gas constant of the gas (air 287.05). Density: ρ = p_abs / (R T)."),
    Field("temperature", "Temperature", "float", 293.15, "K", 1e-6, 1e6, 4),
    Field("atmospheric_pressure", "Atmospheric pressure", "float", 101.325, "kPa", 0.0, 1e6, 4,
          tooltip="All pressures in the model are gauge; the gas law uses absolute pressure."),
    Field("viscosity", "Viscosity μ", "float", 1.81e-5, "Pa s", 0.0, 1e6, 8),
]


def connection_key(name_a, name_b):
    a, b = sorted([name_a, name_b]) if name_b != OUTSIDE else (name_a, name_b)
    return f"{a} ↔ {b}"


class ActivationProject(Project):
    space = ACTIVATION_SPACE
    suffix = DESIGN_SUFFIX

    def __init__(self, step_path=None, part_names=()):
        super().__init__(step_path, part_names)
        self.study = StudySettings()
        self.connections = {}     # connection key -> {"type": OPENING | ORIFICE | CLOSED, "law": text}
        self.results = None       # dict, see run_study

    def connection(self, key, outside=False):
        """Settings of a flow connection (created with the default: open between fluids, closed to the outside)."""
        return self.connections.setdefault(key, {"type": CLOSED if outside else OPENING, "law": ORIFICE_LAW})

    def to_dict(self):
        data = super().to_dict()
        data.update({"type": DESIGN_TYPE, "study": asdict(self.study), "connections": self.connections,
                     "results": self.results})
        return data

    @classmethod
    def load(cls, path):
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("type") != DESIGN_TYPE:
            raise ValueError("Not an activation-function design file.")
        project = cls()
        step = Path(data["step_path"])
        project.step_path = str(step if step.is_absolute() else (path.parent / step).resolve())
        project.parts = [PartSettings(**p) for p in data["parts"]]
        for part in project.parts:  # first version: input/output channel sides
            if part.role in LEGACY_ROLES:
                constant = part.role == "Channel input side"
                part.role = FLUID
                part.props = {"model": CONSTANT_FLUID if constant else DYNAMIC_FLUID,
                              "pressure": float(part.props.get("pressure", 0.0)), "segments": 10,
                              "segment_law": SEGMENT_LAW}
        project.solver = SolverSettings(**data.get("solver", {}))
        known = StudySettings.__dataclass_fields__
        project.study = StudySettings(**{k: v for k, v in data.get("study", {}).items() if k in known})
        project.connections = data.get("connections", {})
        project.results = data.get("results")
        if project.results and "mdot" not in project.results:
            project.results = None  # mapping of the first version (no flow): simulate again
        segment = data.get("output_segment")  # the single output of the version before named outputs
        lumen = next((q for q in project.parts if project.results and q.name == project.results.get("lumen")), None)
        if segment and lumen is not None and not lumen.props.get("outputs"):
            lumen.props["outputs"] = [{"name": "activation", "segment": int(segment)}]
        project.path = str(path)
        return project


def segment_pressures(results) -> np.ndarray:
    """(points, N) gas pressure of every tube segment (kPa): the mean of its two end nodes, as on the wall."""
    P = np.asarray(results["pressures"], float)
    return 0.5 * (P[:, :-1] + P[:, 1:])


# -----------------------------
# The stored FEM solution (the deformed device at every simulated point)
# -----------------------------

def encode_array(a) -> dict:
    """A numpy array as compressed float32 (or int32) base64 text, for the JSON design file."""
    a = np.ascontiguousarray(a, dtype=np.int32 if np.issubdtype(np.asarray(a).dtype, np.integer) else np.float32)
    return {"dtype": str(a.dtype), "shape": list(a.shape),
            "data": base64.b64encode(zlib.compress(a.tobytes(), 6)).decode("ascii")}


def decode_array(d) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(d["data"])), dtype=d["dtype"]).reshape(d["shape"])


def body_kind(body):
    return "rigid" if getattr(body, "is_rigid", False) else "solid" if hasattr(body, "tets") else "shell"


def body_display_coords(body) -> np.ndarray:
    """What is stored of a body per point: node coordinates of a shell, surface-node coordinates of a solid, the
    posed surface mesh of a rigid body."""
    x = body.x.detach().cpu().numpy()
    if body_kind(body) == "solid":
        return x[body.surface_nodes.cpu().numpy()]
    return x


def encode_fem(build, project, frames) -> dict:
    """The deformed device at every point: per body its display triangles, rest coordinates and one frame per
    point (float32, compressed), keyed to the part names."""
    parts = project.parts
    bodies = []
    for k, (index, body) in enumerate(build.shells.items()):
        kind = body_kind(body)
        faces = body.faces_np if hasattr(body, "faces_np") else body.faces.cpu().numpy()
        rest = body.X.detach().cpu().numpy()
        if kind == "solid":  # only the surface: renumber its nodes
            nodes = body.surface_nodes.cpu().numpy()
            renumber = np.full(len(rest), -1, np.int64)
            renumber[nodes] = np.arange(len(nodes))
            faces, rest = renumber[np.asarray(faces)], rest[nodes]
        entry = {"name": parts[index].name, "kind": kind, "faces": encode_array(np.asarray(faces)),
                 "rest": encode_array(rest), "frames": encode_array(np.stack([f[k] for f in frames]) if frames
                                                                  else np.zeros((0,) + rest.shape))}
        if kind == "shell":
            entry["thickness"] = float(body.thickness)
            entry["material"] = body.material
        bodies.append(entry)
    return {"version": 1, "bodies": bodies}


class DesignFrames:
    """The stored FEM solution of a design, interpolated between the simulated points:
    at(dp) -> [{"name", "kind", "faces", "rest", "x", "thickness"}] (x: coordinates at dp, mm)."""

    def __init__(self, fem, dp_all, keep):
        """fem: results["fem"]; dp_all: results["dp"]; keep: indices of the points used (converged), sorted by dp."""
        self.dp = np.asarray(dp_all, float)[keep]
        self.bodies = []
        for b in fem["bodies"]:
            frames = decode_array(b["frames"]).astype(float)
            self.bodies.append({"name": b["name"], "kind": b["kind"], "faces": decode_array(b["faces"]).astype(np.int64),
                                "rest": decode_array(b["rest"]).astype(float), "frames": frames[keep],
                                "thickness": b.get("thickness"), "material": b.get("material")})

    def at(self, dp):
        """Every body at pre-activation dp: linear between the neighbouring points, the end point outside."""
        if len(self.dp) == 1:
            i, w = 0, 0.0
        else:
            dp = min(max(float(dp), self.dp[0]), self.dp[-1])
            i = int(np.clip(np.searchsorted(self.dp, dp) - 1, 0, len(self.dp) - 2))
            w = (dp - self.dp[i]) / (self.dp[i + 1] - self.dp[i]) if self.dp[i + 1] > self.dp[i] else 0.0
        out = []
        for b in self.bodies:
            f = b["frames"]
            x = f[i] if len(f) == 1 or w == 0.0 else (1.0 - w) * f[i] + w * f[i + 1]
            out.append({**{k: b[k] for k in ("name", "kind", "faces", "rest", "thickness", "material")}, "x": x})
        return out


DEFAULT_OUTPUT = "activation"


def clean_outputs(outputs, segments=None):
    """[{"name", "segment"}] with names made unique and non-empty, segments clamped to 1..segments."""
    out, names = [], set()
    for k, o in enumerate(outputs or ()):
        name = str(o.get("name", "") or "").strip() or f"output {k + 1}"
        base, n = name, 2
        while name in names:
            name, n = f"{base} ({n})", n + 1
        names.add(name)
        segment = max(int(o.get("segment", 1)), 1)
        out.append({"name": name, "segment": min(segment, segments) if segments else segment})
    return out


def output_definitions(parts, results=None):
    """The named outputs of a design: those of the fluid filling the tube (results["lumen"]), else of the first
    dynamic fluid that has any; without any, the downstream segment as the one output "activation"."""
    segments = len(results["pressures"][0]) - 1 if results and results.get("pressures") else None
    fluids = [q for q in parts if q.role == FLUID and q.props.get("model") == DYNAMIC_FLUID]
    lumen = next((q for q in fluids if results and q.name == results.get("lumen")), None)
    source = lumen if lumen is not None else next((q for q in fluids if q.props.get("outputs")), None)
    outputs = clean_outputs(source.props.get("outputs") if source is not None else (), segments)
    if not outputs:
        n = segments or (int(source.props.get("segments", 10)) if source is not None else 1)
        outputs = [{"name": DEFAULT_OUTPUT, "segment": n}]
    return outputs


def output_pressures(results, outputs) -> dict:
    """{output name: (points,) gas pressure of its segment (kPa)}."""
    P = segment_pressures(results)
    return {o["name"]: P[:, min(o["segment"], P.shape[1]) - 1] for o in outputs}


def channel_axis(surface, lumen, connections, parts):
    """(unit axis, centre) of the fluid filling the tube: its long direction, pointing from the higher-pressure
    end (constant fluids and outside openings it connects to). connections: [(Connection, settings)]."""
    pts = surface.vertices
    center = _centroid(surface)
    _, _, axes = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
    axis = axes[0]
    ends = []  # (position along the axis, pressure) of constant fluids and outside openings at the tube
    for conn, settings in connections:
        if lumen not in (conn.a, conn.b) or settings["type"] == CLOSED:
            continue
        other = conn.b if conn.a == lumen else conn.a
        if other is None:
            ends.append((float((conn.center - center) @ axis), 0.0))
        elif parts[other].props.get("model") == CONSTANT_FLUID:
            ends.append((float((conn.center - center) @ axis), float(parts[other].props.get("pressure", 0.0))))
    if len(ends) >= 2:
        upstream = max(ends, key=lambda e: e[1])
        if upstream[0] > 0:
            axis = -axis
    return axis, center


def segment_faces(surface, axis, segments, k):
    """Faces of a fluid's surface mesh in segment k (1..segments) along the axis (equal lengths, as the model)."""
    s = surface.vertices[surface.faces].mean(axis=1) @ axis
    ends = surface.vertices @ axis
    bounds = np.linspace(ends.min(), ends.max(), segments + 1)
    return np.nonzero((s >= bounds[k - 1]) & (s <= bounds[k]))[0]


class ActivationDesign:
    """
    A saved activation-function design used as a part: the mapping Δp -> A, mass flow and the
    pressures along the tube, read from its .mad file and interpolated between the simulated points.

        design = ActivationDesign.load("valve.mad")
        design.output("activation", 12.5)  # a named output: pressure of its tube segment at Δp = 12.5 kPa (kPa)
        design.area(12.5)            # smallest cross-section (mm²)
        design.mass_flow(12.5)       # kg/s
        design.end_pressure(12.5)    # pressure at the downstream end of the tube (kPa)
        design.swept_volume(12.5)    # volume the membrane swept towards the tube (mm³)

    The swept volume is what a neuron needs to use the device's membrane as a wall of its pre-activation
    chamber (the "Activation membrane" role); designs simulated before it was recorded have none
    (has_volume False) and must be simulated again for that.
    """

    def __init__(self, results: dict, name: str = "", outputs=None):
        self.name = name
        self.results = results
        self.output_list = clean_outputs(outputs, len(results["pressures"][0]) - 1) or \
            [{"name": DEFAULT_OUTPUT, "segment": len(results["pressures"][0]) - 1}]
        ok = np.asarray(results["converged"], bool)
        if not ok.any():
            raise ValueError(f"{name or 'The design'} has no converged points.")
        order = np.argsort(np.asarray(results["dp"], float)[ok])
        pick = lambda key: np.asarray(results[key], float)[ok][order]  # noqa: E731
        self.dp, self.A, self.mdot, self.p_end = pick("dp"), pick("area"), pick("mdot"), pick("p_end")
        self.p_out = {name: v[ok][order] for name, v in output_pressures(results, self.output_list).items()}
        self.travel_ = pick("travel") if "travel" in results else None
        self.A0 = float(results["A0"])
        self.has_volume = "volume" in results and "membrane_area" in results
        keep = np.nonzero(ok)[0][order]
        self.frames = DesignFrames(results["fem"], results["dp"], keep) if results.get("fem") else None
        self.volume = pick("volume") if self.has_volume else None
        self.membrane_area = float(results["membrane_area"]) if self.has_volume else None

    @classmethod
    def load(cls, path):
        project = ActivationProject.load(path)
        if not project.results:
            raise ValueError(f"{path} holds no simulated mapping yet.")
        return cls(project.results, Path(path).stem, output_definitions(project.parts, project.results))

    def area(self, dp):
        return np.interp(dp, self.dp, self.A)

    def mass_flow(self, dp):
        return np.interp(dp, self.dp, self.mdot)

    def end_pressure(self, dp):
        return np.interp(dp, self.dp, self.p_end)

    @property
    def output_names(self):
        return [o["name"] for o in self.output_list]

    def output(self, name, dp):
        return np.interp(dp, self.dp, self.p_out[name])

    def swept_volume(self, dp):
        return np.interp(dp, self.dp, self.volume)

    @property
    def dp_range(self):
        return float(self.dp[0]), float(self.dp[-1])

    def extrapolating(self, dp) -> bool:
        lo, hi = self.dp_range
        return not (lo - 1e-9 <= dp <= hi + 1e-9)

    def range_warning(self, dp, where="") -> str:
        """The warning shown when the pre-activation leaves the simulated range ("" inside it)."""
        if not self.extrapolating(dp):
            return ""
        lo, hi = self.dp_range
        side = "above" if dp > hi else "below"
        return (f"{where + ': ' if where else ''}pre-activation Δp = {dp:.4g} kPa is {side} the simulated range of "
                f"{self.name or 'the design'} ({lo:.4g} … {hi:.4g} kPa): EXTRAPOLATING, not interpolating - the "
                f"membrane's swept volume is extended linearly and the outputs are held at their end values.")

    def outputs(self, dp) -> dict:
        """The device's outputs at a pressure difference dp (kPa) across its membrane; `extrapolated`
        when dp lies outside the simulated range (the outputs are then those at the nearest end)."""
        lo, hi = self.dp_range
        out = {"dp": float(dp), "outputs": {name: float(self.output(name, dp)) for name in self.output_names},
               "area": float(self.area(dp)), "mdot": float(self.mass_flow(dp)), "p_end": float(self.end_pressure(dp)),
               "extrapolated": self.extrapolating(dp)}
        if self.travel_ is not None:
            out["travel"] = float(np.interp(dp, self.dp, self.travel_))
        return out


# -----------------------------
# Flow connections between fluid bodies
# -----------------------------

@dataclass
class Connection:
    key: str
    a: int                  # fluid body index
    b: object               # other fluid body index, or None for the outside
    faces: np.ndarray       # faces of a's surface mesh on the interface
    area: float             # mm²
    perimeter: float        # mm
    height: float           # mm (smaller in-plane extent)
    width: float            # mm (larger in-plane extent)
    center: np.ndarray


def _probe_targets(surface, meshes, eps):
    """For every face of `surface`, the index into `meshes` of the body just outside it (-1: nothing)."""
    V, F = surface.vertices, surface.faces
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    probes = torch.as_tensor(V[F].mean(axis=1) + eps * n)
    target = np.full(len(F), -1)
    best = np.full(len(F), 0.5)
    for k, m in enumerate(meshes):
        tri = torch.as_tensor(m.vertices[m.faces])
        w = torch.cat([_winding_number(p, tri[:, 0], tri[:, 1], tri[:, 2]) for p in probes.split(256)]).numpy()
        better = w > best
        target[better], best[better] = k, w[better]
    return target


def _patch_geometry(V, F):
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    perimeter = sum(np.linalg.norm(V[np.roll(loop, -1)] - V[loop], axis=1).sum() for loop in boundary_loops(F))
    pts = V[np.unique(F)]
    center = (area[:, None] * tri.mean(axis=1)).sum(axis=0) / area.sum()
    _, _, axes = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
    extents = sorted([np.ptp(pts @ axes[0]), np.ptp(pts @ axes[1])])
    return float(area.sum()), float(perimeter), extents[0], extents[1], center


def detect_connections(surfaces, parts, bodies):
    """
    Flow connections: where a fluid body's surface lies against another fluid body, and where a dynamic
    fluid's surface faces the outside (touches no other body). Found by probing just outside every face
    of each fluid with the other bodies' winding numbers.
    """
    fluids = [i for i, p in enumerate(parts) if p.role == FLUID and i in surfaces]
    others = [i for i, p in enumerate(parts) if p.role not in (IGNORE, UNASSIGNED) and i in surfaces]
    out, seen = [], set()
    for i in fluids:
        candidates = [j for j in others if j != i]
        m = surfaces[i]
        eps = 1e-3 * bodies[i].diagonal
        target = _probe_targets(m, [surfaces[j] for j in candidates], eps)
        tri = m.vertices[m.faces]
        face_area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        groups = {}
        for k, j in enumerate(candidates):
            if parts[j].role == FLUID and (k == target).any():
                groups[j] = np.nonzero(target == k)[0]
        if parts[i].props.get("model") == DYNAMIC_FLUID:
            outside = np.nonzero(target < 0)[0]
            if len(outside):  # the facets of two curved CAD faces leave thin gaps: look further out
                sub = SimpleNamespace(vertices=m.vertices, faces=m.faces[outside])
                again = _probe_targets(sub, [surfaces[j] for j in candidates], 0.02 * bodies[i].diagonal)
                outside = outside[again < 0]
            if face_area[outside].sum() > 0.005 * face_area.sum():  # ignore probing noise at sharp edges
                groups[None] = outside
        for j, faces in groups.items():
            pair = (min(i, j), max(i, j)) if j is not None else (i, None)
            if pair in seen:
                continue
            seen.add(pair)
            key = connection_key(parts[i].name, parts[j].name if j is not None else OUTSIDE)
            area, perimeter, h, w, center = _patch_geometry(m.vertices, m.faces[faces])
            out.append(Connection(key, i, j, faces, area, perimeter, h, w, center))
    return out


# -----------------------------
# Meshing
# -----------------------------

@dataclass
class ActivationMesh:
    sizes: dict
    surfaces: dict                                    # body index -> SurfaceMesh
    midsurfaces: dict = field(default_factory=dict)   # membranes
    volumes: dict = field(default_factory=dict)       # body index -> VolumeMesh (channel and solids)
    channel: int = None

    @property
    def volume(self):
        return self.volumes.get(self.channel)

    def element_count(self):
        return sum(len(m.faces) for m in self.midsurfaces.values()) + sum(len(v.tets) for v in self.volumes.values())


def _roles(project, role):
    return [i for i, p in enumerate(project.parts) if p.role == role]


def free_rigid(project):
    return [i for i in _roles(project, RIGID) if project.parts[i].props.get("motion") == FREE]


def validate(project):
    """Raise ValueError with a readable message when the roles do not make a device."""
    problems = []
    if len(_roles(project, CHANNEL)) != 1:
        problems.append("exactly one part must be the Channel (tube)")
    if not [i for i in _roles(project, FLUID) if project.parts[i].props.get("model") == DYNAMIC_FLUID]:
        problems.append("the fluid filling the tube must be a Fluid with dynamic pressure")
    if not [i for i, p in enumerate(project.parts) if p.role in DEFORMABLE]:
        problems.append("assign the membrane (Membrane or Shell)")
    if problems:
        raise ValueError("To simulate the activation function, " + "; ".join(problems) + ".")


def solid_size(body, part, default=6):
    return element_size(body, RIGID, part.props.get("elements_per_side", default))


def _order(part):
    return 1 if str(part.props.get("element_order", "")).startswith("Linear") else 2


def generate_activation_mesh(cad, project) -> ActivationMesh:
    validate(project)
    c = _roles(project, CHANNEL)[0]
    volumes = {}
    for i in [c] + _roles(project, SOLID):   # volume meshes first: cad.mesh() below replaces gmsh's mesh
        part = project.parts[i]
        volumes[i] = cad.volume_mesh(i, solid_size(cad.bodies[i], part, 6 if i == c else 4), order=_order(part))
    sizes = {}
    for body, p in zip(cad.bodies, project.parts):
        n = p.props.get("elements_per_side", 10)
        sizes[body.index] = element_size(body, p.role if p.role in DEFORMABLE else RIGID, n)
    data = ActivationMesh(sizes, cad.mesh(sizes), volumes=volumes, channel=c)
    for body, p in zip(cad.bodies, project.parts):
        if p.role in DEFORMABLE:
            data.midsurfaces[body.index] = cad.midsurface(body, data.surfaces[body.index])
    return data


# -----------------------------
# Model
# -----------------------------

@dataclass
class FlowModel:
    """The gas side of the device: the tube's segments, the fluid bodies and their connections."""
    lumen: int                  # index of the dynamic fluid filling the tube
    segments: int
    bounds: np.ndarray          # (N+1,) positions of the segment boundaries along the axis (rest, mm)
    station_segment: np.ndarray  # segment of every cross-section station
    segment_law: object
    connections: list           # [(Connection, settings dict)]
    wall: list                  # [(FluidVolume, ("segment", k) | ("fluid", j))] pressures on the tube wall
    names: dict                 # fluid index -> part name
    constant: dict              # fluid index -> pressure (kPa) of constant fluids

    def node_positions(self):
        return self.bounds - self.bounds[0]


@dataclass
class ActivationBuild:
    env: ms.Environment
    membranes: dict            # body index -> Shell
    tube: Solid
    movers: dict               # body index -> RigidBody (free rigid bodies) or Solid (solid parts)
    obstacles: dict            # body index -> Obstacle (fixed rigid bodies)
    load: ms.FluidVolume       # Δp on the membrane(s)
    flow: FlowModel
    sections: ChannelSections
    axis: np.ndarray
    squeeze: np.ndarray        # direction from the membrane towards the tube (closing direction)
    channel_height: float      # inside height of the tube along the closing direction (mm)
    fixed_tube_nodes: int
    tie_nodes: dict            # membrane index -> bonded node indices
    bonds: list                # (membrane index, body index, number of nodes)
    wall_faces: np.ndarray     # tube faces lining the tube's inside
    ties: list
    contacts: list
    self_contact: ShellContact
    thickness: dict
    warnings: list = field(default_factory=list)
    contact_stiffness: float = 0.0
    auto_contact: bool = True
    shells: dict = field(default_factory=dict)  # body index -> body with dofs, for display

    @property
    def A0(self):
        return float(self.sections.rest_areas.min())

    @property
    def membrane_area(self):
        """Rest area of the membrane(s) Δp acts on (mm²)."""
        return float(sum(m.rest_area.sum().item() for m in self.membranes.values()))

    def area_profile(self, x=None):
        return self.sections.profile(self.tube.x.cpu().numpy() if x is None else x)

    def area(self, x=None):
        return float(self.area_profile(x).min())

    def travel(self):
        """How far the part pressing on the tube moved towards it (mm): the first body bonded to a membrane,
        else the membranes' mean displacement."""
        for _, j, _ in self.bonds:
            body = self.movers[j]
            u = body.translation if getattr(body, "is_rigid", False) else (body.x - body.X).mean(dim=0)
            return float(u.cpu().numpy() @ self.squeeze)
        u = [(m.x - m.X).mean(dim=0).cpu().numpy() for m in self.membranes.values()]
        return float(np.mean(u, axis=0) @ self.squeeze)

    def set_contact_stiffness(self, pressure_ref):
        """Penalty stiffness so that contacts penetrate about 1 % of the tube's inside height at
        pressure_ref (MPa); the membrane bonds are 10x stiffer."""
        if self.auto_contact:
            self.contact_stiffness = 4.0 * max(abs(pressure_ref), 1.0 * KPA) / (0.02 * self.channel_height)
        k = self.contact_stiffness
        self.env.contact_stiffness = k
        for c in self.contacts:
            c.set_stiffness(k)
        for t in self.ties:
            t.set_stiffness(10.0 * k)
        if self.self_contact is not None:
            self.self_contact.stiffness = k

    # -----------------------------
    # Flow
    # -----------------------------

    def segment_geometry(self, x=None):
        """Per segment, in SI units: the smallest cross-section in it (A, P, h, w, Dh) and its length L."""
        x = self.tube.x.cpu().numpy() if x is None else x
        g = self.sections.geometry(x, self.squeeze)
        f = self.flow
        out = []
        for k in range(f.segments):
            at = np.nonzero(f.station_segment == k)[0]
            m = at[np.argmin(g["A"][at])] if len(at) else None
            A = g["A"][m] if m is not None else 0.0
            P = g["P"][m] if m is not None else 0.0
            out.append({"A": A * 1e-6, "P": P * 1e-3, "h": g["h"][m] * 1e-3 if m is not None else 0.0,
                        "w": g["w"][m] * 1e-3 if m is not None else 0.0,
                        "L": (f.bounds[k + 1] - f.bounds[k]) * 1e-3,
                        "Dh": 4 * A / P * 1e-3 if P > 0 else 0.0})
        return out

    def network(self, gas, geometry=None):
        """The flow network at the current shape: nodes 0..N along the tube, then one per other fluid."""
        f = self.flow
        net = FlowNetwork(gas)
        tube_nodes = [net.add_node(f"{f.names[f.lumen]} {k}") for k in range(f.segments + 1)]
        node = {}
        for j, name in f.names.items():
            if j != f.lumen:
                node[j] = net.add_node(name, f.constant[j] * 1000.0 if j in f.constant else None)
        outside = None
        for k, g in enumerate(geometry if geometry is not None else self.segment_geometry()):
            net.add_edge(tube_nodes[k], tube_nodes[k + 1], f.segment_law, f"segment {k + 1}", g)
        for conn, settings in f.connections:
            if settings["type"] == CLOSED:
                continue
            ends = []
            for j in (conn.a, conn.b):
                if j is None:
                    if outside is None:
                        outside = net.add_node(OUTSIDE, 0.0)
                    ends.append(outside)
                elif j == f.lumen:  # the tube end nearest to the connection
                    s = float(conn.center @ self.axis)
                    ends.append(tube_nodes[int(np.argmin(np.abs(f.bounds - s)))])
                else:
                    ends.append(node[j])
            if settings["type"] == OPENING:
                net.add_opening(*ends)
            else:
                geometry_c = {"A": conn.area * 1e-6, "P": conn.perimeter * 1e-3, "h": conn.height * 1e-3,
                              "w": conn.width * 1e-3, "L": 0.0,
                              "Dh": 4 * conn.area / conn.perimeter * 1e-3 if conn.perimeter > 0 else 0.0}
                net.add_edge(ends[0], ends[1], compile_flow_law(settings.get("law"), ORIFICE_LAW), conn.key,
                             geometry_c)
        return net, tube_nodes, node

    def solve_flow(self, gas, guess=None):
        """Pressures (kPa) along the tube (N+1 nodes) and of the other fluids, the mass flow through the tube
        (kg/s, positive along the axis) and the new wall pressures (kPa, one per wall FluidVolume)."""
        net, tube_nodes, node = self.network(gas)
        p, flows = net.solve(guess)
        p_kpa = p / 1000.0
        tube = p_kpa[tube_nodes]
        wall = []
        for _, (kind, j) in self.flow.wall:
            wall.append(0.5 * (tube[j] + tube[j + 1]) if kind == "segment" else p_kpa[node[j]])
        mdot = float(np.mean(flows[:self.flow.segments])) if self.flow.segments else 0.0
        return {"tube": tube, "fluids": {self.flow.names[j]: float(p_kpa[n]) for j, n in node.items()},
                "mdot": mdot, "wall": np.array(wall), "raw": p}

    def set_wall_pressures(self, kpa):
        for (volume, _), value in zip(self.flow.wall, kpa):
            volume.P0 = float(value) * KPA


def build_activation(cad, mesh: ActivationMesh, project) -> ActivationBuild:
    validate(project)
    parts, bodies = project.parts, cad.bodies
    warnings = []
    unassigned = [p.name for p in parts if p.role == UNASSIGNED]
    if unassigned:
        warnings.append(f"Unassigned parts are ignored: {', '.join(unassigned)}")
    c = _roles(project, CHANNEL)[0]
    membranes_idx = [i for i, p in enumerate(parts) if p.role in DEFORMABLE]
    free_idx = free_rigid(project)
    fixed_idx = [i for i in _roles(project, RIGID) if i not in free_idx]
    solid_idx = _roles(project, SOLID)
    fluid_idx = _roles(project, FLUID)

    env = ms.Environment()
    obstacles = {i: env.add_obstacle(mesh.surfaces[i].vertices, mesh.surfaces[i].faces, color=ROLE_COLORS[RIGID])
                 for i in fixed_idx}
    rigid_field = ObstacleField(list(obstacles.values())) if obstacles else None

    # --- which fluid every tube face touches (the solid's outward normals point into the fluids)
    vol = mesh.volumes[c]
    V, F = vol.vertices, vol.faces
    centroid = V[F].mean(axis=1)
    normal = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    target = _probe_targets(vol, [mesh.surfaces[j] for j in fluid_idx], 1e-3 * bodies[c].diagonal)
    face_fluid = np.where(target >= 0, np.asarray(fluid_idx)[np.clip(target, 0, None)], -1)
    dynamic = [j for j in fluid_idx if parts[j].props.get("model") == DYNAMIC_FLUID]
    lining = {j: int((face_fluid == j).sum()) for j in dynamic}
    if not lining or max(lining.values()) == 0:
        raise ValueError(f"No dynamic fluid fills {parts[c].name}: make the fluid inside the tube a Fluid body with "
                         f"dynamic pressure that touches the tube's inside.")
    lumen = max(lining, key=lining.get)
    wall_faces = np.nonzero(face_fluid == lumen)[0]

    # --- channel axis: the long direction of the fluid filling the tube, from its higher-pressure end
    lm = mesh.surfaces[lumen]
    pts = lm.vertices
    connections = [(conn, project.connection(conn.key, outside=conn.b is None))
                   for conn in detect_connections(mesh.surfaces, parts, bodies)]
    axis, center = channel_axis(lm, lumen, connections, parts)
    for j, q in enumerate(parts):
        if j != lumen and q.role == FLUID and q.props.get("outputs"):
            warnings.append(f"{q.name} is not the fluid inside the tube: its outputs are not used.")
    s = pts @ axis
    lo, hi = float(s.min()), float(s.max())
    squeeze = squeeze_direction_from(mesh, membranes_idx, center, axis)
    height = float(np.ptp(pts @ squeeze))
    if not (height > 0):
        height = float(np.sort(bodies[lumen].size)[0])

    # --- fixed nodes: tube end faces = planar CAD faces normal to the axis that are not channel wall
    wall = np.zeros(len(F), bool)
    wall[wall_faces] = True
    fixed = np.zeros(len(V), bool)
    for tag in np.unique(vol.face_surface):
        on = (vol.face_surface == tag) & ~wall
        if not on.any():
            continue
        n = normal[on]
        if np.all(np.abs(n @ axis) > 0.99) and np.all(n @ n.mean(axis=0) / np.linalg.norm(n.mean(axis=0)) > 0.999):
            fixed[np.unique(F[on])] = True
    if not fixed.any():
        warnings.append(f"{parts[c].name}: no end faces normal to the channel axis found; the tube is held only by "
                        f"contact.")
    tube = _solid(parts[c], vol, fixed, ROLE_COLORS[CHANNEL])
    env.membrane_list.append(tube)

    # --- moving parts: free rigid bodies and solids
    movers = {}
    for i in free_idx:
        m = mesh.surfaces[i]
        movers[i] = RigidBody(m.vertices, m.faces, color=FREE_RIGID_COLOR, name=parts[i].name)
    for i in solid_idx:
        vm = mesh.volumes[i]
        fixed_s = np.zeros(len(vm.vertices), bool)
        if parts[i].props.get("support") == SOLID_FIXED:
            if rigid_field is None:
                warnings.append(f"{parts[i].name}: no fixed rigid body to hold it; it is free.")
            else:
                surface = np.unique(vm.faces)
                sd, _ = rigid_field.signed_distance(torch.as_tensor(vm.vertices[surface]),
                                                    max_distance=0.1 * mesh.sizes.get(i, 1.0))
                fixed_s[surface[sd.numpy() <= 1e-3 * bodies[i].diagonal]] = True
                if not fixed_s.any():
                    warnings.append(f"{parts[i].name} does not touch a fixed rigid body; it is free.")
        movers[i] = _solid(parts[i], vm, fixed_s, ROLE_COLORS[SOLID])

    # --- membranes: Δp pushes them towards the tube
    membranes, thickness, tie_nodes, bonds, ties = {}, {}, {}, [], []
    load = env.add_fluid_volume(P0=0.0, name="Δp (membrane)", color="#e4572e")
    for i in membranes_idx:
        part, mid = parts[i], mesh.midsurfaces[i]
        t = float(part.props.get("thickness", 0.0) or 0.0) or mid.thickness
        thickness[i] = t
        boundary = ms.boundary_nodes(mid.faces, len(mid.vertices))
        fixed_m = boundary
        if part.props.get("fixed_edges") == TOUCHING_RIGID and rigid_field is not None:
            sd, _ = rigid_field.signed_distance(torch.as_tensor(mid.vertices), max_distance=t)
            fixed_m = boundary & (sd.numpy() <= 0.75 * t + 1e-6 * bodies[i].diagonal)
        shell = env.add_membrane(
            mid.vertices, mid.faces, thickness=t, youngs_modulus=float(part.props["youngs_modulus"]),
            poisson_ratio=float(part.props.get("poisson_ratio", 0.45)),
            material="neo_hookean" if part.props.get("material", NEO_HOOKEAN) == NEO_HOOKEAN else "svk",
            bending=part.role == SHELL, pretension=float(part.props.get("pretension", 0.0)), fixed=fixed_m,
            boundary_rotation="clamped" if part.props.get("edge_rotation", CLAMPED) == CLAMPED else "free",
            contact_offset=0.5 * t, color=ROLE_COLORS[part.role], name=part.name)
        membranes[i] = shell
        n_mean = np.cross(mid.vertices[mid.faces[:, 1]] - mid.vertices[mid.faces[:, 0]],
                          mid.vertices[mid.faces[:, 2]] - mid.vertices[mid.faces[:, 0]]).sum(axis=0)
        load.add_boundary(shell, +1 if n_mean @ (center - mid.vertices.mean(axis=0)) > 0 else -1)
        # bonded to every moving part it touches: nodes within about half a thickness of its surface
        reach = 0.6 * t + 1e-6 * bodies[i].diagonal
        taken = np.zeros(len(mid.vertices), bool)
        for j, body in movers.items():
            if getattr(body, "is_rigid", False):
                sd, _ = body.obstacle.signed_distance(torch.as_tensor(mid.vertices), max_distance=2 * t)
                nodes = np.nonzero((np.abs(sd.numpy()) <= reach) & ~fixed_m & ~taken)[0]
                tie = RigidTie(shell, nodes, body, 1.0) if len(nodes) else None
            else:
                candidates = np.nonzero(~fixed_m & ~taken)[0]
                tie = SurfaceTie(shell, candidates, body, 1.0, max_distance=reach) if len(candidates) else None
                nodes = tie.nodes.cpu().numpy() if tie is not None else []
            if tie is not None and len(nodes):
                ties.append(tie)
                taken[nodes] = True
                bonds.append((i, j, len(nodes)))
        if taken.any():
            tie_nodes[i] = np.nonzero(taken)[0]
    for j, body in movers.items():
        held = any(b[1] == j for b in bonds) or (not getattr(body, "is_rigid", False) and body.fixed.any())
        if not held:
            warnings.append(f"{parts[j].name} is not bonded to a membrane and not held: it can float away.")
    if not bonds:
        warnings.append("Nothing is bonded to the membrane: make the part between the membrane and the tube a free "
                        "rigid body or a solid that touches the membrane's face.")

    # --- the gas: segments of the tube, wall pressures, connections
    N = max(int(parts[lumen].props.get("segments", 10)), 1)
    bounds = np.linspace(lo, hi, N + 1)
    wall_s = centroid[wall_faces] @ axis
    wall_segment = np.clip(((wall_s - lo) / (hi - lo) * N).astype(int), 0, N - 1)
    walls = []
    for k in range(N):
        faces = wall_faces[wall_segment == k]
        if len(faces):
            v = env.add_fluid_volume(P0=0.0, name=f"{parts[lumen].name} segment {k + 1}", color=ROLE_COLORS[FLUID])
            v.add_patch(tube, faces, side=-1)
            walls.append((v, ("segment", k)))
    for j in fluid_idx:  # other fluids against the tube (e.g. the supply at its open end)
        faces = np.nonzero(face_fluid == j)[0]
        if j != lumen and len(faces) and not fixed[np.unique(F[faces])].all():
            v = env.add_fluid_volume(P0=0.0, name=parts[j].name, color=ROLE_COLORS[FLUID])
            v.add_patch(tube, faces, side=-1)
            walls.append((v, ("fluid", j)))
    constant = {j: float(parts[j].props.get("pressure", 0.0)) for j in fluid_idx
                if parts[j].props.get("model") == CONSTANT_FLUID}
    if not any(conn.a == lumen or conn.b == lumen for conn, st in connections if st["type"] != CLOSED):
        warnings.append(f"{parts[lumen].name} has no open connection: no gas flows through the tube.")
    n_stations = max(60, 6 * N)
    pad = 0.01 * (hi - lo)
    # sections cut the whole inside wall (every face against a fluid, except end faces): a section near the
    # junction of two fluids must not lose the triangles assigned to the neighbour
    inside = np.nonzero((face_fluid >= 0) & (np.abs(normal @ axis) < 0.9))[0]
    sections = ChannelSections(V, F[inside], axis, stations=np.linspace(lo + pad, hi - pad, n_stations))
    station_segment = np.clip(((sections.stations - lo) / (hi - lo) * N).astype(int), 0, N - 1)
    flow = FlowModel(lumen, N, bounds, station_segment,
                     compile_flow_law(parts[lumen].props.get("segment_law"), SEGMENT_LAW), connections, walls,
                     {j: parts[j].name for j in fluid_idx}, constant)

    # --- contact: deformable solids (tube and solid parts) against free rigid bodies and each other
    solids = {c: tube, **{j: b for j, b in movers.items() if not getattr(b, "is_rigid", False)}}
    contacts = []
    for j, body in movers.items():
        if getattr(body, "is_rigid", False):
            contacts += [MovingContact(s_, body, 1.0, max_reach=0.25 * height) for s_ in solids.values()]
    for a, A in solids.items():
        for b, B in solids.items():
            if a != b:
                contact = SolidContact(A, B, 1.0, reach=0.25 * height)
                if len(contact.nodes):
                    contacts.append(contact)
    # opposite walls inside the tube: the half facing the membrane against the other half
    rel = centroid[wall_faces] - center
    near_side = wall_faces[rel @ squeeze < 0]
    far_side = np.setdiff1d(wall_faces, near_side)
    self_contact = ShellContact([], 1.0, search_distance=0.1 * height, pairs=[
        (tube, np.unique(F[near_side]), tube, far_side, 0.02 * height),
        (tube, np.unique(F[far_side]), tube, near_side, 0.02 * height)]) or None
    env.couplings.extend(ties + contacts)
    if self_contact is not None:
        env.surface_contacts.append(self_contact)
    env.membrane_list.extend(movers.values())

    k = project.solver.contact_stiffness
    build = ActivationBuild(env, membranes, tube, movers, obstacles, load, flow, sections, axis, squeeze, height,
                            int(fixed.sum()), tie_nodes, bonds, wall_faces, ties, contacts, self_contact, thickness,
                            warnings, contact_stiffness=k, auto_contact=not k)
    build.shells = dict(membranes)
    build.shells[c] = tube
    build.shells.update(movers)
    return build


def squeeze_direction_from(mesh, membranes_idx, tube_center, axis):
    """Unit vector from the membranes (area-weighted centroid) towards the tube, normal to the channel axis."""
    X, w = [], []
    for i in membranes_idx:
        V, F = mesh.midsurfaces[i].vertices, mesh.midsurfaces[i].faces
        area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
        X.append(V[F].mean(axis=1))
        w.append(area)
    X, w = np.vstack(X), np.concatenate(w)
    d = tube_center - (w[:, None] * X).sum(axis=0) / w.sum()
    d -= (d @ axis) * axis
    return d / max(np.linalg.norm(d), 1e-300)


def _solid(part, vm, fixed, color):
    return Solid(vm.vertices, vm.tets, vm.faces, youngs_modulus=float(part.props.get("youngs_modulus", 0.5)),
                 poisson_ratio=float(part.props.get("poisson_ratio", 0.45)), fixed=fixed, color=color, name=part.name)


def _centroid(surface):
    """Volume centroid of a closed outward surface mesh."""
    V, F = surface.vertices, surface.faces
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    return ((a + b + c) / 4.0 * vol[:, None]).sum(axis=0) / vol.sum()


# -----------------------------
# The Δp sweep
# -----------------------------

def gas_of(study):
    return Gas(study.gas_constant, study.temperature, study.atmospheric_pressure * 1000.0, study.viscosity)


def run_study(build: ActivationBuild, project, callback=None, point_callback=None, check=None):
    """
    Sweep the membrane pressure difference over the study range. At every Δp the structure and the flow
    are iterated: solve the structure with the current wall pressures, measure the segments, solve the
    flow network, apply its pressures (under-relaxed when the iteration oscillates), until no wall pressure
    changes by more than the tolerance. Returns (results dict stored in the design file, per-point states).
    """
    study, solver_settings = project.study, project.solver
    gas = gas_of(study)
    dps = np.linspace(study.dp_min, study.dp_max, max(int(study.points), 2))
    p_ref = max([abs(study.dp_min), abs(study.dp_max)] + [abs(v) for v in build.flow.constant.values()])
    build.set_contact_stiffness(p_ref * KPA)

    env = build.env
    env.reset()
    flow = build.solve_flow(gas)
    wall = flow["wall"]
    warm = False
    rows, history_states, frames = [], [], []
    for k, dp in enumerate(dps):
        if check:
            check()
        build.load.P0 = float(dp) * KPA
        omega, last_change, iterations, converged, result = 1.0, np.inf, 0, False, None
        while iterations < study.max_coupling_iterations:
            if check:
                check()
            build.set_wall_pressures(wall)
            saved = [s.get_state().clone() for s in env.membrane_list]
            saved_P = [(v.solved_P0, v.start_P0) for v in env.fluid_volume_list]
            result = env.solve(load_steps=(2 if iterations == 0 else 1) if warm else solver_settings.load_steps,
                               max_iterations=solver_settings.max_iterations, rtol=solver_settings.tolerance,
                               callback=(lambda lam, it, r: callback(k, len(dps), dp, lam, it, r)) if callback else None,
                               warm_start=warm, min_load_increment=1.0 / 64)
            iterations += 1
            if not result.converged:
                for s, u in zip(env.membrane_list, saved):
                    s.set_state(u)
                for v, (solved, start) in zip(env.fluid_volume_list, saved_P):
                    v.solved_P0, v.start_P0 = solved, start
                break
            warm = True
            env.history = env.history[-1:]  # only the converged state is needed (keeps memory flat)
            flow = build.solve_flow(gas, guess=flow["raw"])
            change = float(np.abs(flow["wall"] - wall).max()) if len(wall) else 0.0
            if change <= study.coupling_tolerance:
                converged = True
                break
            if change > last_change:  # oscillating between open and shut: relax harder
                omega = max(0.5 * omega, 0.05)
            last_change = change
            wall = wall + omega * (flow["wall"] - wall)
        profile = build.area_profile()
        row = {"dp": float(dp), "area": build.area(), "mdot": flow["mdot"], "p_end": float(flow["tube"][-1]),
               "volume": float(build.load.delta_volume),
               "pressures": flow["tube"].tolist(), "fluids": flow["fluids"], "travel": build.travel(),
               "iterations": iterations, "converged": bool(converged), "profile": profile.tolist(),
               "message": result.message if result is not None else ""}
        rows.append(row)
        frames.append([body_display_coords(b) for b in build.shells.values()])
        history_states.append({
            "dp": float(dp), "load_factor": 1.0, "converged": bool(converged),
            "shell_coords": [s.x.detach().cpu().clone() for s in build.shells.values()],
            "area": row["area"], "profile": profile, "pressures": flow["tube"]})
        if point_callback:
            point_callback(row)
        if not converged:
            warm = False
            env.reset()
            flow = build.solve_flow(gas)
            wall = flow["wall"]
    f = build.flow
    return {
        "dp": [r["dp"] for r in rows], "area": [r["area"] for r in rows], "mdot": [r["mdot"] for r in rows],
        "p_end": [r["p_end"] for r in rows], "pressures": [r["pressures"] for r in rows],
        "volume": [r["volume"] for r in rows], "membrane_area": build.membrane_area,
        "fem": encode_fem(build, project, frames),
        "fluids": [r["fluids"] for r in rows], "travel": [r["travel"] for r in rows],
        "converged": [r["converged"] for r in rows], "iterations": [r["iterations"] for r in rows],
        "profiles": [r["profile"] for r in rows], "stations": (build.sections.stations - f.bounds[0]).tolist(),
        "node_positions": f.node_positions().tolist(), "rest_profile": build.sections.rest_areas.tolist(),
        "A0": build.A0, "segments": f.segments, "segment_law": f.segment_law.text,
        "lumen": f.names[f.lumen], "axis": build.axis.tolist(), "bounds": f.bounds.tolist(),
        "connections": {conn.key: dict(st) for conn, st in f.connections}, "gas": asdict(study),
        "units": {"dp": "kPa", "area": "mm^2", "mdot": "kg/s", "p_end": "kPa", "pressures": "kPa", "travel": "mm",
                  "stations": "mm", "volume": "mm^3", "membrane_area": "mm^2"},
    }, history_states
