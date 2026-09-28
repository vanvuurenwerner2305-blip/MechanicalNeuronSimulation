"""
Project model: which CAD body plays which role, with which properties, plus solver settings.
Saved as JSON (*.mns). Units: mm, N, MPa; pressures are entered in kPa.
"""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

UNASSIGNED = "Unassigned"
MEMBRANE = "Membrane"
SHELL = "Shell"
RIGID = "Rigid body"
CHAMBER = "Fluid chamber"
IGNORE = "Ignore"
ROLES = [UNASSIGNED, MEMBRANE, SHELL, RIGID, CHAMBER, IGNORE]
DEFORMABLE = (MEMBRANE, SHELL)

ROLE_COLORS = {
    UNASSIGNED: "#c8c8c8",
    MEMBRANE: "#e4572e",
    SHELL: "#f3a712",
    RIGID: "#6c7a89",
    CHAMBER: "#29a0d6",
    IGNORE: "#eeeeee",
}
ROLE_OPACITY = {UNASSIGNED: 1.0, MEMBRANE: 1.0, SHELL: 1.0, RIGID: 1.0, CHAMBER: 0.25, IGNORE: 0.1}

ROLE_HELP = {
    UNASSIGNED: "Not used in the simulation until a role is assigned.",
    MEMBRANE: "Thin deformable sheet without bending stiffness. Simulated on its mid-surface.",
    SHELL: "Thin deformable sheet with bending stiffness. Simulated on its mid-surface.",
    RIGID: "Fixed, undeformable part. Membranes and shells cannot pass through it.",
    CHAMBER: "Fluid/gas region. Its pressure acts on every membrane or shell it touches.",
    IGNORE: "Excluded from the simulation.",
}

# Chamber models
CONSTANT = "Constant pressure (input)"
IDEAL_GAS = "Closed: ideal gas (isothermal)"
LINEAR = "Closed: linear stiffness"
CHAMBER_MODELS = [CONSTANT, IDEAL_GAS, LINEAR]

NEO_HOOKEAN = "Neo-Hookean (incompressible rubber)"
SVK = "St. Venant-Kirchhoff"
ALL_EDGES = "All boundary edges"
TOUCHING_RIGID = "Edges touching rigid bodies"
CLAMPED = "Clamped"
PINNED = "Pinned"


@dataclass
class Field:
    key: str
    label: str
    kind: str                 # "float", "int", "choice"
    default: object
    unit: str = ""
    minimum: float = 0.0
    maximum: float = 1e12
    decimals: int = 4
    choices: tuple = ()
    tooltip: str = ""
    visible_if: tuple = None  # (key, allowed values)
    mesh: bool = False        # changing it invalidates the mesh


_DEFORMABLE_FIELDS = [
    Field("material", "Material model", "choice", NEO_HOOKEAN, choices=(NEO_HOOKEAN, SVK)),
    Field("youngs_modulus", "Young's modulus", "float", 0.1, "MPa", 1e-9, 1e6, 5,
          tooltip="Silicone rubbers are typically 0.05-1 MPa."),
    Field("poisson_ratio", "Poisson's ratio", "float", 0.45, "", 0.0, 0.499, 3,
          visible_if=("material", (SVK,)), tooltip="The neo-Hookean model is incompressible (0.5)."),
    Field("thickness", "Thickness", "float", 0.0, "mm", 0.0, 1e6, 4,
          tooltip="0 = measured from the CAD solid.", mesh=True),
    Field("mesh_size", "Element size", "float", 0.0, "mm", 0.0, 1e6, 3,
          tooltip="0 = automatic (1/15 of the smaller in-plane dimension).", mesh=True),
    Field("fixed_edges", "Fixed edges", "choice", ALL_EDGES, choices=(ALL_EDGES, TOUCHING_RIGID)),
    Field("edge_rotation", "Edge support", "choice", CLAMPED, choices=(CLAMPED, PINNED),
          tooltip="Clamped edges keep their slope, pinned edges can rotate. Only affects shells.",
          visible_if=("__role__", (SHELL,))),
]

ROLE_FIELDS = {
    UNASSIGNED: [],
    IGNORE: [],
    MEMBRANE: _DEFORMABLE_FIELDS + [
        Field("pretension", "Pre-tension", "float", 0.0, "N/mm", 0.0, 1e6, 5,
              tooltip="Isotropic in-plane tension present before any pressure is applied."),
    ],
    SHELL: [Field(f.key, f.label, f.kind, 1.0 if f.key == "youngs_modulus" else f.default, f.unit, f.minimum,
                  f.maximum, f.decimals, f.choices, f.tooltip, f.visible_if, f.mesh) for f in _DEFORMABLE_FIELDS],
    RIGID: [
        Field("mesh_size", "Element size", "float", 0.0, "mm", 0.0, 1e6, 3,
              tooltip="0 = automatic. Only curved surfaces need a fine mesh.", mesh=True),
    ],
    CHAMBER: [
        Field("model", "Pressure model", "choice", CONSTANT, choices=tuple(CHAMBER_MODELS)),
        Field("pressure", "Pressure (gauge)", "float", 0.0, "kPa", -1e6, 1e6, 4,
              tooltip="Gauge pressure: 0 = atmospheric. For a constant-pressure chamber the applied "
                      "pressure; for a closed chamber the pressure at the moment it was sealed."),
        Field("incompressible", "Incompressible fluid", "float", 0.0, "%", 0.0, 99.0, 3,
              visible_if=("model", (IDEAL_GAS,)),
              tooltip="Share of the chamber's initial volume filled with incompressible liquid; the rest "
                      "is gas. All volume change goes into the gas, so more liquid makes the chamber "
                      "stiffer. For a chamber completely full of liquid use 'Closed: linear stiffness'."),
        Field("stiffness", "Volume stiffness", "float", 1e-3, "kPa/mm³", 0.0, 1e9, 6,
              visible_if=("model", (LINEAR,)), tooltip="dP/dV of the closed chamber."),
    ],
}


def default_props(role: str) -> dict:
    return {f.key: f.default for f in ROLE_FIELDS[role]}


@dataclass
class PartSettings:
    name: str
    role: str = UNASSIGNED
    props: dict = field(default_factory=dict)
    visible: bool = True

    def set_role(self, role: str):
        if role != self.role:
            old = self.props
            self.role = role
            self.props = default_props(role)
            self.props.update({k: v for k, v in old.items() if k in self.props})


@dataclass
class SolverSettings:
    load_steps: int = 10
    max_iterations: int = 40
    tolerance: float = 1e-8
    contact_stiffness: float = 0.0   # MPa/mm, 0 = automatic


SOLVER_FIELDS = [
    Field("load_steps", "Load steps", "int", 10, "", 1, 1000,
          tooltip="Pressures are ramped from 0 to full in this many steps (cut automatically if needed)."),
    Field("max_iterations", "Max Newton iterations", "int", 40, "", 1, 1000),
    Field("tolerance", "Residual tolerance", "float", 1e-8, "", 1e-14, 1e-2, 12),
    Field("contact_stiffness", "Contact stiffness", "float", 0.0, "MPa/mm", 0.0, 1e12, 5,
          tooltip="Penalty stiffness. 0 = automatic (penetration ≈ 1% of the thinnest part at the highest pressure)."),
]


class Project:
    def __init__(self, step_path: str = None, part_names=()):
        self.step_path = step_path
        self.parts = [PartSettings(name) for name in part_names]
        self.solver = SolverSettings()
        self.path = None

    def auto_assign_from_names(self, only_unassigned: bool = True):
        """Guess roles from part names (membrane, shell, chamber/fluid/cavity, ...)."""
        rules = [(("membrane", "diaphragm", "skin"), MEMBRANE), (("shell",), SHELL),
                 (("chamber", "fluid", "cavity", "air", "gas", "volume"), CHAMBER),
                 (("ignore",), IGNORE)]
        changed = 0
        for part in self.parts:
            if only_unassigned and part.role != UNASSIGNED:
                continue
            name = part.name.lower()
            role = next((r for keys, r in rules if any(k in name for k in keys)), RIGID)
            if role != part.role:
                part.set_role(role)
                changed += 1
        return changed

    # -----------------------------
    # Persistence
    # -----------------------------

    def to_dict(self):
        return {"version": 1, "step_path": self.step_path,
                "parts": [asdict(p) for p in self.parts], "solver": asdict(self.solver)}

    def save(self, path):
        path = Path(path)
        data = self.to_dict()
        if self.step_path:
            try:
                data["step_path"] = str(Path(self.step_path).resolve().relative_to(path.resolve().parent))
            except ValueError:
                data["step_path"] = str(Path(self.step_path).resolve())
        path.write_text(json.dumps(data, indent=2))
        self.path = str(path)

    @classmethod
    def load(cls, path):
        path = Path(path)
        data = json.loads(path.read_text())
        project = cls()
        step = Path(data["step_path"])
        project.step_path = str(step if step.is_absolute() else (path.parent / step).resolve())
        project.parts = [PartSettings(**p) for p in data["parts"]]
        project.solver = SolverSettings(**data.get("solver", {}))
        project.path = str(path)
        return project

    def match_bodies(self, bodies):
        """Re-attach saved part settings to freshly imported bodies (by name, then by position)."""
        by_name = {p.name: p for p in self.parts}
        matched = []
        for i, body in enumerate(bodies):
            part = by_name.get(body.name) or (self.parts[i] if i < len(self.parts) else None)
            matched.append(PartSettings(body.name, part.role, dict(part.props), part.visible)
                           if part else PartSettings(body.name))
        self.parts = matched
