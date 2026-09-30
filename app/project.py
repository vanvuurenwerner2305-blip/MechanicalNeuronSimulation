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
# A membrane replaced by a pre-simulated activation-function design (its Δp -> swept volume curve)
ACTIVATION_MEMBRANE = "Activation membrane"
ROLES = [UNASSIGNED, MEMBRANE, SHELL, ACTIVATION_MEMBRANE, RIGID, CHAMBER, IGNORE]
DEFORMABLE = (MEMBRANE, SHELL)
SHEETS = DEFORMABLE + (ACTIVATION_MEMBRANE,)  # thin parts meshed on their mid-surface
AUTOMATIC = "Automatic"

# Activation-function space: a membrane squeezes a soft tube (the channel), through whatever is bonded
# to it (a free rigid body or a deformable solid)
CHANNEL = "Channel (tube)"
FLUID = "Fluid"
SOLID = "Solid"
ACTIVATION_ROLES = [UNASSIGNED, MEMBRANE, SHELL, CHANNEL, FLUID, SOLID, RIGID, IGNORE]
CONSTANT_FLUID, DYNAMIC_FLUID = "Constant pressure", "Dynamic pressure"
# flow connections between fluids (and from a dynamic fluid to the outside)
OPENING, ORIFICE, CLOSED = "Opening (no resistance)", "Orifice", "Closed"
CONNECTION_TYPES = (OPENING, ORIFICE, CLOSED)
LEGACY_ROLES = {"Channel input side": "Fluid", "Channel output side": "Fluid"}
FIXED, FREE = "Fixed", "Free (moves as a rigid body)"
SOLID_FREE, SOLID_FIXED = "Free (held by what it is bonded to)", "Fixed where it touches fixed rigid bodies"
NEURON_SPACE, ACTIVATION_SPACE = "neuron", "activation"
SPACE_ROLES = {NEURON_SPACE: ROLES, ACTIVATION_SPACE: ACTIVATION_ROLES}

ROLE_COLORS = {
    UNASSIGNED: "#c8c8c8",
    MEMBRANE: "#e4572e",
    SHELL: "#f3a712",
    ACTIVATION_MEMBRANE: "#d6336c",
    RIGID: "#6c7a89",
    CHAMBER: "#29a0d6",
    IGNORE: "#eeeeee",
    CHANNEL: "#c77dff",
    FLUID: "#1f77b4",
    SOLID: "#b5838d",
}
FREE_RIGID_COLOR = "#8d6e63"
ROLE_OPACITY = {UNASSIGNED: 1.0, MEMBRANE: 1.0, SHELL: 1.0, ACTIVATION_MEMBRANE: 1.0, RIGID: 1.0, CHAMBER: 0.25, IGNORE: 0.1,
                CHANNEL: 1.0, FLUID: 0.35, SOLID: 1.0}

ROLE_HELP = {
    UNASSIGNED: "Not used in the simulation until a role is assigned.",
    MEMBRANE: "Thin deformable sheet without bending stiffness. Simulated on its mid-surface. "
              "A face that touches no chamber sees the surroundings (0 kPa).",
    SHELL: "Thin deformable sheet with bending stiffness. Simulated on its mid-surface. "
           "A face that touches no chamber sees the surroundings (0 kPa).",
    ACTIVATION_MEMBRANE: "The membrane of an activation-function design (*.mad) simulated in the Pre-activation → "
                         "activation space. It is not simulated again: the design's pre-simulated response (the "
                         "volume the membrane sweeps at a pressure difference) replaces it. The driving chamber (the "
                         "pre-activation chamber) pushes it towards the tube; the other side is the tube side. The "
                         "design's outputs - the activation (the pressure of its chosen tube segment), tube area and "
                         "mass flow - are read from it at the solved pre-activation pressure difference.",
    RIGID: "Undeformable part. Membranes and shells cannot pass through it. In the activation-function space it "
           "can also be free: it then moves and tilts as a rigid body, bonded to the membrane that touches it "
           "(e.g. a pusher), and presses on the tube and solids by contact.",
    CHAMBER: "Fluid/gas region. Its pressure acts on every membrane or shell it touches. "
             "Pressures are gauge: the surroundings are 0 kPa.",
    IGNORE: "Excluded from the simulation.",
    CHANNEL: "The soft tube that is squeezed shut. Simulated as a 3D solid (tetrahedra), held fixed at its "
             "two end faces. A dynamic fluid body fills its inside.",
    FLUID: "A body of gas. Constant pressure: a supply or sink held at a set pressure (e.g. the inlet, or 0 kPa at "
           "the far end). Dynamic pressure: its pressure follows from the flow; inside the tube it is split into "
           "segments, each a flow resistance. Where two fluids touch there is a flow connection (Flow tab).",
    SOLID: "Deformable 3D solid (tetrahedra), e.g. a soft pusher. A membrane touching it is bonded to it; it "
           "touches the tube and other solids by contact. Free by default, or fixed where it touches fixed rigid bodies.",
}

# Chamber models
CONSTANT = "Constant pressure (input)"
IDEAL_GAS = "Closed: ideal gas (isothermal)"
INCOMPRESSIBLE = "Closed: incompressible"
VENT = "Vent (open to surroundings, 0 kPa)"
CHAMBER_MODELS = [CONSTANT, IDEAL_GAS, INCOMPRESSIBLE, VENT]
LEGACY_MODELS = {"Closed: linear stiffness": INCOMPRESSIBLE}
INCOMPRESSIBLE_STIFFNESS = 10.0  # kPa per % volume change (default for "incompressible")
# Water is 22 000 kPa/% (bulk modulus 2.2 GPa); stiffer values make the solve slower.

CHAMBER_COLORS = {CONSTANT: "#2ca02c", INCOMPRESSIBLE: "#7b2cbf", VENT: "#e8e8e8"}
GAS_LIGHT, GAS_DARK = (0.66, 0.85, 0.97), (0.03, 0.19, 0.42)  # 0% and 99% incompressible


def _hex(rgb):
    return "#" + "".join(f"{int(round(255 * c)):02x}" for c in rgb)


def part_color(part) -> str:
    """Display colour: by role, and for chambers by pressure model (gas: darker = more liquid)."""
    if part.role == RIGID and part.props.get("motion") == FREE:
        return FREE_RIGID_COLOR
    if part.role == FLUID:
        return CHAMBER_COLORS[CONSTANT] if part.props.get("model") == CONSTANT_FLUID else ROLE_COLORS[FLUID]
    if part.role != CHAMBER:
        return ROLE_COLORS[part.role]
    model = part.props.get("model", CONSTANT)
    if model == IDEAL_GAS:
        f = min(max(float(part.props.get("incompressible", 0.0)), 0.0), 99.0) / 99.0
        return _hex([a + f * (b - a) for a, b in zip(GAS_LIGHT, GAS_DARK)])
    return CHAMBER_COLORS.get(model, CHAMBER_COLORS[CONSTANT])


def part_opacity(part) -> float:
    if part.role == CHAMBER:
        model = part.props.get("model", CONSTANT)
        if model == VENT:
            return 0.04
        if model == IDEAL_GAS:
            return 0.25 + 0.35 * min(float(part.props.get("incompressible", 0.0)), 99.0) / 99.0
        return 0.3
    return ROLE_OPACITY[part.role]


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
    Field("youngs_modulus", "Young's modulus", "float", 0.5, "MPa", 1e-9, 1e6, 5,
          tooltip="Silicone rubbers are typically 0.05-1 MPa."),
    Field("poisson_ratio", "Poisson's ratio", "float", 0.45, "", 0.0, 0.499, 3,
          visible_if=("material", (SVK,)), tooltip="The neo-Hookean model is incompressible (0.5)."),
    Field("thickness", "Thickness", "float", 0.0, "mm", 0.0, 1e6, 4,
          tooltip="Measured from the CAD solid when the role is assigned. Type a value to override it.",
          mesh=True),
    Field("elements_per_side", "Elements per shortest side", "int", 10, "", 1, 1000,
          tooltip="Mesh density: number of elements along the shorter in-plane side of the sheet.",
          mesh=True),
    Field("fixed_edges", "Fixed edges", "choice", ALL_EDGES, choices=(ALL_EDGES, TOUCHING_RIGID)),
    Field("edge_rotation", "Edge support", "choice", CLAMPED, choices=(CLAMPED, PINNED),
          tooltip="Clamped edges keep their slope, pinned edges can rotate. Only affects shells.",
          visible_if=("__role__", (SHELL,))),
]

ROLE_FIELDS = {
    ACTIVATION_MEMBRANE: [
        Field("design", "Activation design", "file", "", tooltip="The activation-function design (*.mad) whose "
              "membrane this part is. Save it from the Pre-activation → activation space after running its study.",
              choices=("Activation design (*.mad)",)),
        Field("driving", "Driving chamber", "choice", AUTOMATIC, choices="__chambers__",
              tooltip="The chamber whose pressure pushes the membrane towards the tube (positive Δp in the "
                      "design). Δp = its pressure - the pressure on the other side (a chamber there, or 0 kPa).\n"
                      "Automatic: the closed chamber it touches (the pre-activation chamber), else the first one."),
        Field("thickness", "Thickness", "float", 0.0, "mm", 0.0, 1e6, 4, mesh=True,
              tooltip="Only used to find the chambers on either side of the part."),
    ],
    UNASSIGNED: [],
    IGNORE: [],
    MEMBRANE: _DEFORMABLE_FIELDS + [
        Field("pretension", "Pre-tension", "float", 0.0, "N/mm", 0.0, 1e6, 5,
              tooltip="Isotropic in-plane tension present before any pressure is applied."),
    ],
    SHELL: [Field(f.key, f.label, f.kind, 1.0 if f.key == "youngs_modulus" else f.default, f.unit, f.minimum,
                  f.maximum, f.decimals, f.choices, f.tooltip, f.visible_if, f.mesh) for f in _DEFORMABLE_FIELDS],
    RIGID: [
        Field("elements_per_side", "Elements per shortest side", "int", 10, "", 1, 1000,
              tooltip="Mesh density along the part's shortest side. Only curved surfaces need a fine mesh.",
              mesh=True),
    ],
    CHAMBER: [
        Field("model", "Pressure model", "choice", CONSTANT, choices=tuple(CHAMBER_MODELS),
              tooltip="Constant pressure: an input held at a set pressure (green).\n"
                      "Ideal gas: sealed air, optionally partly filled with liquid (blue, darker = more liquid).\n"
                      "Incompressible: sealed and filled with liquid (purple), completely or partly.\n"
                      "Vent: open to the surroundings, always 0 kPa (transparent)."),
        Field("pressure", "Pressure (gauge)", "float", 0.0, "kPa", -1e6, 1e6, 4,
              visible_if=("model", (CONSTANT, IDEAL_GAS, INCOMPRESSIBLE)),
              tooltip="Gauge pressure: 0 = surroundings (atmospheric). For a constant-pressure chamber the "
                      "applied pressure; for a closed chamber the pressure at the moment it was sealed."),
        Field("ghost_volume", "Ghost volume", "float", 0.0, "mm³", 0.0, 1e15, 4,
              visible_if=("model", (IDEAL_GAS,)),
              tooltip="Extra volume connected to the chamber but not modelled in CAD (tubing, reservoir). "
                      "It is added to the body's volume; the incompressible share applies to the total."),
        Field("incompressible", "Incompressible fluid", "float", 0.0, "%", 0.0, 99.0, 3,
              visible_if=("model", (IDEAL_GAS,)),
              tooltip="Share of the chamber's total initial volume (body + ghost volume) filled with "
                      "incompressible liquid; the rest is gas. All volume change goes into the gas, so more "
                      "liquid makes the chamber stiffer. For a chamber completely full of liquid use "
                      "'Closed: incompressible'."),
        Field("stiffness", "Stiffness", "float", INCOMPRESSIBLE_STIFFNESS, "kPa per % ΔV", 1e-9, 1e12, 6,
              visible_if=("model", (INCOMPRESSIBLE,)),
              tooltip="Pressure rise per percent of volume decrease (default 10 kPa/%). Water is "
                      "22 000 kPa/%; around 1 000 kPa/% the volume change is already ~0.01 %, and stiffer "
                      "values mainly make the solve slower."),
        Field("fluid_volume", "Fluid volume", "float", 0.0, "mm³", 0.0, 1e15, 4,
              visible_if=("model", (INCOMPRESSIBLE,)),
              tooltip="Volume of liquid sealed in the chamber; defaults to the body's volume (exactly "
                      "full). Less liquid than the chamber gives negative pressure (suction) that pulls the "
                      "walls in towards the fluid volume; more liquid pressurises it. The stiffness is per % "
                      "of the fluid volume. 0 = the body's volume."),
    ],
}


from membrane_sim.flow import LAW_VARIABLES, ORIFICE_LAW, SEGMENT_LAW  # noqa: E402

FLOW_LAW_HELP = ("Pressure drop in Pa for a positive mass flow, SI units:\n"
                 "  mdot  mass flow (kg/s)          rho  gas density (kg/m³, ideal gas at the local pressure)\n"
                 "  mu    viscosity (Pa s)           p    mean absolute pressure (Pa)\n"
                 "  p_up, p_down  absolute pressures either side (Pa)   rho_up  density upstream (kg/m³)\n"
                 "  A     cross-sectional area (m²)  P    wetted perimeter (m)   Dh = 4A/P (m)\n"
                 "  h, w  height and width of the section (m)   L  length of the segment (m)\n"
                 "Numpy functions (sqrt, exp, ...) are allowed. A segment uses the smallest section in it.\n"
                 "A connection's A, P, h, w are those of the contact face between the two bodies; for a smaller\n"
                 "orifice type its area as a number, e.g. (mdot/(0.61*0.05e-6))**2/(2*rho_up) for 0.05 mm².\n"
                 "Laminar (default): 32*mu*L*mdot/(rho*A*Dh**2)    Orifice: (mdot/(0.61*A))**2/(2*rho_up)")

ROLE_FIELDS.update({
    CHANNEL: [
        Field("youngs_modulus", "Young's modulus", "float", 0.5, "MPa", 1e-9, 1e6, 5),
        Field("poisson_ratio", "Poisson's ratio", "float", 0.45, "", 0.0, 0.499, 3,
              tooltip="Compressible neo-Hookean solid. Silicone is nearly incompressible (0.45-0.49); values "
                      "close to 0.5 make the elements too stiff (locking)."),
        Field("element_order", "Elements", "choice", "Quadratic (10-node)",
              choices=("Quadratic (10-node)", "Linear (4-node)"), mesh=True,
              tooltip="Quadratic tetrahedra bend correctly with one or two elements through a wall; linear "
                      "ones are far too stiff in bending unless the mesh is very fine."),
        Field("elements_per_side", "Elements per shortest side", "int", 6, "", 1, 1000, mesh=True,
              tooltip="Mesh density: number of elements along the tube's shortest outside dimension."),
    ],
    FLUID: [
        Field("model", "Pressure", "choice", DYNAMIC_FLUID, choices=(CONSTANT_FLUID, DYNAMIC_FLUID),
              tooltip="Constant pressure: held at the pressure you set (a supply, or 0 kPa for the far end).\n"
                      "Dynamic pressure: follows from the flow through the connections and segments."),
        Field("pressure", "Pressure (gauge)", "float", 0.0, "kPa", -1e6, 1e6, 4,
              visible_if=("model", (CONSTANT_FLUID,))),
        Field("segments", "Segments", "int", 10, "", 1, 1000, visible_if=("model", (DYNAMIC_FLUID,)),
              tooltip="Inside a tube: the number of slices along the tube, each a flow resistance in series with "
                      "its own pressure on the tube wall. Elsewhere a dynamic fluid is one pressure."),
        Field("segment_law", "Segment resistance Δp", "text", SEGMENT_LAW, "Pa",
              visible_if=("model", (DYNAMIC_FLUID,)), tooltip=FLOW_LAW_HELP),
        Field("outputs", "Outputs", "outputs", (), visible_if=("model", (DYNAMIC_FLUID,)),
              tooltip="Named outputs (activations) of the device: the gas pressure of a segment of the fluid inside "
                      "the tube. Choose a segment (it is highlighted in the 3D view), name it and add it; several "
                      "outputs are allowed. Changing them needs no new simulation. Without any, the downstream "
                      "segment is the one output 'activation'."),
    ],
    SOLID: [
        Field("youngs_modulus", "Young's modulus", "float", 0.5, "MPa", 1e-9, 1e6, 5),
        Field("poisson_ratio", "Poisson's ratio", "float", 0.45, "", 0.0, 0.499, 3,
              tooltip="Compressible neo-Hookean solid; values close to 0.5 make the elements too stiff."),
        Field("support", "Support", "choice", SOLID_FREE, choices=(SOLID_FREE, SOLID_FIXED),
              tooltip="Free: held only by the membrane bonded to it (and contact).\n"
                      "Fixed: its nodes that touch a fixed rigid body are held in place."),
        Field("element_order", "Elements", "choice", "Quadratic (10-node)",
              choices=("Quadratic (10-node)", "Linear (4-node)"), mesh=True),
        Field("elements_per_side", "Elements per shortest side", "int", 4, "", 1, 1000, mesh=True),
    ],
})

# The rigid body of the activation-function space can also move
ACTIVATION_ROLE_FIELDS = dict(ROLE_FIELDS)
ACTIVATION_ROLE_FIELDS[RIGID] = [
    Field("motion", "Motion", "choice", FIXED, choices=(FIXED, FREE),
          tooltip="Fixed: held in place.\nFree: moves and tilts as a rigid body (6 degrees of freedom), bonded to "
                  "the membrane that touches it, e.g. a pusher."),
] + ROLE_FIELDS[RIGID]
SPACE_ROLE_FIELDS = {NEURON_SPACE: ROLE_FIELDS, ACTIVATION_SPACE: ACTIVATION_ROLE_FIELDS}


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
            old, same_kind = self.props, self.role in DEFORMABLE and role in DEFORMABLE
            self.role = role
            self.props = default_props(role)
            # keep shared settings (material, thickness...); the mesh density only between membrane and
            # shell: a rigid body's or tube's density would give a solid part a far too fine mesh
            self.props.update({k: v for k, v in old.items()
                               if k in self.props and (same_kind or k != "elements_per_side")})


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


def _relative(target, project_file) -> str:
    """target relative to the project file's folder when it is below it, else absolute."""
    try:
        return str(Path(target).resolve().relative_to(Path(project_file).resolve().parent))
    except ValueError:
        return str(Path(target).resolve())


class Project:
    def __init__(self, step_path: str = None, part_names=()):
        self.step_path = step_path
        self.parts = [PartSettings(name) for name in part_names]
        self.solver = SolverSettings()
        self.path = None

    space = NEURON_SPACE
    suffix = ".mns"

    def auto_assign_from_names(self, only_unassigned: bool = True):
        """Guess roles from part names (membrane, shell, chamber/fluid/cavity, ...)."""
        rules = [(("membrane", "diaphragm", "skin"), MEMBRANE), (("shell",), SHELL),
                 (("chamber", "fluid", "cavity", "air", "gas", "volume"), CHAMBER),
                 (("ignore",), IGNORE)]
        movers = ()  # names of parts that become free rigid bodies
        if self.space == ACTIVATION_SPACE:
            rules = [(("membrane", "diaphragm", "skin"), MEMBRANE), (("shell",), SHELL),
                     (("fluid", "cavity", "inlet", "input", "outlet", "output", "source", "supply", "sink",
                       "ambient", "outside", "air", "gas"), FLUID),
                     (("tube", "channel", "hose"), CHANNEL),
                     (("pusher", "squisher", "plunger", "piston"), RIGID),
                     (("solid", "pad"), SOLID), (("ignore",), IGNORE)]
            movers = ("pusher", "squisher", "plunger", "piston")
        changed = 0
        for part in self.parts:
            if only_unassigned and part.role != UNASSIGNED:
                continue
            name = part.name.lower()
            role = next((r for keys, r in rules if any(k in name for k in keys)), RIGID)
            if role != part.role:
                self.set_role(part, role)
                changed += 1
            if self.space == ACTIVATION_SPACE and role == RIGID and any(k in name for k in movers):
                part.props["motion"] = FREE
            if role == FLUID:  # supplies and sinks hold their pressure; the fluid inside the tube is dynamic
                sources = ("inlet", "input", "source", "supply", "outlet", "sink", "ambient", "outside")
                part.props["model"] = CONSTANT_FLUID if any(k in name for k in sources) else DYNAMIC_FLUID
        return changed

    @property
    def role_fields(self):
        return SPACE_ROLE_FIELDS[self.space]

    def set_role(self, part, role):
        """Assign a role, with the defaults of this space's property fields."""
        part.set_role(role)
        for f in self.role_fields[role]:
            part.props.setdefault(f.key, f.default)

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
            data["step_path"] = _relative(self.step_path, path)
        for part in data["parts"]:
            if part["role"] == ACTIVATION_MEMBRANE and part["props"].get("design"):
                design = Path(part["props"]["design"])
                if not design.is_absolute() and self.path:  # was relative to the old project location
                    design = Path(self.path).parent / design
                part["props"]["design"] = _relative(design, path)
        path.write_text(json.dumps(data, indent=2))
        self.path = str(path)

    @classmethod
    def load(cls, path):
        path = Path(path)
        project = cls.from_dict(json.loads(path.read_text()), path.parent)
        project.path = str(path)
        return project

    @classmethod
    def from_dict(cls, data, folder):
        """A project from its saved dict; relative paths are relative to `folder`."""
        folder = Path(folder)
        project = cls()
        step = Path(data["step_path"])
        project.step_path = str(step if step.is_absolute() else (folder / step).resolve())
        project.parts = [PartSettings(**p) for p in data["parts"]]
        for part in project.parts:
            design = part.props.get("design") if part.role == ACTIVATION_MEMBRANE else None
            if design and not Path(design).is_absolute():
                part.props["design"] = str((folder / design).resolve())
        for part in project.parts:  # projects saved with older chamber models
            model = part.props.get("model")
            if part.role == CHAMBER and model in LEGACY_MODELS:
                part.props["model"] = LEGACY_MODELS[model]
                part.props["stiffness"] = INCOMPRESSIBLE_STIFFNESS
        project.solver = SolverSettings(**data.get("solver", {}))
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
