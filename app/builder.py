"""
Turns a CAD model plus a Project into a membrane_sim.Environment.

  Membrane / Shell  -> Shell on the mid-surface of the thin solid (bending off / on)
  Activation membrane -> EmpiricalMembrane: the membrane of a pre-simulated activation-function design
                       (*.mad), replaced by its Δp -> swept volume curve (one dof); the chambers either side
                       of the part's mid-surface are its driving side and its tube side
  Rigid body        -> Obstacle (closed surface mesh)
  Fluid chamber     -> FluidVolume, coupled to every membrane/shell it touches

Coupling detection: from every mid-surface triangle a probe point is placed just beyond
each face of the thin solid (half the thickness plus a margin, along +-normal). A chamber
acts on the side whose probes fall inside it (generalised winding number of its mesh).
Units in the solver: mm, N, MPa.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

import membrane_sim as ms
from membrane_sim.contact import ObstacleField, _winding_number

from .project import (ACTIVATION_MEMBRANE, AUTOMATIC, CHAMBER, CLAMPED, DEFORMABLE, IDEAL_GAS, IGNORE,
                      INCOMPRESSIBLE, MEMBRANE, NEO_HOOKEAN, RIGID, ROLE_COLORS, SHEETS, SHELL, TOUCHING_RIGID,
                      UNASSIGNED, VENT, INCOMPRESSIBLE_STIFFNESS, part_color)

P_ATM = 0.101325  # MPa
KPA = 1e-3        # kPa -> MPa


# -----------------------------
# Meshing (runs where gmsh may be used)
# -----------------------------

@dataclass
class MeshData:
    sizes: dict
    surfaces: dict                                    # body index -> SurfaceMesh
    midsurfaces: dict = field(default_factory=dict)   # body index -> MidSurface (deformable parts)

    def element_count(self, parts):
        n = 0
        for i, mesh in self.surfaces.items():
            n += len(self.midsurfaces[i].faces) if i in self.midsurfaces else 0
        return n


DEFAULT_ELEMENTS_PER_SIDE = 10


def element_size(body, role, elements_per_side=DEFAULT_ELEMENTS_PER_SIDE):
    """Target element size: the part's shortest side divided by the number of elements along it.
    For membranes/shells the shortest in-plane side (the thickness is not a side of the sheet);
    solids use their shortest side, but never finer than diagonal / (3 n) so thin rigid parts do
    not explode into huge meshes."""
    n = max(int(elements_per_side), 1)
    sides = np.sort(body.size)[::-1]
    if role in SHEETS:
        return float(sides[1] / n)
    return float(max(sides[2] / n, body.diagonal / (3 * n)))


def mesh_sizes(cad, project):
    return {body.index: element_size(body, part.role,
                                     part.props.get("elements_per_side", DEFAULT_ELEMENTS_PER_SIDE))
            for body, part in zip(cad.bodies, project.parts)}


def measure_thickness(cad, surfaces, index):
    """Thickness of a thin solid measured through its largest face (mm)."""
    return float(cad.midsurface(cad.bodies[index], surfaces[index]).thickness)


def generate_mesh(cad, project) -> MeshData:
    sizes = mesh_sizes(cad, project)
    data = MeshData(sizes, cad.mesh(sizes))
    for body, part in zip(cad.bodies, project.parts):
        if part.role in SHEETS:
            data.midsurfaces[body.index] = cad.midsurface(body, data.surfaces[body.index])
    return data


# -----------------------------
# Environment
# -----------------------------

@dataclass
class Coupling:
    shell_index: int
    side: int            # +1: chamber is behind the shell normal (pushes along +normal)
    coverage: float      # fraction of the shell face that touches the chamber


@dataclass
class ActivationLink:
    """A neuron part replaced by an activation-function design's membrane."""
    body: object              # EmpiricalMembrane
    design: object            # app.activation.ActivationDesign
    path: str
    sides: dict               # chamber index -> +1 (driving side) / -1 (tube side)

    def pressure_difference(self, pressures) -> float:
        """Δp across the design's membrane (kPa) from the chamber pressures {chamber index: P (MPa)}."""
        return sum(side * pressures[c] for c, side in self.sides.items()) / KPA

    def outputs(self, pressures) -> dict:
        return self.design.outputs(self.pressure_difference(pressures))

    def range_warning(self, pressures, name="") -> str:
        """The extrapolation warning at these chamber pressures ("" inside the design's simulated range)."""
        return self.design.range_warning(self.pressure_difference(pressures), name)


@dataclass
class BuildResult:
    env: ms.Environment
    shells: dict                       # body index -> Shell
    obstacles: dict                    # body index -> Obstacle
    volumes: dict                      # body index -> FluidVolume
    couplings: dict                    # chamber index -> [Coupling]
    thickness: dict                    # body index -> thickness used (mm)
    contact_stiffness: float
    warnings: list = field(default_factory=list)

    auto_contact: bool = True
    activation: dict = field(default_factory=dict)  # part index -> ActivationLink

    def activation_outputs(self, step=None) -> dict:
        """{part index: design outputs} at a history step (default: the current state)."""
        if step is None:
            pressures = {c: v.P for c, v in self.volumes.items()}
        else:
            pressures = dict(zip(self.volumes.keys(), step["pressures"]))
        return {i: link.outputs(pressures) for i, link in self.activation.items()}

    def range_warnings(self, parts, step=None) -> list:
        """The extrapolation warning of every activation membrane whose pre-activation Δp is outside its design's
        simulated range (at a history step, default the current state)."""
        if step is None:
            pressures = {c: v.P for c, v in self.volumes.items()}
        else:
            pressures = dict(zip(self.volumes.keys(), step["pressures"]))
        return [w for i, link in self.activation.items() if (w := link.range_warning(pressures, parts[i].name))]

    def set_contact_stiffness(self, pressure_ref=None):
        """Automatic penalty stiffness: penetration ~5% of the thinnest part at pressure_ref
        (default: the highest chamber pressure currently set)."""
        if not self.auto_contact:
            return
        if pressure_ref is None:
            pressure_ref = max(abs(v.P0) for v in self.volumes.values()) if self.volumes else 0.0
        self.contact_stiffness = max(pressure_ref, 1.0 * KPA) / (0.05 * min(self.thickness.values(), default=1.0))
        self.env.contact_stiffness = self.contact_stiffness

    def solve(self, settings, callback=None, warm_start=False, load_steps=None, fixed_contact=False):
        if not fixed_contact:
            self.set_contact_stiffness()
        if not warm_start:
            self.env.reset()
        return self.env.solve(load_steps=load_steps or settings.load_steps, max_iterations=settings.max_iterations,
                              rtol=settings.tolerance, callback=callback, warm_start=warm_start)


def fluid_volume(props, body_volume):
    """Liquid in a 'Closed: incompressible' chamber (mm3); 0 or unset = the whole body volume."""
    return float(props.get("fluid_volume", 0.0) or 0.0) or float(body_volume)


def chamber_model(props, body_volume):
    """Keyword arguments for FluidVolume from a chamber's properties (pressures in MPa, gauge)."""
    kwargs = dict(P0=float(props.get("pressure", 0.0)) * KPA, initial_volume=float(body_volume))
    model = props.get("model")
    if model == IDEAL_GAS:
        total = float(body_volume) + max(float(props.get("ghost_volume", 0.0)), 0.0)
        liquid = min(max(float(props.get("incompressible", 0.0)), 0.0), 99.0) / 100.0
        kwargs.update(initial_volume=total, gas_volume=(1.0 - liquid) * total, atmospheric_pressure=P_ATM)
    elif model == INCOMPRESSIBLE:
        # stiffness is given per percent of the liquid's volume: dP/dV = s * 100 / V_liquid.
        # Less liquid than the body volume gives suction, more liquid inflates the chamber.
        liquid = fluid_volume(props, body_volume)
        per_percent = float(props.get("stiffness", INCOMPRESSIBLE_STIFFNESS)) * KPA
        kwargs.update(bulk_stiffness=per_percent * 100.0 / liquid, liquid_volume=liquid)
    elif model == VENT:
        kwargs.update(P0=0.0)
    return kwargs


def build_environment(cad, mesh: MeshData, project) -> BuildResult:
    parts = project.parts
    bodies = cad.bodies
    warnings = []

    rigid = [i for i, p in enumerate(parts) if p.role == RIGID]
    deformable = [i for i, p in enumerate(parts) if p.role in DEFORMABLE]
    linked = [i for i, p in enumerate(parts) if p.role == ACTIVATION_MEMBRANE]
    chambers = [i for i, p in enumerate(parts) if p.role == CHAMBER]
    unassigned = [parts[i].name for i, p in enumerate(parts) if p.role == UNASSIGNED]
    if not deformable and not linked:
        raise ValueError("Assign at least one part as Membrane, Shell or Activation membrane.")
    missing = [parts[i].name for i in deformable + linked if i not in mesh.midsurfaces]
    if missing:
        raise ValueError(f"Mesh is out of date for: {', '.join(missing)}. Generate the mesh again.")
    if unassigned:
        warnings.append(f"Unassigned parts are ignored: {', '.join(unassigned)}")

    env = ms.Environment()

    obstacles = {}
    for i in rigid:
        m = mesh.surfaces[i]
        obstacles[i] = env.add_obstacle(m.vertices, m.faces, color=ROLE_COLORS[RIGID])
    rigid_field = ObstacleField(list(obstacles.values())) if obstacles else None

    shells, thickness = {}, {}
    for i in deformable:
        part, mid = parts[i], mesh.midsurfaces[i]
        t = float(part.props.get("thickness", 0.0) or 0.0) or mid.thickness
        thickness[i] = t
        boundary = ms.boundary_nodes(mid.faces, len(mid.vertices))
        fixed = boundary
        if part.props.get("fixed_edges") == TOUCHING_RIGID:
            if rigid_field is None:
                warnings.append(f"{part.name}: no rigid bodies to attach to, all boundary edges fixed.")
            else:
                sd, _ = rigid_field.signed_distance(torch.as_tensor(mid.vertices), max_distance=t)
                fixed = boundary & (sd.numpy() <= 0.75 * t + 1e-6 * bodies[i].diagonal)
        if not fixed.any():
            warnings.append(f"{part.name}: no fixed nodes - the part is free to move as a rigid body.")
        shells[i] = env.add_membrane(
            mid.vertices, mid.faces, thickness=t,
            youngs_modulus=float(part.props["youngs_modulus"]),
            poisson_ratio=float(part.props.get("poisson_ratio", 0.45)),
            material="neo_hookean" if part.props.get("material", NEO_HOOKEAN) == NEO_HOOKEAN else "svk",
            bending=part.role == SHELL,
            pretension=float(part.props.get("pretension", 0.0)),
            fixed=fixed,
            boundary_rotation="clamped" if part.props.get("edge_rotation", CLAMPED) == CLAMPED else "free",
            contact_offset=0.5 * t,
            color=ROLE_COLORS[part.role], name=part.name)

    empirical = {i: _empirical_membrane(parts[i], mesh.midsurfaces[i], getattr(project, "path", None), warnings)
                 for i in linked}
    for i in linked:
        env.membrane_list.append(empirical[i][0])
    linked_sides = {i: {} for i in linked}  # part -> {chamber: side of its mid-surface}

    volumes, couplings = {}, {}
    for c in chambers:
        part, body = parts[c], bodies[c]
        volume = env.add_fluid_volume(name=part.name, color=part_color(part), **chamber_model(part.props, body.volume))
        volumes[c] = volume
        couplings[c] = []
        cm = mesh.surfaces[c]
        tri = torch.as_tensor(cm.vertices[cm.faces])
        for i in deformable:
            side, coverage = _detect_side(mesh.midsurfaces[i], thickness[i], tri)
            if side == 0:
                continue
            if side == 2:
                warnings.append(f"{part.name} lies on both sides of {parts[i].name}; no net pressure, ignored.")
                continue
            volume.add_boundary(shells[i], side)
            couplings[c].append(Coupling(i, side, coverage))
            if coverage < 0.9:
                warnings.append(f"{part.name} touches only {coverage:.0%} of {parts[i].name}; "
                                f"its pressure is applied to the whole face.")
        for i in linked:
            mid = mesh.midsurfaces[i]
            t = float(parts[i].props.get("thickness", 0.0) or 0.0) or mid.thickness
            side, coverage = _detect_side(mid, t, tri)
            if side in (1, -1):
                linked_sides[i][c] = side
                couplings[c].append(Coupling(i, side, coverage))
            elif side == 2:
                warnings.append(f"{part.name} lies on both sides of {parts[i].name}; no net pressure, ignored.")
        if not couplings[c]:
            warnings.append(f"{part.name} does not touch any membrane or shell.")

    for i in deformable:
        if not shells[i].fluid_volumes:
            warnings.append(f"{parts[i].name} is not loaded by any fluid chamber.")

    activation = {}
    for i in linked:
        body, design, path = empirical[i]
        sides = _orient_activation_membrane(parts, i, linked_sides[i], warnings)
        for c, side in sides.items():
            volumes[c].add_boundary(body, side)
        activation[i] = ActivationLink(body, design, path, sides)

    k = project.solver.contact_stiffness
    env.contact_stiffness = k
    build = BuildResult(env, shells, obstacles, volumes, couplings, thickness, k, warnings, auto_contact=not k,
                        activation=activation)
    build.set_contact_stiffness()
    return build


def _detect_side(mid, t, chamber_triangles, threshold=0.3):
    """0: not touching, +1: chamber behind the normal, -1: in front, 2: both sides."""
    V, F = mid.vertices, mid.faces
    centroid = V[F].mean(axis=1)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    reach = 0.5 * t + max(0.25 * t, 1e-3)
    a, b, c = chamber_triangles[:, 0], chamber_triangles[:, 1], chamber_triangles[:, 2]
    fractions = []
    for sign in (+1, -1):
        probes = torch.as_tensor(centroid + sign * reach * n)
        inside = torch.cat([_winding_number(p, a, b, c) > 0.5 for p in probes.split(256)])
        fractions.append(inside.double().mean().item())
    front, behind = fractions
    if front >= threshold and behind >= threshold:
        return 2, max(front, behind)
    if behind >= threshold:
        return +1, behind
    if front >= threshold:
        return -1, front
    return 0, 0.0


def design_path(part, project_path=None) -> Path:
    """The activation design file of an Activation membrane part (relative paths: next to the project)."""
    text = str(part.props.get("design", "") or "").strip()
    if not text:
        raise ValueError(f"{part.name}: choose its activation design (*.mad) in the part's properties.")
    path = Path(text)
    if not path.is_absolute() and project_path:
        path = Path(project_path).parent / path
    return path


def _empirical_membrane(part, mid, project_path, warnings):
    """(EmpiricalMembrane, ActivationDesign, path) of an Activation membrane part."""
    from .activation import ActivationDesign  # activation imports this module
    path = design_path(part, project_path)
    if not path.exists():
        raise ValueError(f"{part.name}: activation design not found: {path}")
    design = ActivationDesign.load(path)
    if not design.has_volume:
        raise ValueError(f"{part.name}: {path.name} was simulated before the membrane's swept volume was "
                         "recorded. Open it in the Pre-activation → activation space, run its study again and save it.")
    V, F = mid.vertices, mid.faces
    area = 0.5 * float(np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum())
    if abs(area - design.membrane_area) > 0.1 * design.membrane_area:
        warnings.append(f"{part.name} has {area:.4g} mm² of membrane, the design {path.name} "
                        f"{design.membrane_area:.4g} mm²; the design's response is used as simulated.")
    body = ms.EmpiricalMembrane(design.dp * KPA, design.volume, area, name=part.name,
                                color=ROLE_COLORS[ACTIVATION_MEMBRANE])
    return body, design, str(path)


def _orient_activation_membrane(parts, i, touching, warnings):
    """{chamber index: +1 driving side / -1 tube side} from the chambers touching the part's two faces
    ({chamber: side of the mid-surface}). The driving chamber is the chosen one, or automatically the
    closed chamber it touches (the pre-activation chamber), else the first one."""
    name = parts[i].name
    if not touching:
        warnings.append(f"{name} touches no fluid chamber: nothing drives the activation design.")
        return {}
    chosen = parts[i].props.get("driving", AUTOMATIC) or AUTOMATIC
    by_name = {parts[c].name: c for c in touching}
    if chosen != AUTOMATIC and chosen not in by_name:
        warnings.append(f"{name}: the driving chamber {chosen} does not touch it; chosen automatically.")
        chosen = AUTOMATIC
    if chosen == AUTOMATIC:
        closed = [c for c in touching if parts[c].props.get("model") in (IDEAL_GAS, INCOMPRESSIBLE)]
        drive = closed[0] if closed else next(iter(touching))
        if len({touching[c] for c in closed}) > 1:
            warnings.append(f"{name} has closed chambers on both sides; {parts[drive].name} was taken as the "
                            "driving chamber - choose it in the part's properties.")
    else:
        drive = by_name[chosen]
    front = touching[drive]
    sides = {c: (1 if s == front else -1) for c, s in touching.items()}
    for sign, what in ((1, "driving"), (-1, "tube")):
        several = [parts[c].name for c, s in sides.items() if s == sign]
        if len(several) > 1:
            warnings.append(f"{name}: several chambers on its {what} side ({', '.join(several)}); each one's "
                            "full pressure acts on it.")
    return sides
