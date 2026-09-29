"""
CAD layer: STEP import through gmsh (OpenCASCADE), per-body surface meshing and
mid-surface extraction of thin bodies.

Every solid in the STEP file is one body. Lengths are in millimetres.
gmsh must be initialised from the main thread (CadModel.load_step); meshing may then run
from a worker thread as long as only one thread uses gmsh at a time. gmsh holds a single
global model, so only the most recently loaded CadModel can be meshed.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import gmsh
import numpy as np


@dataclass
class CadBody:
    index: int
    tag: int
    name: str
    volume: float
    area: float
    bbox: np.ndarray                                   # (xmin, ymin, zmin, xmax, ymax, zmax)
    surfaces: list = field(default_factory=list)       # [(surface tag, area)]

    @property
    def size(self) -> np.ndarray:
        return self.bbox[3:] - self.bbox[:3]

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.size))

    @property
    def thickness_estimate(self) -> float:
        """2V/A: the thickness for a thin plate-like body."""
        return 2.0 * self.volume / self.area

    @property
    def primary_surface(self) -> int:
        return max(self.surfaces, key=lambda s: s[1])[0]


@dataclass
class SurfaceMesh:
    vertices: np.ndarray        # (V, 3)
    faces: np.ndarray           # (F, 3), outward oriented
    face_surface: np.ndarray    # (F,) CAD surface tag of every triangle


@dataclass
class VolumeMesh:
    vertices: np.ndarray        # (N, 3)
    tets: np.ndarray            # (T, 4) or (T, 10)
    faces: np.ndarray           # (F, 3) boundary sub-triangles, outward oriented
    face_surface: np.ndarray    # (F,) CAD surface tag of every boundary triangle


@dataclass
class MidSurface:
    vertices: np.ndarray
    faces: np.ndarray           # normal points out of the solid through the primary CAD face
    thickness: float            # median measured thickness
    node_thickness: np.ndarray


_ACTIVE = None  # the CadModel whose STEP file is currently in gmsh's (single, global) model


class CadModel:
    def __init__(self):
        self.path = None
        self.bodies = []

    def _ensure_active(self):
        """gmsh holds one model: if another CadModel (the other space) loaded a file since, load ours again.
        Re-importing the same file gives the same entity tags."""
        if _ACTIVE is not self and self.path:
            self.load_step(self.path)

    # -----------------------------
    # Import
    # -----------------------------

    def load_step(self, path: str):
        if not gmsh.isInitialized():
            gmsh.initialize()
        gmsh.clear()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.model.add("cad")
        gmsh.model.occ.importShapes(str(path), highestDimOnly=True)
        gmsh.model.occ.synchronize()

        global _ACTIVE
        _ACTIVE = self
        self.path = str(path)
        self.bodies = []
        entities = gmsh.model.getEntities(3)
        names = [gmsh.model.getEntityName(3, tag).split("/")[-1].strip() for _, tag in entities]
        if any(not n for n in names) or len(set(names)) < len(names):
            # Multi-body parts (e.g. Fusion 360 exports) carry the body names on the solids, not
            # on products; gmsh does not read those, so take them from the file when they line up.
            solid_names = _step_solid_names(path)
            if len(solid_names) == len(entities) and all(solid_names):
                names = solid_names
        used = {}
        for i, (_, tag) in enumerate(entities):
            name = names[i] or f"Body {i + 1}"
            if name in used:
                used[name] += 1
                name = f"{name} ({used[name]})"
            else:
                used[name] = 1
            surfaces = [(abs(s), gmsh.model.occ.getMass(2, abs(s)))
                        for _, s in gmsh.model.getBoundary([(3, tag)], oriented=False)]
            self.bodies.append(CadBody(
                index=i, tag=tag, name=name,
                volume=gmsh.model.occ.getMass(3, tag),
                area=sum(a for _, a in surfaces),
                bbox=np.array(gmsh.model.occ.getBoundingBox(3, tag)),
                surfaces=surfaces))
        if not self.bodies:
            raise ValueError("The STEP file contains no solids.")
        return self.bodies

    # -----------------------------
    # Meshing
    # -----------------------------

    def mesh(self, sizes: dict) -> dict:
        """
        Triangulate the boundary of every body. sizes: {body index: target element size (mm)}.
        Returns {body index: SurfaceMesh}.
        """
        self._ensure_active()
        gmsh.model.mesh.clear()
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        for body in self.bodies:
            points = gmsh.model.getBoundary([(3, body.tag)], recursive=True)
            gmsh.model.mesh.setSize(points, float(sizes.get(body.index, body.diagonal / 15.0)))
        gmsh.model.mesh.generate(2)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        lookup = np.zeros(int(node_tags.max()) + 1, dtype=np.int64)
        lookup[node_tags.astype(np.int64)] = np.arange(len(node_tags))
        coords = coords.reshape(-1, 3)

        meshes = {}
        for body in self.bodies:
            faces, owner = [], []
            for surface, _ in body.surfaces:
                tri = gmsh.model.mesh.getElementsByType(2, surface)[1].astype(np.int64).reshape(-1, 3)
                faces.append(lookup[tri])
                owner.append(np.full(len(tri), surface))
            faces = np.vstack(faces)
            used, inverse = np.unique(faces, return_inverse=True)
            vertices = coords[used]
            faces = orient_outward(vertices, inverse.reshape(-1, 3))
            meshes[body.index] = SurfaceMesh(vertices, faces, np.concatenate(owner))
        return meshes

    def volume_mesh(self, index: int, size: float, order: int = 2) -> "VolumeMesh":
        """
        Tetrahedral mesh of one body (e.g. a soft tube), 10-node tetrahedra for order 2.
        Boundary triangles come back split into flat sub-triangles (4 per 6-node triangle),
        outward oriented, each with its CAD surface tag. Clears any other mesh in gmsh.
        """
        self._ensure_active()
        body = self.bodies[index]
        gmsh.model.mesh.clear()
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.MeshOnlyVisible", 1)
        try:
            gmsh.model.setVisibility(gmsh.model.getEntities(3), 0, recursive=True)
            gmsh.model.setVisibility([(3, body.tag)], 1, recursive=True)
            points = gmsh.model.getBoundary([(3, body.tag)], recursive=True)
            gmsh.model.mesh.setSize(points, float(size))
            gmsh.model.mesh.generate(3)
            if order == 2:
                gmsh.model.mesh.setOrder(2)
            node_tags, coords, _ = gmsh.model.mesh.getNodes()
            lookup = np.zeros(int(node_tags.max()) + 1, dtype=np.int64)
            lookup[node_tags.astype(np.int64)] = np.arange(len(node_tags))
            coords = coords.reshape(-1, 3)
            tet_type = 11 if order == 2 else 4
            tets = lookup[gmsh.model.mesh.getElementsByType(tet_type, body.tag)[1].astype(np.int64)]
            tets = tets.reshape(-1, 10 if order == 2 else 4)
            tri_type = 9 if order == 2 else 2
            faces, owner = [], []
            for surface, _ in body.surfaces:
                tri = lookup[gmsh.model.mesh.getElementsByType(tri_type, surface)[1].astype(np.int64)]
                tri = tri.reshape(-1, 6 if order == 2 else 3)
                if order == 2:  # corners 0 1 2, mid-edge nodes 3 (01), 4 (12), 5 (20)
                    tri = np.concatenate([tri[:, [0, 3, 5]], tri[:, [3, 1, 4]], tri[:, [5, 4, 2]],
                                          tri[:, [3, 4, 5]]])
                faces.append(tri)
                owner.append(np.full(len(tri), surface))
        finally:
            gmsh.model.setVisibility(gmsh.model.getEntities(3), 1, recursive=True)
            gmsh.option.setNumber("Mesh.MeshOnlyVisible", 0)
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
            gmsh.model.mesh.clear()
        used, inverse = np.unique(tets, return_inverse=True)
        renumber = np.full(len(coords), -1, dtype=np.int64)
        renumber[used] = np.arange(len(used))
        faces = renumber[np.vstack(faces)]
        if (faces < 0).any():
            raise ValueError(f"{body.name}: surface and volume meshes do not match.")
        vertices = coords[used]
        return VolumeMesh(vertices, inverse.reshape(tets.shape), orient_outward(vertices, faces),
                          np.concatenate(owner))

    # -----------------------------
    # Thin bodies
    # -----------------------------

    @staticmethod
    def midsurface(body: CadBody, mesh: SurfaceMesh) -> MidSurface:
        """
        Mid-surface of a thin solid: the mesh of its largest CAD face, with every node moved
        half the local thickness inwards (thickness measured by a ray cast through the solid).
        """
        on_primary = mesh.face_surface == body.primary_surface
        used, inverse = np.unique(mesh.faces[on_primary], return_inverse=True)
        V = mesh.vertices[used]
        F = inverse.reshape(-1, 3)

        normals = _vertex_normals(V, F)
        opposite = mesh.vertices[mesh.faces[~on_primary]]
        d = _first_ray_hit(V, -normals, opposite, t_min=1e-6 * body.diagonal)
        finite = np.isfinite(d)
        median = float(np.median(d[finite])) if finite.any() else body.thickness_estimate
        bad = ~finite | (d > 3.0 * median) | (d < median / 3.0)
        d[bad] = median
        return MidSurface(V - 0.5 * d[:, None] * normals, F, median, d)

    def close(self):
        if gmsh.isInitialized():
            gmsh.finalize()


# -----------------------------
# Geometry helpers
# -----------------------------

def _step_solid_names(path):
    """Names of the solid bodies (MANIFOLD_SOLID_BREP / BREP_WITH_VOIDS) in shape-representation order."""
    text = Path(path).read_text(errors="ignore")
    solids = dict(re.findall(r"#(\d+)\s*=\s*(?:MANIFOLD_SOLID_BREP|BREP_WITH_VOIDS)\s*\(\s*'([^']*)'", text))
    order = []
    for items in re.findall(r"SHAPE_REPRESENTATION\s*\(\s*'[^']*'\s*,\s*\(([^)]*)\)", text):
        for ref in re.findall(r"#(\d+)", items):
            if ref in solids and ref not in order:
                order.append(ref)
    return [solids[r] for r in order]


def orient_outward(vertices, faces):
    """
    Make a closed triangle mesh consistently oriented with normals pointing out of the solid:
    each connected shell is made consistent by walking over shared edges, then flipped so
    that outer shells enclose positive volume and cavity shells negative volume.
    """
    faces = faces.copy()
    edge_faces = {}
    for f, (a, b, c) in enumerate(faces):
        for i, j in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault((min(i, j), max(i, j)), []).append(f)

    component = np.full(len(faces), -1)
    n_components = 0
    for seed in range(len(faces)):
        if component[seed] >= 0:
            continue
        component[seed] = n_components
        stack = [seed]
        while stack:
            f = stack.pop()
            a, b, c = faces[f]
            for i, j in ((a, b), (b, c), (c, a)):
                for g in edge_faces[(min(i, j), max(i, j))]:
                    if g == f or component[g] >= 0:
                        continue
                    ga, gb, gc = faces[g]
                    if (i, j) in ((ga, gb), (gb, gc), (gc, ga)):  # same direction: inconsistent
                        faces[g] = faces[g][::-1]
                    component[g] = n_components
                    stack.append(g)
        n_components += 1

    for k in range(n_components):
        if _signed_volume(vertices, faces[component == k]) < 0:
            faces[component == k] = faces[component == k][:, ::-1]
    if n_components > 1:  # shells nested an odd number of times bound cavities
        from membrane_sim.contact import _winding_number
        import torch
        for k in range(n_components):
            probe = torch.as_tensor(vertices[faces[component == k][0, 0]][None], dtype=torch.float64)
            depth = 0
            for m in range(n_components):
                if m != k:
                    tri = torch.as_tensor(vertices[faces[component == m]], dtype=torch.float64)
                    depth += int(_winding_number(probe, tri[:, 0], tri[:, 1], tri[:, 2]).item() > 0.5)
            if depth % 2:
                faces[component == k] = faces[component == k][:, ::-1]
    return faces


def _signed_volume(vertices, faces):
    v = vertices[faces]
    return np.einsum("fi,fi->f", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0


def _vertex_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])  # area weighted
    normals = np.zeros_like(V)
    for k in range(3):
        np.add.at(normals, F[:, k], n)
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def _first_ray_hit(origins, directions, triangles, t_min, chunk_elements=2_000_000):
    """Distance along each ray to the first triangle hit (Moller-Trumbore); inf when missed."""
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    result = np.full(len(origins), np.inf)
    chunk = max(1, chunk_elements // max(1, len(triangles)))
    tol = 1e-9
    for start in range(0, len(origins), chunk):
        o = origins[start:start + chunk, None, :]
        d = directions[start:start + chunk, None, :]
        p = np.cross(d, e2)
        det = np.einsum("ntk,tk->nt", p, e1)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / det
            s = o - v0
            u = np.einsum("ntk,ntk->nt", s, p) * inv
            q = np.cross(s, e1)
            v = np.einsum("ntk,ntk->nt", d, q) * inv
            t = np.einsum("tk,ntk->nt", e2, q) * inv
            hit = (np.abs(det) > 1e-14) & (u >= -tol) & (v >= -tol) & (u + v <= 1 + tol) & (t > t_min)
        result[start:start + chunk] = np.where(hit, t, np.inf).min(axis=1)
    return result
