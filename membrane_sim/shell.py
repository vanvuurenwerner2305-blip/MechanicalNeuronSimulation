"""
Thin shell made of 3-node triangles.

Unknowns: node positions x (3 per node) and, when bending is on, one mid-edge normal
rotation psi per edge.

The shell energy is the sum of
  * membrane energy: geometrically exact (large strain) plane-stress triangle,
    incompressible neo-Hookean or St. Venant-Kirchhoff, plus an optional isotropic
    pre-tension (energy = tension * current area);
  * bending energy: geometrically nonlinear Morley triangle. Every edge carries a
    mid-edge normal; its rotation relative to a face is
        phi = theta/2 + s*psi     (interior edge, theta = dihedral angle, s = +-1 edge orientation)
        phi = psi                 (free boundary edge: zero moment)
        phi = angle(n_face, N0)   (clamped boundary edge: mid-edge normal held at the rest normal)
    and the face curvature is constant,
        kappa = 1/A * sum_i l_i * phi_i * m_i (x) m_i,
    (l_i edge length, m_i in-plane outward edge normal, both from the rest state), with
    energy A * D/2 * (nu tr(dk)^2 + (1-nu) dk:dk). The psi dofs make this the Morley
    plate element, which converges independently of mesh orientation.

Element gradients and Hessians come from torch.func (exact derivatives of the energy),
which is what gives the Newton solver its consistent tangent.
"""
from functools import partial

import numpy as np
import torch
from torch.func import vmap, grad, hessian

from .mesh import boundary_nodes

DTYPE = torch.float64
EDGE_INTERIOR, EDGE_FREE, EDGE_CLAMPED = 0, 1, 2


# -----------------------------
# Element energies (single element, vmapped below)
# -----------------------------

def _right_cauchy_green(xe, Dm_inv):
    Ds = torch.stack((xe[1] - xe[0], xe[2] - xe[0]), dim=1)  # (3, 2)
    F = Ds @ Dm_inv                                           # (3, 2) surface deformation gradient
    area = 0.5 * torch.linalg.norm(torch.linalg.cross(Ds[:, 0], Ds[:, 1]))
    return F.T @ F, area


def _neo_hookean_energy(xe, Dm_inv, A0, mu_t, lam_t, tension):
    """Incompressible neo-Hookean membrane (thickness change eliminated): mu t/2 (I1 + 1/J^2 - 3)."""
    C, area = _right_cauchy_green(xe, Dm_inv)
    I1 = C[0, 0] + C[1, 1]
    J2 = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    return A0 * 0.5 * mu_t * (I1 + 1.0 / J2 - 3.0) + tension * area


def _svk_energy(xe, Dm_inv, A0, mu_t, lam_t, tension):
    """Plane-stress St. Venant-Kirchhoff membrane on the Green-Lagrange strain."""
    C, area = _right_cauchy_green(xe, Dm_inv)
    E = 0.5 * (C - torch.eye(2, dtype=C.dtype, device=C.device))
    trE = E[0, 0] + E[1, 1]
    return A0 * (0.5 * lam_t * trE ** 2 + mu_t * (E * E).sum()) + tension * area


def _edge_rotations(q, N0, etype, esign):
    """
    Mid-edge normal rotations phi (3,) of one triangle.
    q = [a, b, c, opposite node across edge 0, 1, 2 (18 coords), psi_0, psi_1, psi_2];
    edge k is opposite vertex k.
    """
    xb, psi = q[:18].reshape(6, 3), q[18:]
    a, b, c = xb[0], xb[1], xb[2]
    n1 = torch.linalg.cross(b - a, c - a)
    verts = (a, b, c)
    phis = []
    for k in range(3):
        i, j, l = verts[(k + 1) % 3], verts[(k + 2) % 3], xb[3 + k]
        e = j - i
        n_neighbour = torch.linalg.cross(i - j, l - j)
        n2 = torch.where(etype[k] == EDGE_INTERIOR, n_neighbour,
                         torch.where(etype[k] == EDGE_CLAMPED, N0, n1))
        theta = torch.atan2(torch.dot(torch.linalg.cross(n1, n2), e) / torch.linalg.norm(e),
                            torch.dot(n1, n2))
        phi = torch.where(etype[k] == EDGE_INTERIOR, 0.5 * theta + esign[k] * psi[k],
                          torch.where(etype[k] == EDGE_CLAMPED, theta, psi[k]))
        phis.append(phi)
    return torch.stack(phis)


def _bending_energy(q, N0, etype, esign, phi0, M):
    d = _edge_rotations(q, N0, etype, esign) - phi0
    return d @ M @ d


class Shell:
    def __init__(self,
                 vertices,
                 faces,
                 thickness: float,
                 youngs_modulus: float,
                 poisson_ratio: float = 0.3,
                 material: str = "neo_hookean",
                 bending: bool = True,
                 pretension: float = 0.0,
                 fixed=None,
                 boundary_rotation: str = "clamped",
                 color: str = "red",
                 name: str = None,
                 device="cpu"):
        """
        vertices, faces   : rest mesh (see mesh.py for orientation convention).
        thickness         : shell thickness t.
        youngs_modulus    : E. For "neo_hookean" the material is incompressible (nu = 0.5,
                            shear modulus E/3) and poisson_ratio is ignored.
        material          : "neo_hookean" (rubber, large strain) or "svk".
        bending           : include bending stiffness D = E t^3 / (12 (1 - nu^2)).
        pretension        : isotropic in-plane pre-tension (force / length).
        fixed             : bool mask or index list of pinned nodes. Default: all boundary nodes.
        boundary_rotation : "clamped" (boundary edges between pinned nodes keep their rest
                            slope) or "free" (hinged).
        """
        self.device = torch.device(device)
        self.color = color
        self.name = name
        self.faces_np = np.asarray(faces, dtype=np.int64)
        X_np = np.asarray(vertices, dtype=float)
        n_nodes = X_np.shape[0]

        # Material
        E, t = float(youngs_modulus), float(thickness)
        if material == "neo_hookean":
            nu, mu, lam = 0.5, E / 3.0, 0.0
            energy_fn = _neo_hookean_energy
        elif material == "svk":
            nu = float(poisson_ratio)
            mu, lam = E / (2.0 * (1.0 + nu)), E * nu / (1.0 - nu ** 2)
            energy_fn = _svk_energy
        else:
            raise ValueError(f"Unknown material '{material}'")
        self.material = material
        self.thickness = t
        self.youngs_modulus = E
        self.poisson_ratio = nu
        self.bending = bending
        self.bending_stiffness = E * t ** 3 / (12.0 * (1.0 - nu ** 2)) if bending else 0.0
        self.pretension = float(pretension)

        # Boundary conditions
        if fixed is None:
            fixed_mask = boundary_nodes(self.faces_np, n_nodes)
        else:
            fixed = np.asarray(fixed)
            fixed_mask = fixed.astype(bool) if fixed.dtype == bool else np.isin(np.arange(n_nodes), fixed)

        # Rest geometry
        F = self.faces_np
        a, b, c = X_np[F[:, 0]], X_np[F[:, 1]], X_np[F[:, 2]]
        n = np.cross(b - a, c - a)
        A0 = 0.5 * np.linalg.norm(n, axis=1)
        if np.any(A0 <= 0):
            raise ValueError("Mesh contains degenerate triangles.")
        N0 = n / (2.0 * A0[:, None])
        e1 = (b - a) / np.linalg.norm(b - a, axis=1, keepdims=True)
        e2 = np.cross(N0, e1)
        Dm = np.stack([np.stack([np.einsum("ij,ij->i", e1, b - a), np.einsum("ij,ij->i", e1, c - a)], 1),
                       np.stack([np.einsum("ij,ij->i", e2, b - a), np.einsum("ij,ij->i", e2, c - a)], 1)], 1)

        nodal_area = np.zeros(n_nodes)
        for k in range(3):
            np.add.at(nodal_area, F[:, k], A0 / 3.0)

        # Edge topology and mid-edge rotation dofs
        topo = self._build_edge_topology(F, fixed_mask, boundary_rotation)
        bend_nodes, edge_type, edge_id, edge_sign, n_edges = topo
        self.n_edges = n_edges if bending else 0
        edge_fixed = np.zeros(self.n_edges, dtype=bool)
        if bending:
            edge_fixed[edge_id[edge_type == EDGE_CLAMPED]] = True  # rotation prescribed by the clamp

        # Tensors
        tt = partial(torch.as_tensor, dtype=DTYPE, device=self.device)
        self.X = tt(X_np)
        self.x = self.X.clone()
        self.psi = torch.zeros(self.n_edges, dtype=DTYPE, device=self.device)
        self.faces = torch.as_tensor(F, device=self.device)
        self.fixed = torch.as_tensor(fixed_mask, device=self.device)
        self.fixed_dofs = torch.as_tensor(np.concatenate([np.repeat(fixed_mask, 3), edge_fixed]), device=self.device)
        self.rest_area = tt(A0)
        self.nodal_area = tt(nodal_area)
        self.volume_origin = self.X.mean(dim=0)  # reference point for cone volumes (see fluid.py)
        self._Dm_inv = tt(np.linalg.inv(Dm))
        self._N0 = tt(N0)
        self._bend_nodes = torch.as_tensor(bend_nodes, device=self.device)
        self._edge_id = torch.as_tensor(edge_id, device=self.device)
        self._edge_type = torch.as_tensor(edge_type, device=self.device)
        self._edge_sign = tt(edge_sign)
        self._M = tt(self._bending_matrices(X_np, F, N0, A0, nu, self.bending_stiffness))

        # Local dof maps: node dofs are node*3 + component, edge dofs follow after all nodes
        comp = torch.arange(3, device=self.device)
        self.face_dofs = (3 * self.faces[:, :, None] + comp).reshape(-1, 9)
        self.bend_dofs = torch.cat([(3 * self._bend_nodes[:, :, None] + comp).reshape(-1, 18),
                                    3 * n_nodes + self._edge_id], dim=1)

        # Vectorised element kernels
        mem = partial(energy_fn, mu_t=mu * t, lam_t=lam * t, tension=self.pretension)
        self._mem_e, self._mem_g, self._mem_h = vmap(mem), vmap(grad(mem)), vmap(hessian(mem))
        self._bend_e = vmap(_bending_energy)
        self._bend_g = vmap(grad(_bending_energy))
        self._bend_h = vmap(hessian(_bending_energy))
        if bending:
            self._phi0 = vmap(_edge_rotations)(self._bend_q(self.X, self.psi), self._N0, self._edge_type,
                                               self._edge_sign)

        # Pressure couplings are registered by FluidVolume.add_boundary
        self.fluid_volumes = []

    # -----------------------------
    # Setup helpers
    # -----------------------------

    @staticmethod
    def _build_edge_topology(F, fixed_mask, boundary_rotation):
        directed = {}
        for f in range(len(F)):
            for k in range(3):
                key = (F[f, (k + 1) % 3], F[f, (k + 2) % 3])
                if key in directed:
                    raise ValueError("Shell mesh is non-manifold or inconsistently oriented.")
                directed[key] = (f, k)

        bend_nodes = np.concatenate([F, F], axis=1)  # boundary edges point at the own opposite vertex (unused)
        edge_type = np.full((len(F), 3), EDGE_FREE, dtype=np.int64)
        edge_id = np.zeros((len(F), 3), dtype=np.int64)
        edge_sign = np.zeros((len(F), 3))
        undirected = {}
        for (i, j), (f, k) in directed.items():
            edge_id[f, k] = undirected.setdefault((min(i, j), max(i, j)), len(undirected))
            edge_sign[f, k] = 1.0 if i < j else -1.0
            twin = directed.get((j, i))
            if twin is not None:
                g, m = twin
                bend_nodes[f, 3 + k] = F[g, m]
                edge_type[f, k] = EDGE_INTERIOR
            elif boundary_rotation == "clamped" and fixed_mask[i] and fixed_mask[j]:
                edge_type[f, k] = EDGE_CLAMPED
        return bend_nodes, edge_type, edge_id, edge_sign, len(undirected)

    @staticmethod
    def _bending_matrices(X, F, N0, A0, nu, D):
        """Per-face 3x3 matrix M with energy = dphi^T M dphi."""
        verts = X[F]                                                   # (F, 3, 3)
        e = np.stack([verts[:, (k + 2) % 3] - verts[:, (k + 1) % 3] for k in range(3)], axis=1)
        l = np.linalg.norm(e, axis=2)                                  # (F, 3)
        m = np.cross(e, N0[:, None, :]) / l[..., None]                 # in-plane outward edge normals
        G = nu + (1.0 - nu) * np.einsum("fid,fjd->fij", m, m) ** 2
        return D / (2.0 * A0)[:, None, None] * l[:, :, None] * l[:, None, :] * G

    def _bend_q(self, x, psi):
        return torch.cat([x[self._bend_nodes].reshape(-1, 18), psi[self._edge_id]], dim=1)

    # -----------------------------
    # State
    # -----------------------------

    @property
    def n_nodes(self) -> int:
        return self.X.shape[0]

    @property
    def n_dof(self) -> int:
        return 3 * self.n_nodes + self.n_edges

    def get_state(self) -> torch.Tensor:
        return torch.cat([self.x.reshape(-1), self.psi])

    def set_state(self, u: torch.Tensor):
        self.x = u[:3 * self.n_nodes].reshape(-1, 3)
        self.psi = u[3 * self.n_nodes:]

    def reset(self):
        self.x = self.X.clone()
        self.psi = torch.zeros_like(self.psi)

    def displacement(self) -> torch.Tensor:
        return self.x - self.X

    def fluid_volume_contacts(self, fluid_volume_tuple):
        """
        (volume_behind, volume_in_front): the volume behind the shell (opposite to its normal)
        pushes along +normal, the one in front pushes back, as in the 2D simulator.
        Either entry may be None (ambient pressure 0).
        """
        behind, in_front = fluid_volume_tuple
        if behind is not None:
            behind.add_boundary(self, +1)
        if in_front is not None:
            in_front.add_boundary(self, -1)

    # -----------------------------
    # Element contributions
    # -----------------------------

    def element_terms(self, tangent: bool = True):
        """
        Yield (local_dofs (E, d), energy (E,), gradient (E, d), hessian (E, d, d) or None)
        for the membrane and bending element groups at the current state.
        """
        args = (self.x[self.faces], self._Dm_inv, self.rest_area)
        yield (self.face_dofs,
               self._mem_e(*args),
               self._mem_g(*args).reshape(-1, 9),
               self._mem_h(*args).reshape(-1, 9, 9) if tangent else None)

        if self.bending:
            args = (self._bend_q(self.x, self.psi), self._N0, self._edge_type, self._edge_sign, self._phi0, self._M)
            yield (self.bend_dofs,
                   self._bend_e(*args),
                   self._bend_g(*args),
                   self._bend_h(*args) if tangent else None)

    def to(self, device):
        """Move all tensors to another device."""
        self.device = torch.device(device)
        for name in ("X", "x", "psi", "faces", "fixed", "fixed_dofs", "rest_area", "nodal_area", "volume_origin",
                     "_Dm_inv", "_N0", "_bend_nodes", "_edge_id", "_edge_type", "_edge_sign", "_M",
                     "face_dofs", "bend_dofs") + (("_phi0",) if self.bending else ()):
            setattr(self, name, getattr(self, name).to(self.device))
        return self
