"""
Deformable 3D solid made of tetrahedra (linear 4-node or quadratic 10-node), e.g. the soft tube
that an activation-function device squeezes shut.

Material: compressible neo-Hookean,
    W = mu/2 (I1 - 3 - 2 ln J) + lambda/2 (ln J)^2,
integrated with 1 (TET4) or 4 (TET10) Gauss points per element. Quadratic elements are the
default in the app: a tube is squeezed by bending its walls, and linear tetrahedra with only a
few elements through a wall are far too stiff in bending.

The solid presents the same interface to the solver as a Shell (node coordinates as dofs,
`element_terms`, `faces` for contact and chamber volumes). Its `faces` are the boundary
triangles; a quadratic (6-node) boundary triangle is split into 4 flat sub-triangles through
its mid-edge nodes, which is what contact, chamber volumes and cross-sections use.
"""
from functools import partial

import numpy as np
import torch
from torch.func import grad, hessian, vmap

DTYPE = torch.float64

# 4-point Gauss rule on the tetrahedron (degree 2), barycentric coordinates
_A, _B = 0.5854101966249685, 0.1381966011250105
TET_GAUSS_4 = np.array([[_A, _B, _B, _B], [_B, _A, _B, _B], [_B, _B, _A, _B], [_B, _B, _B, _A]])
TET10_EDGES = [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)]  # gmsh order of the mid-edge nodes


def _shape_gradients_bary(L, order):
    """dN/dL (n_nodes, 4) at barycentric point L for TET4 / TET10 (edges in TET10_EDGES order)."""
    if order == 1:
        return np.eye(4)
    d = np.zeros((10, 4))
    for i in range(4):
        d[i, i] = 4.0 * L[i] - 1.0
    for k, (i, j) in enumerate(TET10_EDGES):
        d[4 + k, i] = 4.0 * L[j]
        d[4 + k, j] = 4.0 * L[i]
    return d


def _neo_hookean(xe, dNdX, wdV, mu, lam):
    """xe (n, 3) element nodes; dNdX (q, n, 3); wdV (q,) quadrature weight x rest volume."""
    F = torch.einsum("ai,qaj->qij", xe, dNdX)
    # explicit triple product: torch.linalg.det's second derivative is NaN at F = I
    J = (F[:, :, 0] * torch.linalg.cross(F[:, :, 1], F[:, :, 2], dim=-1)).sum(-1)
    I1 = (F * F).sum(dim=(1, 2))
    lnJ = torch.log(J)
    W = 0.5 * mu * (I1 - 3.0 - 2.0 * lnJ) + 0.5 * lam * lnJ ** 2
    return (wdV * W).sum()


def _element_energy(q, dNdX, wdV, mu, lam):
    return _neo_hookean(q.reshape(-1, 3), dNdX, wdV, mu, lam)


class Solid:
    sheet_contact = False   # not part of ShellContact (mid-surface sheets)
    is_rigid = False

    def __init__(self, vertices, tets, faces, youngs_modulus: float, poisson_ratio: float = 0.45,
                 fixed=None, color: str = "#c77dff", name: str = None, device="cpu"):
        """
        vertices : (N, 3) rest node coordinates.
        tets     : (T, 4) or (T, 10) node indices (10-node: mid-edge nodes in gmsh order, or any
                   order: they are matched to their edges from the rest geometry).
        faces    : (F, 3) boundary triangles with outward normals (sub-triangles for quadratic
                   meshes), used for contact, chamber volumes and display.
        fixed    : bool mask or index list of fixed nodes.
        """
        self.device = torch.device(device)
        self.color, self.name = color, name
        X = np.asarray(vertices, dtype=float)
        T = np.asarray(tets, dtype=np.int64)
        n = len(X)
        order = 2 if T.shape[1] == 10 else 1
        if order == 2:
            T = _match_mid_nodes(X, T)
        self.order = order
        E, nu = float(youngs_modulus), float(poisson_ratio)
        self.youngs_modulus, self.poisson_ratio = E, nu
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.material = "neo_hookean_solid"

        # Rest geometry: shape function gradients per Gauss point
        points = TET_GAUSS_4 if order == 2 else np.array([[0.25, 0.25, 0.25, 0.25]])
        weights = np.full(len(points), 1.0 / len(points))
        Xe = X[T]                                                          # (T, n, 3)
        dNdX, wdV = [], []
        for L, w in zip(points, weights):
            dL = _shape_gradients_bary(L, order)                          # (n, 4)
            dxi = dL[:, 1:] - dL[:, :1]                                    # d/d(L1, L2, L3), L0 = 1 - sum
            J0 = np.einsum("tai,aj->tij", Xe, dxi)                         # dX/dxi (T, 3, 3)
            det = np.linalg.det(J0)
            if np.any(det <= 0):
                bad = int((det <= 0).sum())
                raise ValueError(f"{bad} inverted or degenerate tetrahedra in the solid mesh.")
            dNdX.append(np.einsum("aj,tji->tai", dxi, np.linalg.inv(J0)))  # (T, n, 3)
            wdV.append(w * det / 6.0)
        dNdX = np.stack(dNdX, axis=1)                                      # (T, q, n, 3)
        wdV = np.stack(wdV, axis=1)                                        # (T, q)

        F = np.asarray(faces, dtype=np.int64)
        tri = X[F]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        nodal_area = np.zeros(n)
        for k in range(3):
            np.add.at(nodal_area, F[:, k], area / 3.0)

        if fixed is None:
            fixed_mask = np.zeros(n, dtype=bool)
        else:
            fixed = np.asarray(fixed)
            fixed_mask = fixed.astype(bool) if fixed.dtype == bool else np.isin(np.arange(n), fixed)

        tt = partial(torch.as_tensor, dtype=DTYPE, device=self.device)
        self.X = tt(X)
        self.x = self.X.clone()
        self.tets = torch.as_tensor(T, device=self.device)
        self.tets_np = T
        self.faces = torch.as_tensor(F, device=self.device)
        self.faces_np = F
        self.fixed = torch.as_tensor(fixed_mask, device=self.device)
        self.fixed_dofs = torch.as_tensor(np.repeat(fixed_mask, 3), device=self.device)
        self.nodal_area = tt(nodal_area)
        self.rest_area = tt(area)
        self.surface_nodes = torch.as_tensor(np.unique(F), device=self.device)
        self.volume_origin = self.X.mean(dim=0)
        self.n_edges = 0
        self.thickness = float(np.cbrt(wdV.sum() / max(len(T), 1)))  # typical element size (reporting only)
        self.rest_volume = float(wdV.sum())
        self.fluid_volumes = []
        self.contact_offset = 0.0
        self.allow_initial_overlap = True  # rest geometry may touch obstacles exactly (faceted CAD)

        comp = torch.arange(3, device=self.device)
        self.tet_dofs = (3 * self.tets[:, :, None] + comp).reshape(len(T), -1)
        self.face_dofs = (3 * self.faces[:, :, None] + comp).reshape(-1, 9)
        self._dNdX, self._wdV = tt(dNdX), tt(wdV)
        energy = partial(_element_energy, mu=self.mu, lam=self.lam)
        self._e, self._g, self._h = vmap(energy), vmap(grad(energy)), vmap(hessian(energy))

    # -----------------------------
    # State (same interface as Shell)
    # -----------------------------

    @property
    def n_nodes(self) -> int:
        return self.X.shape[0]

    @property
    def n_dof(self) -> int:
        return 3 * self.n_nodes

    def get_state(self) -> torch.Tensor:
        return self.x.reshape(-1)

    def set_state(self, u: torch.Tensor):
        self.x = u.reshape(-1, 3)

    def reset(self):
        self.x = self.X.clone()

    def displacement(self) -> torch.Tensor:
        return self.x - self.X

    def element_terms(self, tangent: bool = True, chunk: int = 4000):
        q = self.x[self.tets].reshape(len(self.tets_np), -1)
        for s in range(0, len(q), chunk):
            args = (q[s:s + chunk], self._dNdX[s:s + chunk], self._wdV[s:s + chunk])
            yield (self.tet_dofs[s:s + chunk], self._e(*args), self._g(*args),
                   self._h(*args) if tangent else None)

    def volume_ratio(self, x=None) -> np.ndarray:
        """J = V/V0 per element (at its first Gauss point) for post-processing."""
        x = self.x if x is None else torch.as_tensor(x, dtype=DTYPE, device=self.device)
        F = torch.einsum("tai,taj->tij", x[self.tets], self._dNdX[:, 0])
        return torch.linalg.det(F).cpu().numpy()


def _match_mid_nodes(X, T):
    """Put the 6 mid-edge nodes of every 10-node tetrahedron in TET10_EDGES order, matching each
    to the edge whose midpoint is nearest (robust to the mesher's convention and to curved
    boundary edges, whose mid nodes sit on the CAD surface)."""
    T = T.copy()
    mids = X[T[:, 4:]]                                                               # (T, 6, 3)
    targets = np.stack([0.5 * (X[T[:, i]] + X[T[:, j]]) for i, j in TET10_EDGES], 1)  # (T, 6, 3)
    d = np.linalg.norm(targets[:, :, None] - mids[:, None], axis=3)                  # (T, edge, mid)
    best = d.argmin(axis=2)
    ok = (np.sort(best, axis=1) == np.arange(6)).all(axis=1)                          # a permutation
    T[ok, 4:] = np.take_along_axis(T[ok, 4:], best[ok], axis=1)
    for e in np.nonzero(~ok)[0]:  # ambiguous (strongly curved): greedy on the smallest distances
        order, used, assign = np.argsort(d[e], axis=None), set(), {}
        for flat in order:
            k, m = divmod(int(flat), 6)
            if k not in assign and m not in used:
                assign[k] = m
                used.add(m)
        T[e, 4:] = T[e, 4:][[assign[k] for k in range(6)]]
    return T
