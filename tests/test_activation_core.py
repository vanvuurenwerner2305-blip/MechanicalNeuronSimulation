"""Core pieces of the activation-function model: solid tetrahedra, rigid bodies, capped chamber
walls and channel cross-sections."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from membrane_sim.fluid import FluidVolume, boundary_loops  # noqa: E402
from membrane_sim.lumen import ChannelSections  # noqa: E402
from membrane_sim.rigid import RigidBody, rotation_matrix  # noqa: E402
from membrane_sim.solid import TET10_EDGES, Solid  # noqa: E402

CUBE_TETS = np.array([[0, 1, 3, 7], [0, 1, 7, 5], [0, 5, 7, 4], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7]])


def cube(order=1, size=(1.0, 2.0, 0.5)):
    X = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float) * size
    tets = CUBE_TETS.copy()
    for t in tets:  # positive orientation
        if np.linalg.det(np.stack([X[t[1]] - X[t[0]], X[t[2]] - X[t[0]], X[t[3]] - X[t[0]]])) < 0:
            t[[1, 2]] = t[[2, 1]]
    if order == 2:
        mids, rows = {}, []
        V = list(X)
        for t in tets:
            row = list(t)
            for i, j in TET10_EDGES:
                key = tuple(sorted((t[i], t[j])))
                if key not in mids:
                    mids[key] = len(V)
                    V.append(0.5 * (X[t[i]] + X[t[j]]))
                row.append(mids[key])
            rows.append(row)
        X, tets = np.array(V), np.array(rows)
    return X, tets


def neo_hookean_density(F, mu, lam):
    J = np.linalg.det(F)
    return 0.5 * mu * (np.trace(F.T @ F) - 3 - 2 * np.log(J)) + 0.5 * lam * np.log(J) ** 2


@pytest.mark.parametrize("order", [1, 2])
def test_solid_energy_of_a_homogeneous_deformation_is_exact(order):
    X, tets = cube(order)
    solid = Solid(X, tets, np.zeros((0, 3), int), youngs_modulus=2.0, poisson_ratio=0.4)
    F = np.array([[1.2, 0.1, 0.0], [0.05, 0.9, 0.1], [0.0, -0.1, 1.1]])
    solid.set_state(torch.as_tensor(X @ F.T).reshape(-1))
    energy = sum(float(e.sum()) for _, e, _, _ in solid.element_terms(False))
    assert solid.rest_volume == pytest.approx(1.0)
    assert energy == pytest.approx(neo_hookean_density(F, solid.mu, solid.lam), rel=1e-10)


def test_solid_gradient_and_hessian_are_consistent_at_rest_and_deformed():
    X, tets = cube(2)
    solid = Solid(X, tets, np.zeros((0, 3), int), youngs_modulus=1.0, poisson_ratio=0.45)
    rng = np.random.default_rng(1)
    for amplitude in (0.0, 0.05):  # exactly at rest the Hessian must be finite too
        solid.set_state(torch.as_tensor(X + amplitude * rng.standard_normal(X.shape)).reshape(-1))
        for dofs, e, g, H in solid.element_terms(True):
            assert torch.isfinite(H).all()
            q = solid.x[solid.tets].reshape(len(tets), -1)
            v = torch.as_tensor(rng.standard_normal(q.shape))
            h = 1e-6
            gp = solid._g(q + h * v, solid._dNdX, solid._wdV)
            gm = solid._g(q - h * v, solid._dNdX, solid._wdV)
            assert torch.allclose((gp - gm) / (2 * h), torch.einsum("eij,ej->ei", H, v), atol=1e-6)


def test_rotation_matrix_is_smooth_through_zero():
    theta = torch.zeros(3, dtype=torch.float64)
    J = torch.func.jacfwd(rotation_matrix)(theta)
    H = torch.func.hessian(lambda t: rotation_matrix(t)[0, 1])(theta)
    assert torch.isfinite(J).all() and torch.isfinite(H).all()
    t = torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64)
    R = rotation_matrix(t)
    assert torch.allclose(R @ R.T, torch.eye(3, dtype=torch.float64), atol=1e-14)
    assert torch.allclose(R @ t, t)


def test_rigid_body_moves_its_surface():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    body = RigidBody(V, [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    body.set_state(torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, np.pi / 2], dtype=torch.float64))
    x = body.x.numpy()
    c = V.mean(axis=0)
    assert np.allclose(x.mean(axis=0), c + [1, 2, 3])
    assert np.allclose(body.to_body(body.x).numpy(), V)


def tube(n_theta=24, n_z=20, radius=1.0, length=4.0):
    """Open cylinder surface, normals pointing inwards (as a solid tube's inner wall), ends at z=0, length."""
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z = np.linspace(0, length, n_z + 1)
    X = np.array([[radius * np.cos(a), radius * np.sin(a), zz] for zz in z for a in th])
    F = []
    for k in range(n_z):
        for i in range(n_theta):
            a, b = k * n_theta + i, k * n_theta + (i + 1) % n_theta
            c, d = a + n_theta, b + n_theta
            F += [[a, c, b], [b, c, d]]   # normal points to the axis
    F = np.array(F)
    fixed = (X[:, 2] == 0) | (X[:, 2] == length)
    return X, F, fixed


class _Wall:
    """Minimal body for a SurfacePatch."""
    def __init__(self, X, F, fixed):
        self.X = torch.as_tensor(X)
        self.x = self.X.clone()
        self.faces_np = F
        self.fixed = torch.as_tensor(fixed)
        self.device = torch.device("cpu")
        self.fluid_volumes = []


def test_capped_patches_split_a_channel_volume_exactly():
    X, F, fixed = tube()
    wall = _Wall(X, F, fixed)
    centroid_z = X[F].mean(axis=1)[:, 2]
    first, second = np.nonzero(centroid_z < 2.0)[0], np.nonzero(centroid_z >= 2.0)[0]
    whole, a, b = FluidVolume(), FluidVolume(), FluidVolume()
    whole.add_patch(wall, np.arange(len(F)), side=-1)
    a.add_patch(wall, first, side=-1)
    b.add_patch(wall, second, side=-1)
    assert len(a.patches[0].loops) == 1  # the moving loop between the halves; the fixed end is not capped
    rng = np.random.default_rng(0)
    x = X + np.where(fixed[:, None], 0.0, 0.2 * rng.standard_normal(X.shape))
    wall.x = torch.as_tensor(x)
    assert a.compute_delta_volume() + b.compute_delta_volume() == pytest.approx(whole.compute_delta_volume(), rel=1e-12)
    # with free ends (both loops capped), squeezing to 90 % radius removes exactly 19 % of the volume
    free = _Wall(X, F, np.zeros(len(X), bool))
    v = FluidVolume()
    v.add_patch(free, np.arange(len(F)), side=-1)
    free.x = torch.as_tensor(X * [0.9, 0.9, 1.0])
    polygon = 0.5 * 24 * np.sin(2 * np.pi / 24)
    assert v.compute_delta_volume() == pytest.approx(-0.19 * polygon * 4.0, rel=1e-12)


def test_patch_volume_gradient_matches_finite_differences():
    X, F, fixed = tube(12, 6)
    wall = _Wall(X, F, fixed)
    v = FluidVolume()
    patch = v.add_patch(wall, np.nonzero(X[F].mean(axis=1)[:, 2] < 2.0)[0], side=-1)
    rng = np.random.default_rng(3)
    wall.x = torch.as_tensor(X + np.where(fixed[:, None], 0.0, 0.1 * rng.standard_normal(X.shape)))
    g = torch.zeros(3 * len(X), dtype=torch.float64)
    for _, dofs, gl, _ in v.volume_terms(False):
        g.index_add_(0, dofs.reshape(-1), gl.reshape(-1))
    d = torch.as_tensor(rng.standard_normal(3 * len(X)))
    x0, h = wall.x.clone(), 1e-6
    wall.x = x0 + h * d.reshape(-1, 3)
    vp = v.compute_delta_volume()
    wall.x = x0 - h * d.reshape(-1, 3)
    vm = v.compute_delta_volume()
    assert (vp - vm) / (2 * h) == pytest.approx(float(g @ d), rel=1e-7)


def test_boundary_loops_of_a_strip():
    X, F, _ = tube(8, 2)
    loops = boundary_loops(F)
    assert sorted(len(l) for l in loops) == [8, 8]


def test_channel_sections_measure_the_area_and_follow_the_deformation():
    X, F, fixed = tube(48, 10, radius=1.0)
    sections = ChannelSections(X, F, axis=[0, 0, 1], n_stations=9)
    polygon = 0.5 * 48 * np.sin(2 * np.pi / 48)
    assert np.allclose(sections.rest_areas, polygon, rtol=1e-9)
    squeezed = X * [1.0, 0.25, 1.0]  # flattened to an ellipse
    assert np.allclose(sections.profile(squeezed), 0.25 * polygon, rtol=1e-9)
    inverted = X * [1.0, -0.2, 1.0]  # walls pushed through each other: clipped at 0
    assert sections.minimum(inverted) == 0.0


def _boundary(tets, X):
    """Outward boundary triangles of a TET4 mesh."""
    from collections import Counter
    faces = []
    for t in tets:
        for f in ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)):
            tri = t[list(f)]
            faces.append(tri)
    count = Counter(tuple(sorted(f)) for f in faces)
    out = [f for f in faces if count[tuple(sorted(f))] == 1]
    out = np.array(out)
    for f in out:  # orient outward from the centroid of the (convex) body
        n = np.cross(X[f[1]] - X[f[0]], X[f[2]] - X[f[0]])
        if n @ (X[f].mean(axis=0) - X.mean(axis=0)) < 0:
            f[[1, 2]] = f[[2, 1]]
    return out


class _Solver:
    def __init__(self, bodies):
        self.dof_offset, off = {}, 0
        for b in bodies:
            self.dof_offset[id(b)] = off
            off += b.n_dof
        self.n = off


def _check_coupling(coupling, bodies, states):
    solver = _Solver(bodies)
    coupling.bind(solver)

    def total(u):
        for b in bodies:
            o = solver.dof_offset[id(b)]
            b.set_state(u[o:o + b.n_dof])
        e = 0.0
        g = torch.zeros(solver.n, dtype=torch.float64)
        H = torch.zeros(solver.n, solver.n, dtype=torch.float64)
        for dofs, en, gr, he in coupling.terms(True):
            e += float(en.sum())
            g.index_add_(0, dofs.reshape(-1), gr.reshape(-1))
            for k in range(len(dofs)):
                H[dofs[k][:, None], dofs[k][None, :]] += he[k]
        return e, g, H

    u = torch.cat([s.reshape(-1) for s in states])
    e, g, H = total(u)
    assert e > 0
    rng = np.random.default_rng(5)
    v = torch.as_tensor(rng.standard_normal(len(u)))
    h = 1e-6
    ep, gp, _ = total(u + h * v)
    em, gm, _ = total(u - h * v)
    assert (ep - em) / (2 * h) == pytest.approx(float(g @ v), rel=1e-6)
    assert torch.allclose((gp - gm) / (2 * h), H @ v, rtol=1e-5, atol=1e-8)


def test_solid_contact_is_consistent_and_neutral_at_rest():
    from membrane_sim.solid_contact import SolidContact
    X, tets = cube(1, size=(1.0, 1.0, 1.0))
    top = X + [0.1, 0.2, 1.0]  # a cube resting on the other one
    A = Solid(top, tets, _boundary(tets, top), 1.0)
    B = Solid(X, tets, _boundary(tets, X), 1.0)
    contact = SolidContact(A, B, stiffness=10.0, reach=0.5)
    contact.bind(_Solver([A, B]))
    assert contact.terms(True) == []            # touching at rest: no force
    pushed = torch.as_tensor(top - [0, 0, 0.05])  # 0.05 into the lower cube
    _check_coupling(contact, [A, B], [pushed, torch.as_tensor(X)])


def test_membrane_bonded_to_a_solid_follows_it():
    from membrane_sim.solid_contact import SurfaceTie
    X, tets = cube(1, size=(1.0, 1.0, 1.0))
    solid = Solid(X, tets, _boundary(tets, X), 1.0)
    import membrane_sim as ms  # noqa: F811
    V, F = ms.rectangle_mesh([0, 0, 1.25], [1, 0, 0], [0, 1, 0], 4, 4)
    sheet = ms.Shell(V, F, thickness=0.5, youngs_modulus=1.0)
    tie = SurfaceTie(sheet, np.arange(len(V)), solid, stiffness=10.0, max_distance=0.3)
    assert len(tie.nodes) == len(V)
    tie.bind(_Solver([sheet, solid]))
    assert float(sum(e.sum() for _, e, _, _ in tie.terms(False))) == pytest.approx(0.0, abs=1e-20)
    moved = torch.as_tensor(X + [0.0, 0.0, 0.1])
    _check_coupling(tie, [sheet, solid], [torch.cat([sheet.X.reshape(-1), sheet.psi]), moved])
