import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import membrane_sim as ms  # noqa: E402
from membrane_sim.solver import NewtonSolver  # noqa: E402


# -----------------------------
# Consistency of the Newton system
# -----------------------------

def _perturbed_system():
    env = ms.Environment(contact_stiffness=50.0)
    shell = env.add_disk_membrane((0, 0, 0), (0, 0, 1), 1.0, rings=3, thickness=0.05, youngs_modulus=1.0)
    below = env.add_fluid_volume(P0=0.02, bulk_stiffness=0.3)
    shell.fluid_volume_contacts((below, None))
    env.add_obstacle(*ms.box_mesh((-2, -2, 0.1), (2, 2, 1)))

    g = torch.Generator().manual_seed(0)
    free = ~shell.fixed
    bump = 0.3 * (1 - (shell.X[:, :2] ** 2).sum(1))
    shell.x = shell.X.clone()
    shell.x[free, 2] += bump[free]
    shell.x[free] += 0.02 * torch.randn(int(free.sum()), 3, generator=g, dtype=torch.float64)
    shell.psi = 0.05 * torch.randn(shell.n_edges, generator=g, dtype=torch.float64)
    solver = NewtonSolver(env.membrane_list, env.fluid_volume_list, env.obstacle_list, contact_stiffness=50.0)
    return solver


def test_residual_is_energy_gradient_and_tangent_is_its_jacobian():
    solver = _perturbed_system()
    lam = 0.7
    state = solver.evaluate(lam)
    assert 0 < solver.evaluate(lam, tangent=False)["residual"].norm()
    K = state["K"].toarray() + state["U"] @ np.diag(state["c"]) @ state["U"].T
    R0 = state["residual"].numpy()
    u0 = solver.get_u().clone()

    rng = np.random.default_rng(1)
    for _ in range(3):
        direction = rng.standard_normal(solver.n_free)
        h = 1e-6
        values = []
        for sign in (+1, -1):
            u = u0.clone()
            u[solver._free_t] += sign * h * torch.as_tensor(direction)
            solver.set_u(u)
            values.append(solver.evaluate(lam, tangent=False))
        solver.set_u(u0)

        dE = (values[0]["energy"] - values[1]["energy"]) / (2 * h)
        assert dE == pytest.approx(R0 @ direction, rel=1e-5, abs=1e-9)

        dR = (values[0]["residual"] - values[1]["residual"]).numpy() / (2 * h)
        assert np.linalg.norm(dR - K @ direction) <= 1e-5 * np.linalg.norm(dR)


def test_contact_is_active_in_consistency_check():
    solver = _perturbed_system()
    shell = solver.shells[0]
    assert (shell.x[:, 2] > 0.1).any()


# -----------------------------
# Analytical benchmarks
# -----------------------------

def _pressurised(env, shell, pressure):
    volume = env.add_fluid_volume(P0=pressure)
    shell.fluid_volume_contacts((volume, None))
    return volume


def test_clamped_square_plate_matches_kirchhoff_solution():
    env = ms.Environment()
    E, t, nu, a = 1e6, 0.01, 0.3, 1.0
    shell = env.add_rectangular_membrane((0, 0, 0), (a, 0, 0), (0, a, 0), divisions=(32, 32), thickness=t,
                                         youngs_modulus=E, poisson_ratio=nu, material="svk")
    D = shell.bending_stiffness
    q = 1e-5 * D / 0.00126  # centre deflection t/1000: linear regime
    _pressurised(env, shell, q)
    assert env.solve(load_steps=1).converged
    w = shell.displacement()[:, 2].max().item()
    assert w == pytest.approx(0.00126 * q * a ** 4 / D, rel=0.04)


def test_clamped_circular_plate_matches_kirchhoff_solution():
    env = ms.Environment()
    E, t, nu, a = 1e6, 0.01, 0.3, 1.0
    shell = env.add_disk_membrane((0, 0, 0), (0, 0, 1), a, rings=24, thickness=t,
                                  youngs_modulus=E, poisson_ratio=nu, material="svk")
    D = shell.bending_stiffness
    q = 1e-5 * 64 * D / a ** 4
    _pressurised(env, shell, q)
    assert env.solve(load_steps=1).converged
    w = shell.displacement()[:, 2].max().item()
    assert w == pytest.approx(q * a ** 4 / (64 * D), rel=0.02)


def test_bending_energy_is_independent_of_mesh_orientation():
    energies = []
    for angle in (0.0, 30.0, 45.0):
        th = np.radians(angle)
        rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        V, F = ms.rectangle_mesh((-0.5, -0.5, 0), (1, 0, 0), (0, 1, 0), 24, 24)
        V = V @ rot.T
        shell = ms.Shell(V, F, 0.01, 1e6, 0.3, material="svk", fixed=np.ones(len(V), bool), boundary_rotation="free")
        R = 2.0
        X = V.copy()
        X[:, 0], X[:, 2] = R * np.sin(V[:, 0] / R), R * (1 - np.cos(V[:, 0] / R))
        shell.x = torch.tensor(X)
        assert NewtonSolver([shell], [], [], 0.0).newton(1.0)[0]  # relax the edge rotations
        bending = list(shell.element_terms(False))[1][1].sum().item()
        energies.append(bending / (shell.bending_stiffness / (2 * R ** 2)))
    assert max(energies) - min(energies) < 0.02
    assert energies[0] == pytest.approx(1.0, abs=0.07)  # remaining gap: O(h) free-edge boundary layer


def test_membrane_inflation_matches_hencky_solution():
    env = ms.Environment()
    E, t, a = 1.0, 0.01, 1.0
    shell = env.add_disk_membrane((0, 0, 0), (0, 0, 1), a, rings=12, thickness=t, youngs_modulus=E,
                                  poisson_ratio=0.3, material="svk", bending=False)
    q = 1e-3 * E * t / a
    _pressurised(env, shell, q)
    assert env.solve(load_steps=5).converged
    w = shell.displacement()[:, 2].max().item()
    assert w / a / (q * a / (E * t)) ** (1 / 3) == pytest.approx(0.655, rel=0.02)


# -----------------------------
# Fluid volumes
# -----------------------------

def test_delta_volume_of_imposed_paraboloid():
    env = ms.Environment()
    shell = env.add_disk_membrane((3, -1, 2), (0, 0, 1), 1.0, rings=30, thickness=0.01, youngs_modulus=1.0)
    volume = env.add_fluid_volume()
    shell.fluid_volume_contacts((volume, None))
    w0 = 0.2
    r2 = ((shell.X[:, :2] - torch.tensor([3.0, -1.0], dtype=torch.float64)) ** 2).sum(1)
    shell.x = shell.X.clone()
    shell.x[:, 2] += w0 * (1 - r2)
    assert volume.compute_delta_volume() == pytest.approx(np.pi * w0 / 2, rel=5e-3)


def test_volume_stiff_chamber_equals_constant_pressure_at_same_final_pressure():
    def build(P0, K):
        env = ms.Environment()
        shell = env.add_disk_membrane((0, 0, 0), (0, 0, 1), 1.0, rings=6, thickness=0.05, youngs_modulus=1.0)
        chamber = env.add_fluid_volume(P0=P0, bulk_stiffness=K)
        shell.fluid_volume_contacts((chamber, None))
        return env, shell, chamber

    env, shell, chamber = build(0.02, 0.01)
    assert env.solve().converged
    assert chamber.P == pytest.approx(0.02 - 0.01 * chamber.delta_volume, rel=1e-12)
    assert chamber.P < 0.02

    env2, shell2, _ = build(chamber.P, 0.0)
    assert env2.solve().converged
    assert torch.allclose(shell.x, shell2.x, atol=1e-7)


def test_custom_pressure_law_matches_equivalent_linear_law():
    results = []
    for law in (None, lambda dV: 0.02 - 0.01 * dV):
        env = ms.Environment()
        shell = env.add_disk_membrane((0, 0, 0), (0, 0, 1), 1.0, rings=6, thickness=0.05, youngs_modulus=1.0)
        chamber = env.add_fluid_volume(P0=0.02, bulk_stiffness=0.01, pressure_law=law)
        shell.fluid_volume_contacts((chamber, None))
        assert env.solve().converged
        results.append(shell.x.clone())
    assert torch.allclose(*results, atol=1e-9)


# -----------------------------
# Contact
# -----------------------------

def test_membrane_inflates_against_flat_obstacle():
    gap, k = 0.05, 1e3
    env = ms.Environment(contact_stiffness=k)
    shell = env.add_disk_membrane((0, 0, 0), (0, 0, 1), 1.0, rings=8, thickness=0.05, youngs_modulus=1.0)
    q = 0.01
    _pressurised(env, shell, q)
    env.add_obstacle(*ms.box_mesh((-2, -2, gap), (2, 2, 1)))
    assert env.solve().converged
    z = shell.x[:, 2]
    assert z.max().item() > gap                    # in contact ...
    assert z.max().item() < gap + 2 * q / k        # ... with penetration ~ pressure / stiffness


def test_inverted_obstacle_signed_distance():
    container = ms.Obstacle(*ms.box_mesh((0, 0, 0), (1, 1, 1)), inverted=True)
    pts = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.9], [0.5, 0.5, 1.2]], dtype=torch.float64)
    sd, grad = container.signed_distance(pts, max_distance=1.0)
    assert sd.tolist() == pytest.approx([0.5, 0.1, -0.2])
    assert grad[1].tolist() == pytest.approx([0, 0, -1])


def test_extruded_nonconvex_polygon_is_closed_and_consistent():
    poly = np.array([[0, 0], [2, 0], [2, 2], [1, 1], [0, 2]], dtype=float)  # arrow head, area 3
    V, F = ms.extrude_polygon(poly[::-1], -0.5, 0.5)  # clockwise input gets reoriented
    assert ms.closed_mesh_volume(V, F) == pytest.approx(3.0)
    obstacle = ms.Obstacle(V, F)
    sd, _ = obstacle.signed_distance(torch.tensor([[1.0, 0.5, 0.0], [1.0, 1.5, 0.0]], dtype=torch.float64), 1.0)
    assert sd[0] < 0 < sd[1]
