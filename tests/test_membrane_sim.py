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


def _brute_force_signed_distance(vertices, faces, points):
    from membrane_sim.contact import _closest_point_pairs, _winding_number
    V = torch.as_tensor(vertices, dtype=torch.float64)
    tri = V[torch.as_tensor(faces)]
    n, T = len(points), len(tri)
    pi = torch.arange(n).repeat_interleave(T)
    ti = torch.arange(T).repeat(n)
    q, _ = _closest_point_pairs(points[pi], tri[ti, 0], tri[ti, 1], tri[ti, 2])
    dist = ((points[pi] - q) ** 2).sum(-1).reshape(n, T).min(1).values.sqrt()
    inside = _winding_number(points, tri[:, 0], tri[:, 1], tri[:, 2]) > 0.5
    return torch.where(inside, -dist, dist)


@pytest.mark.parametrize("shape", ["prism", "hollow_box"])
def test_signed_distance_matches_brute_force(shape):
    if shape == "prism":
        poly = np.array([[0, 0], [4, 0], [4, 3], [2, 1], [0, 3]], dtype=float)
        V, F = ms.extrude_polygon(poly, 0.0, 2.0)
    else:  # box with an inner cavity, like a CAD frame around fluid chambers
        Vo, Fo = ms.box_mesh((0, 0, 0), (4, 3, 2))
        Vi, Fi = ms.box_mesh((1, 1, 0.5), (3, 2, 1.5))
        V, F = np.vstack([Vo, Vi]), np.vstack([Fo, Fi[:, ::-1] + len(Vo)])
    g = torch.Generator().manual_seed(3)
    points = torch.rand(400, 3, generator=g, dtype=torch.float64) * torch.tensor([5.0, 4.0, 3.0]) - 0.5
    sd, grad = ms.Obstacle(V, F).signed_distance(points, max_distance=10.0)
    reference = _brute_force_signed_distance(V, F, points)
    assert torch.allclose(sd, reference, atol=1e-10)
    assert torch.allclose(grad.norm(dim=1), torch.ones(len(points), dtype=torch.float64))


def test_warm_start_reaches_the_same_state_as_a_fresh_solve():
    def build():
        env = ms.Environment(contact_stiffness=1e3)
        shell = env.add_disk_membrane((0, 0, 0), (0, 0, 1), 1.0, rings=6, thickness=0.05, youngs_modulus=1.0)
        drive = env.add_fluid_volume(P0=0.01)
        gas = env.add_fluid_volume(P0=0.0, pressure_law=lambda dV, P0: (0.1 + P0) * 2.0 / (2.0 + dV) - 0.1,
                                   initial_volume=2.0)
        shell.fluid_volume_contacts((drive, gas))
        env.add_obstacle(*ms.box_mesh((-2, -2, 0.08), (2, 2, 1)))
        return env, shell, drive, gas

    env, shell, drive, gas = build()
    assert env.solve().converged
    drive.P0, gas.P0 = 0.02, 0.001
    assert env.solve(load_steps=2, warm_start=True).converged

    env2, shell2, drive2, gas2 = build()
    drive2.P0, gas2.P0 = 0.02, 0.001
    assert env2.solve().converged
    assert torch.allclose(shell.x, shell2.x, atol=1e-7)
    assert gas.P == pytest.approx(gas2.P, rel=1e-8)


# -----------------------------
# Gas chambers with an incompressible share
# -----------------------------

def _gas_system(liquid_fraction, drive=0.02, P0=0.0, V0=2.0):
    env = ms.Environment()
    shell = env.add_disk_membrane((0, 0, 0), (0, 0, 1), 1.0, rings=6, thickness=0.05, youngs_modulus=1.0)
    inlet = env.add_fluid_volume(P0=drive)
    gas = env.add_fluid_volume(P0=P0, initial_volume=V0, gas_volume=(1 - liquid_fraction) * V0)
    shell.fluid_volume_contacts((inlet, gas))
    return env, shell, gas


def test_gas_chamber_energy_residual_and_tangent_are_consistent():
    env, shell, gas = _gas_system(0.6)
    shell.x = shell.X.clone()
    free = ~shell.fixed
    shell.x[free, 2] += 0.2 * (1 - (shell.X[free, :2] ** 2).sum(1))
    solver = NewtonSolver(env.membrane_list, env.fluid_volume_list, [], 0.0)
    state = solver.evaluate(0.8)
    K = state["K"].toarray() + state["U"] @ np.diag(state["c"]) @ state["U"].T
    u0, h = solver.get_u().clone(), 1e-6
    d = torch.as_tensor(np.random.default_rng(0).standard_normal(solver.n_free))
    out = []
    for sign in (1, -1):
        u = u0.clone()
        u[solver._free_t] += sign * h * d
        solver.set_u(u)
        out.append(solver.evaluate(0.8, tangent=False))
    solver.set_u(u0)
    assert (out[0]["energy"] - out[1]["energy"]) / (2 * h) == pytest.approx(state["residual"] @ d, rel=1e-5)
    dR = (out[0]["residual"] - out[1]["residual"]).numpy() / (2 * h)
    assert np.linalg.norm(dR - K @ d.numpy()) <= 1e-5 * np.linalg.norm(dR)


def test_gas_chamber_follows_boyle_on_the_gas_share_and_liquid_stiffens_it():
    deflection = {}
    for liquid in (0.0, 0.6):
        env, shell, gas = _gas_system(liquid, P0=0.005)
        assert env.solve().converged
        Vg = (1 - liquid) * 2.0
        expected = (0.101325 + 0.005) * Vg / (Vg + gas.delta_volume) - 0.101325
        assert gas.P == pytest.approx(expected, rel=1e-10)
        deflection[liquid] = shell.displacement()[:, 2].max().item()
    assert deflection[0.6] < deflection[0.0]


def test_gas_pocket_can_not_be_squeezed_to_zero():
    env, shell, gas = _gas_system(0.99, drive=5.0, V0=0.1)  # overwhelming drive, tiny gas pocket
    assert env.solve(load_steps=20).converged
    assert gas.gas_volume + gas.delta_volume > 0
    assert gas.P == pytest.approx(5.0, rel=1e-3)  # the gas pressure balances the drive
