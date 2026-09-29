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


def test_cached_contact_queries_match_the_full_search():
    from membrane_sim.contact import CachedSignedDistance, ObstacleField
    from membrane_sim.shell_contact import ShellContact, pairs_within
    g = torch.Generator().manual_seed(4)

    field = ObstacleField([ms.Obstacle(*ms.box_mesh((0, 0, 0), (1, 1, 1))),
                           ms.Obstacle(*ms.box_mesh((-2, -2, -2), (3, 3, 3)), inverted=True)])
    points = torch.rand(200, 3, generator=g, dtype=torch.float64) * 1.6 - 0.3
    cache = CachedSignedDistance(field, 0.2, skin=0.1)
    for k in range(20):  # small moves reuse the lists, every 5th move forces a rebuild
        points = points + torch.randn(points.shape, generator=g, dtype=torch.float64) * (0.3 if k % 5 == 0 else 0.01)
        sd, grad, H = cache.signed_distance(points, hessian=True)
        ref_sd, ref_grad, ref_H = field.signed_distance(points, 0.2, hessian=True)
        near = ref_sd < 0.2
        assert torch.equal(sd < 0.2, near)
        assert torch.equal(sd[near], ref_sd[near]) and torch.equal(grad[near], ref_grad[near])
        assert torch.equal(H[near], ref_H[near])
    assert cache.rebuilds < 20

    env = ms.Environment()
    lower = env.add_rectangular_membrane((0, 0, 0), (1, 0, 0), (0, 1, 0), divisions=(10, 10),
                                         thickness=0.05, youngs_modulus=1.0)
    upper = env.add_rectangular_membrane((0, 0, 0.08), (1, 0, 0), (0, 1, 0), divisions=(9, 9),
                                         thickness=0.05, youngs_modulus=1.0)
    contact = ShellContact([lower, upper], 1.0, search_distance=0.1)
    for k in range(12):
        for s in (lower, upper):
            s.x = s.x + torch.randn(s.x.shape, generator=g, dtype=torch.float64) * 0.004 * (~s.fixed)[:, None]
        for pair in contact.pairs:
            radius = pair["h_max"] + contact.search_distance
            pi, t, _, _ = contact._pairs_within(pair, radius)
            ref_pi, ref_t, _, _ = pairs_within(pair["A"].x[pair["nodes"]], pair["B"].x, pair["B"].faces, radius)
            assert sorted(zip(pi.tolist(), t.tolist())) == sorted(zip(ref_pi.tolist(), ref_t.tolist()))
        step = {id(s): torch.randn(s.x.shape, generator=g, dtype=torch.float64) * 0.05 * (~s.fixed)[:, None]
                for s in (lower, upper)}
        reference = 1.0
        for pair in contact.pairs:  # the unfiltered bound: every pair within the search radius
            A, B, nodes = pair["A"], pair["B"], pair["nodes"]
            pi, t, d, _ = pairs_within(A.x[nodes], B.x, B.faces, pair["h_max"] + contact.search_distance)
            closing = step[id(A)][nodes[pi]].norm(dim=1) + step[id(B)][B.faces[t]].norm(dim=2).max(dim=1).values
            reference = min(reference, (0.9 * d / closing).min().item())
        assert contact.safe_step(step) == pytest.approx(reference, rel=1e-12)


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



def test_part_filled_liquid_chamber_law_is_consistent():
    V = ms.FluidVolume(P0=0.01, bulk_stiffness=3.0, initial_volume=10.0, liquid_volume=6.0)
    assert V.pressure(0.0) == pytest.approx(0.01 - 3.0 * 4.0)   # 4 mm3 too little liquid: suction
    assert V.pressure(-4.0) == pytest.approx(0.01)              # shrunk to the liquid volume
    h = 1e-6
    for dV in (-5.0, -4.0, -1.0, 0.5):
        assert (V.pressure_potential(dV + h) - V.pressure_potential(dV - h)) / (2 * h) ==             pytest.approx(V.pressure(dV), rel=1e-7)
        assert V.pressure_slope(dV) == -3.0


def test_liquid_chamber_is_pulled_to_its_fluid_volume():
    def solve(liquid=None, V0=1.0, drive=0.0):
        env = ms.Environment()
        shell = env.add_disk_membrane((0, 0, 0), (0, 0, 1), 1.0, rings=6, thickness=0.05, youngs_modulus=1.0)
        outside = env.add_fluid_volume(P0=drive)
        kw = dict() if liquid is None else dict(bulk_stiffness=100.0, liquid_volume=liquid)
        chamber = env.add_fluid_volume(initial_volume=V0, **kw)
        shell.fluid_volume_contacts((outside, chamber))
        assert env.solve().converged
        chamber.start = env.history[0]['pressures'][1]
        return chamber
    stroke = -solve(drive=0.02).delta_volume   # volume scale the membrane sweeps at 20 kPa
    assert stroke > 0
    under = solve(liquid=1.0 - 0.5 * stroke)   # too little liquid: sucks the membrane in
    assert under.start == pytest.approx(-100.0 * 0.5 * stroke)  # step 0: full suction, nothing moved yet
    assert under.P < 0 and under.delta_volume == pytest.approx(-0.5 * stroke, rel=0.02)
    over = solve(liquid=1.0 + 0.5 * stroke)    # too much liquid: inflates it
    assert over.P > 0 and over.delta_volume == pytest.approx(0.5 * stroke, rel=0.02)

# -----------------------------
# Contact between membranes
# -----------------------------

def _below(lower, upper):
    """True if every free node of `lower` is on the -normal side of `upper` (normals are +z)."""
    from membrane_sim.shell_contact import _face_normals, closest_on_mesh
    points = lower.x[~lower.fixed]
    _, q, t = closest_on_mesh(points, upper.x, upper.faces, 10.0)
    return bool((((points - q) * _face_normals(upper.x, upper.faces)[t]).sum(-1) < 0).all())


def _stacked_sheets(gap=0.2, pressure=0.05, k=1e3, with_upper=True):
    env = ms.Environment(contact_stiffness=k)
    lower = env.add_rectangular_membrane((0, 0, 0), (1, 0, 0), (0, 1, 0), divisions=(10, 10), thickness=0.05,
                                         youngs_modulus=1.0, contact_offset=0.025, name="lower")
    upper = None
    if with_upper:  # a different mesh, so nodes do not sit exactly above the other sheet's vertices
        upper = env.add_rectangular_membrane((0, 0, gap), (1, 0, 0), (0, 1, 0), divisions=(7, 9), thickness=0.05,
                                             youngs_modulus=1.0, contact_offset=0.025, name="upper")
    drive = env.add_fluid_volume(P0=pressure)
    lower.fluid_volume_contacts((drive, None))  # pushes the lower sheet up (+z) into the upper one
    return env, lower, upper


def test_sheet_contact_energy_residual_and_tangent_are_consistent():
    from membrane_sim.shell_contact import closest_on_mesh
    env, lower, upper = _stacked_sheets()
    free = ~lower.fixed
    lower.x = lower.X.clone()
    lower.x[free, 2] += 0.17 * torch.sin(np.pi * lower.X[free, 0]) * torch.sin(np.pi * lower.X[free, 1])
    # break the symmetry: a node exactly over a triangle edge sits where the pair energy is only C1
    lower.x[free] += 1e-3 * torch.randn(int(free.sum()), 3, generator=torch.Generator().manual_seed(1),
                                        dtype=torch.float64)
    solver = NewtonSolver(env.membrane_list, env.fluid_volume_list, [], 1e3)
    d, _, _ = closest_on_mesh(lower.x[free], upper.x, upper.faces, 1.0)
    assert (d < 0.05).any()  # contact is active
    state = solver.evaluate(1.0)
    K = state["K"].toarray() + state["U"] @ np.diag(state["c"]) @ state["U"].T
    u0, h = solver.get_u().clone(), 1e-7
    direction = torch.as_tensor(np.random.default_rng(2).standard_normal(solver.n_free))
    out = []
    for sign in (1, -1):
        u = u0.clone()
        u[solver._free_t] += sign * h * direction
        solver.set_u(u)
        out.append(solver.evaluate(1.0, tangent=False))
    solver.set_u(u0)
    assert (out[0]["energy"] - out[1]["energy"]) / (2 * h) == pytest.approx(state["residual"] @ direction, rel=1e-5)
    dR = (out[0]["residual"] - out[1]["residual"]).numpy() / (2 * h)
    assert np.linalg.norm(dR - K @ direction.numpy()) <= 1e-4 * np.linalg.norm(dR)


def test_inflating_sheet_pushes_the_sheet_above_without_passing_through():
    from membrane_sim.shell_contact import closest_on_mesh
    env, lower, _ = _stacked_sheets(with_upper=False)
    assert env.solve().converged
    free_rise = lower.displacement()[:, 2].max().item()
    assert free_rise > 0.3  # alone, the lower sheet would pass the upper one's position

    env, lower, upper = _stacked_sheets()
    assert env.solve().converged
    assert upper.displacement()[:, 2].max().item() > 0.05          # the upper sheet is lifted
    assert lower.displacement()[:, 2].max().item() < free_rise
    d, _, _ = closest_on_mesh(lower.x[~lower.fixed], upper.x, upper.faces, 1.0)
    assert d.min().item() > 0.05 - 0.02                            # separation ~ thickness, small penetration
    assert _below(lower, upper)                                    # never crossed


def test_sheets_do_not_cross_under_a_hard_push():
    env, lower, upper = _stacked_sheets(gap=0.1, pressure=0.5, k=50.0)  # soft penalty, strong push
    result = env.solve()
    assert result.converged
    assert _below(lower, upper)


# -----------------------------
# Mechanical weights of input paths (membrane_sim.characterise)
# -----------------------------

def _disk(env, x, z=0.0, radius=1.0, name=None):
    return env.add_disk_membrane((x, 0, z), (0, 0, 1), radius, rings=6, thickness=0.05, youngs_modulus=1.0, name=name)


def test_direct_paths_rebuild_the_activation_pressure():
    env = ms.Environment()
    a = env.add_fluid_volume(P0=0.002, bulk_stiffness=0.05, name="act")
    for x, r, p, name in ((0, 1.0, 0.02, "in1"), (3, 0.8, 0.008, "in2")):
        _disk(env, x, radius=r, name=f"m_{name}").fluid_volume_contacts((env.add_fluid_volume(P0=p, name=name), a))
    assert env.solve().converged
    w = ms.input_weights(env, a)
    by = {i["input"]: i for i in w["inputs"]}
    assert set(by) == {"in1", "in2"} and by["in1"]["shells"] == ["m_in1"]
    for name, p in (("in1", 0.02), ("in2", 0.008)):
        assert by[name]["dp"] == pytest.approx(p - a.P, rel=1e-12) and by[name]["W"] > 0
    assert sum(i["dV"] for i in w["inputs"]) == pytest.approx(w["chamber"]["dV"])  # volume balance
    assert w["chamber"]["W"] == pytest.approx(1 / 0.05) and w["chamber"]["p"] == pytest.approx(0.002)
    assert ms.rebuild_activation_pressure(w) == pytest.approx(a.P, rel=1e-9)


def test_bulk_modulus_path_is_one_weight_and_stiffer_fluid_transfers_more():
    def solve(liquid_fraction):
        env = ms.Environment()
        a = env.add_fluid_volume(P0=0.0, bulk_stiffness=0.02, name="act")
        inp = env.add_fluid_volume(P0=0.01, name="input")
        weight = env.add_fluid_volume(P0=0.0, initial_volume=2.0, gas_volume=(1 - liquid_fraction) * 2.0,
                                      name="weight")
        _disk(env, 0, 0.0, name="outer").fluid_volume_contacts((inp, weight))
        _disk(env, 0, 0.6, name="inner").fluid_volume_contacts((weight, a))
        _disk(env, 4, 0.0, name="bias_m").fluid_volume_contacts((env.add_fluid_volume(P0=0.004, name="bias"), a))
        assert env.solve().converged
        w = ms.input_weights(env, a)
        assert ms.rebuild_activation_pressure(w) == pytest.approx(a.P, rel=1e-9)
        return {i["input"]: i for i in w["inputs"]}

    soft, stiff = solve(0.0), solve(0.9)
    assert set(soft) == {"input", "bias"}
    assert soft["input"]["shells"] == ["inner"]  # the weight chamber and outer membrane are inside the path
    assert stiff["input"]["W"] > soft["input"]["W"] > 0


def test_membrane_to_the_surroundings_is_an_ambient_input():
    env = ms.Environment()
    a = env.add_fluid_volume(P0=0.01, bulk_stiffness=0.05, name="act")
    _disk(env, 0, name="window").fluid_volume_contacts((None, a))  # nothing behind: 0 gauge
    assert env.solve().converged
    (path,) = ms.input_weights(env, a)["inputs"]
    assert path["input"] == "ambient" and path["p"] == 0.0 and path["dp"] == pytest.approx(-a.P)
    assert path["W"] > 0


def test_tangent_weight_matches_finite_difference():
    def solve(p):
        env = ms.Environment()
        a = env.add_fluid_volume(P0=0.002, name="act")  # held at a fixed pressure
        _disk(env, 0).fluid_volume_contacts((env.add_fluid_volume(P0=p, name="in"), a))
        assert env.solve(rtol=1e-11).converged
        return ms.input_weights(env, a)["inputs"][0]

    p, h = 0.01, 1e-5
    w = solve(p)
    fd = (solve(p + h)["dV"] - solve(p - h)["dV"]) / (2 * h)
    assert w["W_tan"] == pytest.approx(fd, rel=1e-4)
    assert w["W_tan"] < w["W"]  # a membrane stiffens as it inflates


def _synthetic_samples(w1=([2.0, 0.5], [2.0, 0.5])):
    """A neuron with W1(dp) given per side (dp > 0, dp < 0), W2 = 3 and W0 = 1 about p0 = 0, sampled on an
    input grid that gives dp1 both signs."""
    samples = []
    for p1 in np.linspace(0, 20, 6):
        for p2 in (0.0, 10.0):
            terms = [(p1, w1, False), (p2, [3.0], False), (0.0, [1.0], True)]
            pa = ms.solve_activation(terms)
            samples.append({"p_a": pa, "terms": {"in1": (p1, p1 - pa, float(ms.evaluate_weight(w1, p1 - pa))),
                                                 "in2": (p2, p2 - pa, 3.0), "W0": (0.0, pa, 1.0)}})
    return samples


def _degrees(result):
    return {k: {side: f["degree"] for side, f in w["sides"].items() if f is not None}
            for k, w in result["fits"].items()}


def test_solve_activation_is_the_weighted_average_for_constant_weights():
    pa = ms.solve_activation([(10.0, [2.0], False), (4.0, [1.0], False), (0.0, [1.0], True)])
    assert pa == pytest.approx((2 * 10 + 1 * 4 + 0) / 4)
    # a piecewise weight uses the side of its own dp: dp1 = 10 - pa > 0 picks 2, not 7
    pa = ms.solve_activation([(10.0, ([2.0], [7.0]), False), (4.0, [1.0], False), (0.0, [1.0], True)])
    assert pa == pytest.approx((2 * 10 + 1 * 4 + 0) / 4)


def test_equation_fit_uses_the_lowest_degrees_that_meet_the_tolerance():
    samples = _synthetic_samples()
    exact = ms.fit_neuron_equation(samples, tolerance=1e-6)
    assert exact["met"] and exact["error"] < 1e-6
    assert _degrees(exact) == {"in1": {"+": 1, "-": 1}, "in2": {"+": 0, "-": 0}, "W0": {"+": 0}}
    for side in "+-":
        assert exact["fits"]["in1"]["sides"][side]["coefficients"] == pytest.approx([2.0, 0.5], rel=1e-8)
    loose = ms.fit_neuron_equation(samples, tolerance=100.0)
    assert all(d == 0 for w in _degrees(loose).values() for d in w.values())  # constants are enough
    assert loose["error"] == pytest.approx(max(loose["errors"]))


def test_each_sign_of_dp_gets_its_own_polynomial():
    # W1 jumps from 5 (dp < 0) to 2 (dp > 0): one constant per side is exact, one polynomial is not
    samples = _synthetic_samples(w1=([2.0], [5.0]))
    result = ms.fit_neuron_equation(samples, tolerance=1e-9)
    assert result["met"] and _degrees(result)["in1"] == {"+": 0, "-": 0}
    assert result["fits"]["in1"]["sides"]["+"]["coefficients"] == pytest.approx([2.0])
    assert result["fits"]["in1"]["sides"]["-"]["coefficients"] == pytest.approx([5.0])
    assert set(result["weight_errors"]["in1"]) == {"+", "-"}


def test_lowest_total_order_is_never_above_the_greedy_rule():
    for samples in (_synthetic_samples(), _synthetic_samples(w1=([2.0, 0.3, 0.01], [5.0, -0.2]))):
        for tolerance in (1e-6, 1e-3, 0.1, 1.0):
            lowest = ms.fit_neuron_equation(samples, tolerance, ms.LOWEST_TOTAL)
            greedy = ms.fit_neuron_equation(samples, tolerance, ms.BIGGEST_ERROR)
            assert lowest["met"] and greedy["met"]
            order = lambda r: sum(ms.weight_degree(f) for f in r["fits"].values())
            assert order(lowest) <= order(greedy)


def test_equation_fit_stops_at_its_budget_and_can_be_cancelled():
    # an unreachable tolerance used to try every degree combination (hours on a real sweep)
    samples = _synthetic_samples(w1=([2.0, 0.3, 0.01], [5.0, -0.2]))
    noise = iter(1 + 0.01 * np.random.default_rng(0).standard_normal(3 * len(samples)))
    for s in samples:  # like solver output: no piece is ever fitted exactly, so the degree caps stay high
        s["terms"] = {k: (p, dp, W * next(noise)) for k, (p, dp, W) in s["terms"].items()}
    result = ms.fit_neuron_equation(samples, tolerance=0.0, max_evaluations=25)
    assert result["stopped"] and not result["met"] and result["evaluated"] == 25
    assert result["error"] == pytest.approx(max(result["errors"]))  # the best combination found is returned
    assert not ms.fit_neuron_equation(samples, tolerance=1.0)["stopped"]

    class Cancelled(Exception):
        pass
    calls = []

    def check():
        calls.append(1)
        if len(calls) > 3:
            raise Cancelled()
    with pytest.raises(Cancelled):
        ms.fit_neuron_equation(samples, tolerance=0.0, check=check)


def test_sensitivity_is_undefined_where_the_weighted_average_model_breaks():
    # Eq. 4.10: dp_a/dW_k = (p_k - p_a) / sum W
    S = ms.activation_sensitivities({"in": (4.0, 2.0), "W0": (1.0, 2.0)})
    assert S["in"] == pytest.approx(1.0) and S["W0"] == pytest.approx(-0.25)
    # NeuronTest2 at Input1 = 0: a pre-pressurised weight chamber forces secant weights that sum to ~0,
    # one of them negative; that state must not dominate the fit
    S = ms.activation_sensitivities({"Input1": (-0.035, -152.6), "Input2": (-0.035, 109.67),
                                     "ambient": (-0.035, 42.63), "W0": (0.035, 0.3)})
    assert all(np.isnan(v) for v in S.values())


def test_measured_bias_enters_the_equation():
    # path "in1" pushes b = 4 mm3 at zero pressure difference (e.g. a pre-pressurised weight chamber):
    # dV1 = b + 2 dp1; W2 = 3; W0 = 1. At equal pressures (all 0) p_a = b / sum W = 4 / 6, not 0.
    b = 4.0
    samples = []
    for p1 in np.linspace(-10, 10, 7):
        for p2 in (0.0, 5.0):
            pa = ms.solve_activation([(p1, [2.0], False), (p2, [3.0], False), (0.0, [1.0], True)], bias=b)
            assert 2.0 * (p1 - pa) + 3.0 * (p2 - pa) + (0.0 - pa) + b == pytest.approx(0.0, abs=1e-9)
            dp1 = p1 - pa
            samples.append({"p_a": pa, "terms": {"in1": (p1, dp1, (b + 2.0 * dp1) / dp1),  # secant incl. bias
                                                 "in2": (p2, p2 - pa, 3.0), "W0": (0.0, pa, 1.0)},
                            "volumes": {"in1": b + 2.0 * dp1}})
    fit = ms.fit_neuron_equation(samples, tolerance=1e-9, bias={"in1": b})
    assert fit["met"] and fit["bias"] == pytest.approx(b) and fit["fits"]["in1"]["bias"] == pytest.approx(b)
    assert all(d == 0 for w in _degrees(fit).values() for d in w.values())  # W1 = 2 exactly once b is out
    assert fit["fits"]["in1"]["sides"]["+"]["coefficients"] == pytest.approx([2.0])
    latex = ms.equation_align(ms.neuron_equation_latex([("in1", fit["fits"]["in1"]), ("in2", fit["fits"]["in2"])]))
    assert "+ B}" in latex and "b_{1}" in latex


def test_a_multivalued_equation_is_judged_by_its_worst_root():
    # W1(x) = (x - 6)^2 against a constant W2 = 0.5: with x = 10 - p_a the residual W1(x) x - 0.5 (10 - x)
    # changes sign three times on [0, 10], so the equation allows three activation pressures
    terms = [(10.0, [36.0, -12.0, 1.0], False), (0.0, [0.5], False)]
    roots = ms.solve_activation(terms, all_roots=True)
    assert len(roots) == 3 and roots == sorted(roots)
    assert ms.solve_activation(terms, p_hint=roots[-1]) == pytest.approx(roots[-1])
    # the fit scores such an equation by its worst root, not by the root nearest the simulation
    samples = [{"p_a": roots[0], "terms": {"in1": (10.0, 10.0 - roots[0], 36.0 - 12.0 * (10.0 - roots[0])
                                                    + (10.0 - roots[0]) ** 2),
                                           "in2": (0.0, -roots[0], 0.5)}}]
    fit = ms.fit_neuron_equation(samples, tolerance=1e-6)
    assert fit["ambiguous"] == 0 or fit["error"] >= roots[-1] - roots[0] - 1e-6


def test_weight_without_usable_points_is_left_out():
    fit = ms.polyfit_weight([0.0, np.nan], [1.0, 2.0], 0)  # dp = 0 is left out
    assert fit["kind"] == "none"
    samples = [{"p_a": 5.0, "terms": {"in": (10.0, 5.0, 1.0), "out": (0.0, -5.0, 1.0),
                                      "W0": (0.0, 0.0, float("nan"))}}]
    result = ms.fit_neuron_equation(samples, tolerance=1e-9)
    assert set(result["fits"]) == {"in", "out"} and result["error"] < 1e-9
    assert result["fits"]["in"]["sides"]["-"] is None  # only sampled with dp > 0


def test_neuron_equation_latex():
    poly = lambda *c: {"kind": "polynomial", "degree": len(c) - 1, "coefficients": list(c)}
    pw = lambda positive, negative: {"kind": "piecewise", "sides": {"+": positive, "-": negative}}
    equation = ms.neuron_equation_latex(
        [("Chamber_Left", pw(poly(1.5e4, -83.6, 0.0, 2.3e-5), poly(3000.0))), ("Chamber_Right", pw(None, poly(2000.0)))],
        (pw(poly(35.2), None), 0.0), "Chamber_Middle")
    assert equation["main"] == (r"p_{\mathrm{Chamber\_Middle}} = \frac{W_{1}\,p_{\mathrm{Chamber\_Left}} + "
                                r"W_{2}\,p_{\mathrm{Chamber\_Right}} + W_{0}\,p_{0}}{W_{1} + W_{2} + W_{0}}")
    w1, w2, w0 = equation["weights"]
    assert w1["pieces"] == [(r"15000-83.6\,\Delta p_{1}+2.3 \times 10^{-5}\,\Delta p_{1}^{3}", r"\Delta p_{1} > 0"),
                            ("3000", r"\Delta p_{1} < 0")]
    assert w1["definition"] == r"\Delta p_{1} = p_{\mathrm{Chamber\_Left}} - p_{\mathrm{Chamber\_Middle}}"
    assert w2["pieces"][0][0] == "2000" and r"\mathrm{sampled\ only}\ \Delta p_{2} < 0" in w2["pieces"][0][1]
    align = ms.equation_align(equation)
    assert r"W_{1}(\Delta p_{1}) &= \begin{cases} 15000" in align and r"3000, & \Delta p_{1} < 0 \end{cases}" in align
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure()
    lines = ms.equation_lines(equation)
    assert [(n, level) for _, n, level in lines] == [(None, 0), (0, 1), (0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2),
                                                     (None, 2)]
    for k, (line, _, _) in enumerate(lines):
        figure.text(0, k / len(lines), f"${line}$")
    FigureCanvasAgg(figure).draw()  # mathtext can render every line
