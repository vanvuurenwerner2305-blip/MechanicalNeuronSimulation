"""Full neuron with only a pre-activation-to-activation design: the valve simulated again at every point, so its
own parameters (Δp, orifice areas, supply/sink pressures, materials) can be swept."""
from types import SimpleNamespace

import numpy as np
import pytest

from app.activation import Connection, display_bodies, generate_activation_mesh, orifice_area
from app.cad import CadModel
from app.full_neuron import PREACTIVATION, SOLVER_KEY, FullNeuronProject, make_dataset, run_valve_grid

KEY = "OutletFluid ↔ TubeFluid"   # the fixture's orifice


@pytest.fixture
def valve(design):
    project = FullNeuronProject()
    project.import_design(design)
    return project


def _worker(rows):
    return SimpleNamespace(check=lambda: None, report=lambda *a: None, log=lambda *a: None,
                           item=SimpleNamespace(emit=rows.append))


def test_an_orifice_sees_its_own_area_or_the_contact_face():
    conn = Connection("a", 0, 1, np.zeros((0, 3), int), 3.0, 6.0, 1.0, 2.0, np.zeros(3))
    assert orifice_area(conn, {"type": "Orifice"}) == 3.0
    assert orifice_area(conn, {"type": "Orifice", "area": 0.0}) == 3.0
    assert orifice_area(conn, {"type": "Orifice", "area": 0.5}) == 0.5


def test_a_design_on_its_own_offers_its_parameters_and_records(valve, tmp_path):
    assert valve.valve_only and valve.ready
    params = {(p, f) for p, f, _, _ in valve.parameters()}
    assert (PREACTIVATION, "dp") in params and (KEY, "area") in params
    assert ("InletFluid", "pressure") in params                      # the supply
    assert any(f == "youngs_modulus" for _, f in params)
    assert "out:activation" in valve.recorded() and "area" in valve.recorded()
    # values changed here are applied on top of the embedded design, kept in the file, and not in the design
    valve.set_valve_value(KEY, "area", 0.25)
    valve.set_valve_value("InletFluid", "pressure", 12.0)
    valve.set_valve_value(SOLVER_KEY, "max_iterations", 55)
    p = valve.valve_project()
    assert p.connections[KEY]["area"] == 0.25 and p.solver.max_iterations == 55
    assert next(q for q in p.parts if q.name == "InletFluid").props["pressure"] == 12.0
    assert "area" not in valve.design_project().connections[KEY]
    swept = valve.fingerprint([(KEY, "area")])
    valve.set_valve_value(KEY, "area", 0.5)
    assert valve.fingerprint([(KEY, "area")]) == swept != valve.fingerprint()
    valve.save(tmp_path / "v.mfn")
    again = FullNeuronProject.load(tmp_path / "v.mfn")
    assert again.valve_only and again.valve_value(KEY, "area") == 0.5


def test_sweeping_the_orifice_area_simulates_the_valve_at_every_point(valve, design, tmp_path):
    """Area 0 (the contact face) reproduces the design's own simulation; a smaller orifice changes the outputs."""
    project = valve.valve_project()
    cad = CadModel()
    project.match_bodies(cad.load_step(project.step_path))
    mesh = generate_activation_mesh(cad, project)
    small = 0.1 * float(np.pi * 0.8 ** 2)   # a tenth of the bore
    valve.sweep["axes"] = [{"part": PREACTIVATION, "field": "dp", "from": 0.0, "to": 30.0, "points": 2, "also": []},
                           {"part": KEY, "field": "area", "from": 0.0, "to": small, "points": 2, "also": []}]
    axes = valve.sweep_axes()
    rows = []
    build = run_valve_grid(_worker(rows), cad, project, mesh, axes)
    assert len(rows) == 4 and all(r["converged"] for r in rows)
    # Δp is the innermost axis: the area changes once
    assert [r["index"][1] for r in rows] == [0, 0, 1, 1]
    d = valve.design()
    by = {r["index"]: r["values"] for r in rows}
    for i, dp in enumerate((0.0, 30.0)):
        assert by[i, 0]["out:activation"] == pytest.approx(d.output("activation", dp), abs=0.05)
        assert by[i, 0]["area"] == pytest.approx(d.area(dp), rel=0.02, abs=1e-3)
        assert by[i, 1]["mdot"] < by[i, 0]["mdot"]
    valve.characterisation = make_dataset(valve, axes, rows, display_bodies(build, project))
    valve.save(tmp_path / "v.mfn")
    c = FullNeuronProject.load(tmp_path / "v.mfn").dataset()
    assert c.names == [f"{PREACTIVATION}.dp", f"{KEY}.area"]
    assert c.shapes() and c.evaluate({c.names[0]: 0.0, c.names[1]: 0.0})["out:activation"] == \
        pytest.approx(by[0, 0]["out:activation"])


def test_sweeping_the_membrane_size_meshes_the_valve_again(valve, tmp_path):
    """A membrane's in-plane size stretches its CAD body and remeshes: a bigger membrane sweeps more volume, every
    size keeps its own mesh in the stored shapes, and the CAD is left as imported."""
    from app.analysis import evaluate_point
    from app.project import DEFORMABLE
    project = valve.valve_project()
    cad = CadModel()
    project.match_bodies(cad.load_step(project.step_path))
    i, part = next((i, p) for i, p in enumerate(project.parts) if p.role in DEFORMABLE)
    valve.design_extents = {p.name: list(b.size) for p, b in zip(project.parts, cad.bodies)}
    size = cad.bodies[i].size.copy()
    thin = int(np.argmin(size))
    field = f"size_{'xyz'[(thin + 1) % 3]}"
    params = {(p, f) for p, f, _, _ in valve.parameters()}
    assert (part.name, field) in params and (part.name, f"size_{'xyz'[thin]}") not in params
    side = size[(thin + 1) % 3]
    other = f"size_{'xyz'[(thin + 2) % 3]}"   # the other in-plane direction follows the axis ("also")
    axis = {"part": part.name, "field": field, "from": side, "to": 1.25 * side, "points": 2, "also": []}
    assert valve.also_candidates(axis) == [f"{part.name}@{other}"]
    axis["also"] = [f"{part.name}@{other}"]
    valve.sweep["axes"] = [{"part": PREACTIVATION, "field": "dp", "from": 0.0, "to": 20.0, "points": 2, "also": []},
                           axis]
    axes = valve.sweep_axes()
    assert valve.axis_label(axes[1][0], field)[0].endswith(f"(= {part.name} size along {other[-1]})")
    rows = []
    build = run_valve_grid(_worker(rows), cad, project, None, axes)
    assert len(rows) == 4 and all(r["converged"] for r in rows)
    by = {r["index"]: r["values"] for r in rows}
    assert by[1, 1]["volume"] > by[1, 0]["volume"] > by[0, 0]["volume"]
    assert np.allclose(cad.bodies[i].size, size) and not cad.extents
    assert len(build.display) == 2 * len(build.shells)
    meshes = [b["rest"] for b in build.display if b["name"] == part.name]
    plane = [(thin + 1) % 3, (thin + 2) % 3]
    assert np.ptp(meshes[0], axis=0)[plane] == pytest.approx([side, side], rel=0.02)   # y = x at every size
    assert np.ptp(meshes[1], axis=0)[plane] == pytest.approx([1.25 * side, 1.25 * side], rel=0.02)
    valve.characterisation = make_dataset(valve, axes, rows, build.display)
    c = valve.dataset()
    shapes = c.shapes()
    _, coords, _, _ = evaluate_point(c, {0: 20.0, 1: 1.2 * side}, True, shapes)
    shown = [b["name"] for b, x in zip(shapes, coords) if np.all(np.isfinite(x))]
    assert sorted(shown) == sorted({b["name"] for b in shapes})   # one mesh per part: the bigger one
    big = next(k for k, b in enumerate(shapes) if b["name"] == part.name and np.all(np.isfinite(coords[k])))
    assert k_index_ok(shapes, big, part.name)


def test_sweeping_a_rigid_body_displacement_builds_the_valve_again(valve):
    """A rigid body's set-up move (Move x/y/z) is a sweep axis: each displacement rebuilds the valve on the same mesh,
    and the block under the tube, moved up into it, pinches it before any pressure."""
    project = valve.valve_project()
    cad = CadModel()
    project.match_bodies(cad.load_step(project.step_path))
    params = {(p, f) for p, f, _, _ in valve.parameters()}
    assert {("Rigid Body", f"move_{a}") for a in "xyz"} <= params
    assert valve.valve_value("Rigid Body", "move_y") == 0.0
    mesh = generate_activation_mesh(cad, project)
    valve.sweep["axes"] = [{"part": "Rigid Body", "field": "move_y", "from": 0.0, "to": 0.3, "points": 2, "also": []}]
    axes = valve.sweep_axes()
    rows, logs = [], []
    worker = _worker(rows)
    worker.log = logs.append
    build = run_valve_grid(worker, cad, project, mesh, axes)
    assert len(rows) == 2 and all(r["converged"] for r in rows)
    assert not any(m.startswith("Meshing") for m in logs)          # same mesh, built again
    assert any("Rigid Body moved y +0.3 mm" in m for m in logs)
    assert len(build.display) == 2 * len(display_bodies(build, project))
    block = next(i for i, p in enumerate(project.parts) if p.name == "Rigid Body")
    assert project.parts[block].props["move_y"] == 0.3                # the last build: the block 0.3 mm up
    lowest = float(build.obstacles[block].vertices[:, 1].min())
    assert lowest == pytest.approx(float(mesh.surfaces[block].vertices[:, 1].min()) + 0.3)
    by = {r["index"]: r["values"] for r in rows}
    assert by[1, ]["area"] < by[0, ]["area"]                           # the tube squeezed from below


def test_the_shut_tube_cutoff_holds_the_flow_and_extrapolates_the_volume():
    from app.activation import ShutHold
    from app.project import SolverSettings
    hold = ShutHold(SolverSettings(shut_cutoff="On", shut_mdot=1e-4))

    def row(dp, mdot, volume):
        return {"dp": dp, "mdot": mdot, "volume": volume, "travel": volume / 100, "area": mdot * 1e4,
                "pressures": [1.0, 2.0], "converged": True, "iterations": 3, "message": ""}
    for dp, mdot, volume in ((0.0, 3e-4, 0.0), (5.0, 2e-4, 50.0), (10.0, 5e-5, 80.0)):
        assert not hold.held(dp)
        hold.add(row(dp, mdot, volume))
    assert hold.shut["dp"] == 10.0 and hold.held(12.0) and not hold.held(10.0)
    held = hold.row(15.0)
    assert held["held"] and held["dp"] == 15.0 and held["mdot"] == 5e-5 and held["area"] == 0.5
    assert held["volume"] == pytest.approx(80.0 + 6.0 * 5.0) and held["travel"] == pytest.approx(0.8 + 0.06 * 5.0)
    held["pressures"][0] = 9.0
    assert hold.shut["pressures"][0] == 1.0          # a copy
    hold.reset()
    assert not hold.held(15.0)
    off = ShutHold(SolverSettings())
    off.add(row(0.0, 0.0, 0.0))
    assert not off.held(5.0)


def test_with_the_cutoff_on_the_valve_grid_holds_the_points_above_the_shut_tube(valve, design):
    """Each Δp line runs rising whatever the axis' direction; above the first point below the mass-flow threshold
    the points are held (flow results copied, volume extrapolated) and marked extrapolated in the dataset."""
    from app.activation import ActivationDesign
    d = ActivationDesign.load(design)
    mdot = [abs(d.mass_flow(x)) for x in (0.0, 10.0, 20.0, 30.0)]
    assert mdot[1] > mdot[2]
    valve.set_valve_value(SOLVER_KEY, "shut_cutoff", "On")
    valve.set_valve_value(SOLVER_KEY, "shut_mdot", 0.5 * (mdot[1] + mdot[2]))   # shut from Δp 20
    project = valve.valve_project()
    assert project.solver.cutoff
    cad = CadModel()
    project.match_bodies(cad.load_step(project.step_path))
    mesh = generate_activation_mesh(cad, project)
    valve.sweep["axes"] = [{"part": PREACTIVATION, "field": "dp", "from": 30.0, "to": 0.0, "points": 4, "also": []},
                           {"part": "InletFluid", "field": "pressure", "from": 10.0, "to": 9.5, "points": 2,
                            "also": []}]
    axes = valve.sweep_axes()
    rows = []
    build = run_valve_grid(_worker(rows), cad, project, mesh, axes)
    assert len(rows) == 8 and all(r["converged"] for r in rows)
    assert [r["params"][0] for r in rows] == [0.0, 10.0, 20.0, 30.0] * 2   # rising in both lines
    limit = project.solver.shut_mdot
    for line in (rows[:4], rows[4:]):   # held exactly above the first point below the threshold
        shut = next(k for k, r in enumerate(line) if abs(r["values"]["mdot"]) < limit)
        assert [bool(r.get("held")) for r in line] == [k > shut for k in range(4)]
    assert [bool(r.get("held")) for r in rows[:4]] == [False, False, False, True]
    by = {tuple(r["params"]): r for r in rows}
    shut, held = by[20.0, 10.0]["values"], by[30.0, 10.0]["values"]
    for key in ("area", "mdot", "p_end", "out:activation"):
        assert held[key] == shut[key]
    slope = (shut["volume"] - by[10.0, 10.0]["values"]["volume"]) / 10.0
    assert held["volume"] == pytest.approx(shut["volume"] + 10.0 * slope)
    assert by[30.0, 10.0]["coords"] is by[20.0, 10.0]["coords"]
    assert by[20.0, 10.0]["values"]["area"] == pytest.approx(d.area(20.0), rel=0.02, abs=1e-3)
    data = make_dataset(valve, axes, rows, display_bodies(build, project))
    flags = np.array(data["extrapolated"]).reshape(4, 2)   # in the axis order: Δp 30 first
    assert all(flags[r["index"]] == bool(r.get("held")) for r in rows) and flags.any()


def test_with_the_cutoff_on_the_design_sweep_holds_the_points_above_the_shut_tube(design, tmp_path):
    from app.activation import ActivationDesign, ActivationProject, build_activation, run_study
    d = ActivationDesign.load(design)
    project = ActivationProject.load(design)
    cad = CadModel()
    project.match_bodies(cad.load_step(project.step_path))
    project.solver.shut_cutoff = "On"
    project.solver.shut_mdot = 0.5 * (abs(d.mass_flow(10.0)) + abs(d.mass_flow(20.0)))
    build = build_activation(cad, generate_activation_mesh(cad, project), project)
    results, states = run_study(build, project)
    assert results["held"] == [False, False, False, True] and all(results["converged"])
    assert results["area"][3] == results["area"][2] and results["mdot"][3] == results["mdot"][2]
    assert results["volume"][3] == pytest.approx(2 * results["volume"][2] - results["volume"][1])
    assert states[3]["dp"] == 30.0 and states[3]["shell_coords"] is states[2]["shell_coords"]
    project.results = results
    project.save(tmp_path / "held.mad")
    again = ActivationDesign.load(tmp_path / "held.mad")
    assert again.volume[3] == pytest.approx(results["volume"][3])
    assert again.area(30.0) == pytest.approx(results["area"][2])


def k_index_ok(shapes, k, name):
    """The drawn mesh of `name` is its second one (the 1.25x membrane)."""
    return [j for j, b in enumerate(shapes) if b["name"] == name][1] == k
