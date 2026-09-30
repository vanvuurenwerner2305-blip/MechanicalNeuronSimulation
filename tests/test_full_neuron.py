"""Full-neuron workbench without the GUI: import a neuron and a design, link them, record, sweep, save and load."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.full_neuron import (ACTIVATION, NEURON, Characterisation, FullNeuronProject, make_dataset, record_values,
                             predicted_state, rotation, run_grid, run_points, serpentine, shell_bodies,
                             transform_points)
from app.project import ACTIVATION_MEMBRANE, AUTOMATIC, CHAMBER, INCOMPRESSIBLE, MEMBRANE, Project

KPA = 1e-3


@pytest.fixture(scope="module")
def neuron_mns(tmp_path_factory):
    """The example neuron as a saved project: inputs Left (10 kPa) and Right (0 kPa), pre-activation chamber Middle."""
    import make_neuron_step
    from app.cad import CadModel
    folder = tmp_path_factory.mktemp("neuron")
    step = folder / "neuron.step"
    make_neuron_step.build(step)
    cad = CadModel()
    cad.load_step(step)
    project = Project(str(step), [b.name for b in cad.bodies])
    project.auto_assign_from_names()
    for part in project.parts:
        if part.role == CHAMBER:
            part.props.update({"Chamber_Left": dict(pressure=10.0), "Chamber_Right": dict(pressure=0.0),
                               "Chamber_Middle": dict(model=INCOMPRESSIBLE, stiffness=10.0)}[part.name])
        if part.role == MEMBRANE:
            part.props["thickness"] = 1.0
    path = folder / "neuron.mns"
    project.save(path)
    return path


@pytest.fixture
def full(neuron_mns, design):
    project = FullNeuronProject()
    project.import_neuron(neuron_mns)
    project.import_design(design)
    return project


def test_linking_replaces_the_chosen_membrane(full, design):
    assert full.link["part"] is None                               # nothing named in the .mns: choose it
    with pytest.raises(ValueError, match="Link the design"):
        full.linked_project()
    assert set(full.link_candidates()) == {"Membrane_Left", "Membrane_Right"}
    full.link = {"part": "Membrane_Right", "driving": AUTOMATIC}
    linked = full.linked_project()
    part = next(p for p in linked.parts if p.name == "Membrane_Right")
    assert part.role == ACTIVATION_MEMBRANE and Path(part.props["design"]) == Path(design).resolve()
    assert part.props["thickness"] == 1.0
    assert full.part("Membrane_Right").role == MEMBRANE              # the embedded neuron keeps its roles


def test_catalogue_and_default_record(full):
    full.link = {"part": "Membrane_Right", "driving": AUTOMATIC}
    keys = [k for k, _, _ in full.catalogue()]
    assert {"P:Chamber_Left", "P:Chamber_Middle", "dV:Chamber_Middle", "dp", "out:activation", "area",
            "mdot"} <= set(keys)
    assert full.recorded() == ["P:Chamber_Middle", "dp", "out:activation"]   # pre-activation + outputs
    full.record = ["mdot", "P:Chamber_Left", "unknown"]
    assert full.recorded() == ["P:Chamber_Left", "mdot"]                     # catalogue order, known keys only
    assert full.inputs() == ["Chamber_Left", "Chamber_Right"]


def test_solve_and_sweep_record_the_chosen_quantities(full):
    from app.builder import generate_mesh
    from app.cad import CadModel
    full.link = {"part": "Membrane_Right", "driving": AUTOMATIC}
    full.part("Chamber_Left").props["pressure"] = 12.0              # a fluid parameter changed in the workbench
    linked = full.linked_project()
    cad = CadModel()
    cad.load_step(linked.step_path)
    mesh = generate_mesh(cad, linked)
    rows = []
    worker = SimpleNamespace(check=lambda: None, report=lambda *a: None, log=lambda *a: None,
                             item=SimpleNamespace(emit=rows.append))
    build = run_points(worker, cad, linked, mesh)                   # one solve at the set pressures
    assert len(rows) == 1 and rows[0]["converged"] and rows[0]["a"] is None
    v = rows[0]["values"]
    assert v["P:Chamber_Left"] == pytest.approx(12.0)
    assert v["dp"] == pytest.approx(v["P:Chamber_Middle"] - v["P:Chamber_Right"])
    link = build.activation[[p.name for p in linked.parts].index("Membrane_Right")]
    assert v["out:activation"] == pytest.approx(link.design.output("activation", v["dp"]))
    assert len(rows[0]["coords"]) == len(build.shells) and "Membrane_Right" not in \
        [linked.parts[i].name for i in build.shells]

    rows.clear()
    run_points(worker, cad, linked, mesh, "Chamber_Left", [5.0, 15.0], "Chamber_Right", [0.0, 2.0])
    assert len(rows) == 4 and all(r["converged"] for r in rows)
    by = {(r["a"], r["b"]): r["values"] for r in rows}
    assert by[15.0, 0.0]["P:Chamber_Middle"] > by[5.0, 0.0]["P:Chamber_Middle"]
    assert by[15.0, 0.0]["dp"] > by[5.0, 0.0]["dp"]
    assert by[5.0, 2.0]["P:Chamber_Right"] == pytest.approx(2.0)
    assert set(record_values(build, linked.parts)) >= {"dp", "out:activation", "area", "mdot"}


def test_workbench_file_round_trips_without_touching_the_neuron(full, neuron_mns, tmp_path):
    original = Path(neuron_mns).read_text()
    full.link = {"part": "Membrane_Right", "driving": "Chamber_Middle"}
    full.part("Chamber_Left").props["pressure"] = 7.5
    full.transforms[ACTIVATION] = {"position": [80.0, 0.0, 5.0], "rotation": [0.0, 0.0, 90.0]}
    full.record = ["P:Chamber_Middle", "out:activation"]
    full.sweep["axes"] = [{"part": "Chamber_Left", "field": "pressure", "from": 0.0, "to": 30.0, "points": 4}]
    path = tmp_path / "sub" / "full.mfn"
    path.parent.mkdir()
    full.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert not Path(data["design"]).is_absolute() or Path(data["design"]).exists()
    loaded = FullNeuronProject.load(path)
    assert loaded.link == full.link and loaded.record == full.record
    assert loaded.transforms[ACTIVATION]["rotation"] == [0.0, 0.0, 90.0]
    assert loaded.transforms[NEURON] == {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}
    assert loaded.part("Chamber_Left").props["pressure"] == 7.5
    assert Path(loaded.neuron.step_path).exists() and Path(loaded.design_path) == Path(full.design_path)
    assert loaded.sweep["axes"] == full.sweep["axes"]
    assert Path(neuron_mns).read_text() == original                   # the imported .mns is not changed


def test_a_design_without_its_swept_volume_is_refused(design, tmp_path):
    data = json.loads(Path(design).read_text(encoding="utf-8"))
    del data["results"]["volume"]
    old = tmp_path / "old.mad"
    old.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="swept volume"):
        FullNeuronProject().import_design(old)


def test_placement_rotates_about_the_centre_then_moves():
    points = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    center = np.array([2.0, 0.0, 0.0])
    out = transform_points(points, center, [0.0, 10.0, 0.0], [0.0, 0.0, 90.0])
    assert np.allclose(out, [[2.0, 9.0, 0.0], [2.0, 11.0, 0.0]])
    assert np.allclose(rotation([0, 0, 0]), np.eye(3))


def test_a_pre_activation_outside_the_design_range_is_flagged(full):
    from app.builder import generate_mesh
    from app.cad import CadModel
    full.link = {"part": "Membrane_Right", "driving": AUTOMATIC}
    linked = full.linked_project()
    cad = CadModel()
    cad.load_step(linked.step_path)
    mesh = generate_mesh(cad, linked)
    rows, logged = [], []
    worker = SimpleNamespace(check=lambda: None, report=lambda *a: None, log=logged.append,
                             item=SimpleNamespace(emit=rows.append))
    lo, hi = full.design().dp_range
    run_points(worker, cad, linked, mesh, "Chamber_Left", [0.5 * (lo + hi), hi + 10.0])
    inside, outside = sorted(rows, key=lambda r: r["a"])
    assert inside["values"]["warnings"] == [] and not inside["values"]["extrapolated"]
    assert outside["values"]["dp"] > hi and outside["values"]["extrapolated"]
    assert len(outside["values"]["warnings"]) == 1 and "EXTRAPOLATING" in outside["values"]["warnings"][0]
    assert any("EXTRAPOLATING" in line for line in logged)


def test_serpentine_steps_one_axis_at_a_time():
    points = serpentine([3, 2, 4])
    assert len(points) == 24 and len(set(points)) == 24
    for p, q in zip(points, points[1:]):
        assert sum(abs(a - b) for a, b in zip(p, q)) == 1
    assert serpentine([]) == [()]


def test_old_workbench_sweeps_become_axes(full, tmp_path):
    path = tmp_path / "old.mfn"
    full.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["sweep"] = {"a": "Chamber_Left", "a_from": 0.0, "a_to": 30.0, "a_points": 7, "b": None}
    data.pop("characterisation")
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = FullNeuronProject.load(path)
    assert loaded.sweep["axes"] == [{"part": "Chamber_Left", "field": "pressure", "from": 0.0, "to": 30.0,
                                     "points": 7}]
    assert loaded.characterisation is None and loaded.dataset() is None


def test_characterisation_over_several_parameters_is_saved_and_interpolated(full, tmp_path):
    """Two pressures and a (non-pressure) stiffness swept together; every recorded quantity stored on the grid,
    saved in the .mfn and read back as a Characterisation that interpolates."""
    from app.builder import generate_mesh
    from app.cad import CadModel
    full.link = {"part": "Membrane_Right", "driving": AUTOMATIC}
    names = {(p, f) for p, f, _, _ in full.parameters()}
    assert {("Chamber_Left", "pressure"), ("Chamber_Middle", "stiffness"), ("Chamber_Middle", "fluid_volume")} <= names
    assert ("Chamber_Left", "stiffness") not in names                   # a constant chamber has no stiffness
    full.sweep["axes"] = [
        {"part": "Chamber_Left", "field": "pressure", "from": 4.0, "to": 12.0, "points": 3},
        {"part": "Chamber_Right", "field": "pressure", "from": 0.0, "to": 2.0, "points": 2},
        {"part": "Chamber_Middle", "field": "stiffness", "from": 5.0, "to": 20.0, "points": 2},
        {"part": "Chamber_Left", "field": "pressure", "from": 0.0, "to": 1.0, "points": 9}]   # repeat: ignored
    axes = full.sweep_axes()
    assert [(p, f, len(v)) for p, f, v in axes] == [("Chamber_Left", "pressure", 3), ("Chamber_Right", "pressure", 2),
                                                     ("Chamber_Middle", "stiffness", 2)]
    full.record = ["P:Chamber_Middle", "dV:Chamber_Middle", "dp", "out:activation", "mdot"]
    linked = full.linked_project()
    cad = CadModel()
    cad.load_step(linked.step_path)
    mesh = generate_mesh(cad, linked)
    rows = []
    worker = SimpleNamespace(check=lambda: None, report=lambda *a: None, log=lambda *a: None,
                             item=SimpleNamespace(emit=rows.append))
    build = run_grid(worker, cad, linked, mesh, axes)
    assert len(rows) == 12 and all(r["converged"] for r in rows)
    by = {tuple(r["params"]): r["values"] for r in rows}
    # a stiffer pre-activation chamber moves less and its pressure is further from the inputs' mean
    soft, stiff = by[12.0, 0.0, 5.0], by[12.0, 0.0, 20.0]
    assert abs(stiff["dV:Chamber_Middle"]) < abs(soft["dV:Chamber_Middle"])
    assert by[12.0, 0.0, 20.0]["P:Chamber_Middle"] > by[4.0, 0.0, 20.0]["P:Chamber_Middle"]

    data = make_dataset(full, axes, rows, shell_bodies(build, linked.parts), store_shapes=True, seconds=1.0)
    full.characterisation = data
    path = tmp_path / "characterised.mfn"
    full.save(path)
    loaded = FullNeuronProject.load(path)
    assert loaded.dataset_current()
    c = loaded.dataset()
    assert c.shape == (3, 2, 2) and c.complete and c.solved.all() and c.converged.all()
    assert c.names == ["Chamber_Left.pressure", "Chamber_Right.pressure", "Chamber_Middle.stiffness"]
    assert c.keys == ["P:Chamber_Middle", "dV:Chamber_Middle", "dp", "out:activation", "mdot"]
    grid = c.grid("P:Chamber_Middle")
    assert grid[2, 0, 1] == pytest.approx(stiff["P:Chamber_Middle"])
    # at a grid point the interpolation is the stored value; between points it lies between them
    at = c.evaluate({"Chamber_Left.pressure": 12.0, "Chamber_Right.pressure": 0.0, "Chamber_Middle.stiffness": 20.0})
    assert at["dp"] == pytest.approx(stiff["dp"])
    mid = c.evaluate({"Chamber_Left.pressure": 10.0, "Chamber_Right.pressure": 0.0, "Chamber_Middle.stiffness": 20.0})
    lo, hi = sorted((by[8.0, 0.0, 20.0]["P:Chamber_Middle"], stiff["P:Chamber_Middle"]))
    assert lo <= mid["P:Chamber_Middle"] <= hi
    assert c.out_of_range({"Chamber_Left.pressure": 30.0}) == ["Chamber_Left.pressure"]
    # the deformed sheets come back for the 3D view
    shapes = c.shapes()
    assert [b["name"] for b in shapes] == [linked.parts[i].name for i in build.shells]
    back = c.rows(shapes)
    assert len(back) == 12 and np.allclose(back[-1]["coords"][0], rows[[tuple(r["index"]) for r in rows]
                                                                       .index(back[-1]["index"])]["coords"][0],
                                           atol=1e-4)
    # changing a fixed parameter makes it outdated, changing a swept one does not
    loaded.part("Chamber_Left").props["pressure"] = 99.0
    assert loaded.dataset_current()
    loaded.part("Chamber_Middle").props["fluid_volume"] = 1.0
    assert not loaded.dataset_current()


def test_an_interrupted_characterisation_is_kept_incomplete(full):
    full.link = {"part": "Membrane_Right", "driving": AUTOMATIC}
    axes = [("Chamber_Left", "pressure", [0.0, 5.0, 10.0])]
    row = {"index": (1,), "params": [5.0], "converged": True,
           "values": {"P:Chamber_Middle": 2.0, "dp": 1.5, "out:activation": 0.3, "warnings": []}, "coords": None}
    c = Characterisation(make_dataset(full, axes, [row], None, store_shapes=True))
    assert not c.complete and c.solved.tolist() == [False, True, False] and c.shapes() is None
    assert np.isnan(c.grid("dp")[0]) and c.grid("dp")[1] == 1.5
    assert [r["params"] for r in c.rows()] == [[5.0]]


def test_the_next_point_starts_on_the_line_through_the_last_two():
    import torch
    axes = [("A", "pressure", [0.0, 1.0, 3.0]), ("B", "pressure", [0.0, 5.0])]
    states = {(0, 0): [torch.tensor([0.0, 1.0])], (1, 0): [torch.tensor([2.0, 1.5])]}
    guess = predicted_state(states, (1, 0), (2, 0), axes)             # step 2 after a step of 1: t = 2
    assert torch.allclose(guess[0], torch.tensor([6.0, 2.5]))
    assert predicted_state(states, (1, 0), (1, 1), axes) is None      # turning onto another axis: no line
    assert predicted_state({(1, 0): states[(1, 0)]}, (1, 0), (2, 0), axes) is None
