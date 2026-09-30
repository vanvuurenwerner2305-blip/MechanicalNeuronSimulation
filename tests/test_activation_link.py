"""An activation-function design used in a neuron: its membrane replaced by the pre-simulated response."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import membrane_sim as ms  # noqa: E402
from membrane_sim.empirical import empirical_curve  # noqa: E402

KPA = 1e-3


def disk_response(dps):
    """Swept volume of a clamped disk membrane at each Δp (kPa), and the model with it."""
    env = ms.Environment(contact_stiffness=1.0)
    shell = env.add_disk_membrane([0, 0, 0], [0, 0, 1], 5.0, rings=8, thickness=0.5, youngs_modulus=0.5)
    load = env.add_fluid_volume([(shell, +1)])
    volumes = []
    for k, dp in enumerate(dps):
        load.P0 = dp * KPA
        assert env.solve(load_steps=2, warm_start=k > 0).converged
        volumes.append(load.delta_volume)
    return np.asarray(volumes)


def test_empirical_curve_is_monotone_mirrored_and_through_rest():
    dp, V = empirical_curve([5.0, 10.0, 20.0, 15.0], [1.0, 2.0, 2.0, 1.5])
    # (0, 0) added; 15 and 20 dropped (the volume does not grow past 10); mirrored below 0
    assert list(dp) == [-10.0, -5.0, 0.0, 5.0, 10.0] and list(V) == [-2.0, -1.0, 0.0, 1.0, 2.0]


def test_empirical_membrane_terms_are_consistent_and_replace_the_membrane():
    dps = np.linspace(0.0, 20.0, 11)
    volumes = disk_response(dps)
    wall = ms.EmpiricalMembrane(dps * KPA, volumes, area=np.pi * 25.0, name="valve")
    for v in (-30.0, 10.0, 60.0, 1.5 * volumes[-1]):  # energy -> dp -> stiffness, inside and extrapolated
        h = 1e-4
        e, dp, k = wall.response(v)
        assert dp == pytest.approx((wall.response(v + h)[0] - wall.response(v - h)[0]) / (2 * h), rel=1e-5, abs=1e-12)
        assert k == pytest.approx((wall.response(v + h)[1] - wall.response(v - h)[1]) / (2 * h), rel=1e-5)
    assert wall.response(0.0)[:2] == (pytest.approx(0.0), pytest.approx(0.0))

    # constant pressure behind it: it sweeps the simulated volume
    env = ms.Environment(contact_stiffness=1.0)
    env.membrane_list.append(wall)
    drive = env.add_fluid_volume([(wall, +1)], P0=10.0 * KPA)
    assert env.solve(load_steps=2).converged
    assert drive.delta_volume == pytest.approx(volumes[5], rel=1e-6)

    # a sealed gas chamber closed by the real membrane and by its empirical stand-in reach the same state
    def sealed(body_of):
        env = ms.Environment(contact_stiffness=1.0)
        body = body_of(env)
        gas = env.add_fluid_volume([(body, +1)], P0=10.0 * KPA, gas_volume=500.0)
        assert env.solve(load_steps=4).converged
        return gas.P / KPA
    fe = sealed(lambda env: env.add_disk_membrane([0, 0, 0], [0, 0, 1], 5.0, rings=8, thickness=0.5,
                                                  youngs_modulus=0.5))
    wall.reset()
    empirical = sealed(lambda env: env.membrane_list.append(wall) or wall)
    assert empirical == pytest.approx(fe, abs=0.05)  # kPa, interpolation between 2 kPa samples


def test_design_records_the_swept_volume(design):
    from app.activation import ActivationDesign
    d = ActivationDesign.load(design)
    assert d.has_volume and d.membrane_area > 0
    # at dp = 0 the gas in the tube (10 kPa inlet) pushes the membrane back a little
    assert d.volume[0] < 0 and np.all(np.diff(d.volume) > 0)
    out = d.outputs(45.0)
    assert out["extrapolated"] and out["area"] == pytest.approx(d.A[-1])


def test_named_outputs_are_the_pressures_of_their_segments(design, tmp_path):
    from app.activation import (ActivationDesign, ActivationProject, clean_outputs, output_definitions,
                                output_pressures, segment_pressures)
    project = ActivationProject.load(design)
    r = project.results
    P = np.asarray(r["pressures"])
    assert segment_pressures(r).shape == (len(r["dp"]), 4)
    # without outputs: the downstream segment is the one output "activation"
    assert output_definitions(project.parts, r) == [{"name": "activation", "segment": 4}]
    assert ActivationDesign.load(design).output_names == ["activation"]
    tube = next(q for q in project.parts if q.name == r["lumen"])
    tube.props["outputs"] = [{"name": "a1", "segment": 2}, {"name": "a1", "segment": 9}, {"name": "", "segment": 1}]
    outputs = output_definitions(project.parts, r)          # unique names, segments clamped
    assert outputs == [{"name": "a1", "segment": 2}, {"name": "a1 (2)", "segment": 4}, {"name": "output 3", "segment": 1}]
    assert np.allclose(output_pressures(r, outputs)["a1"], 0.5 * (P[:, 1] + P[:, 2]))
    path = tmp_path / "valve_outputs.mad"                   # saved with the design, no new simulation
    project.save(path)
    d = ActivationDesign.load(path)
    assert d.output_names == ["a1", "a1 (2)", "output 3"]
    out = d.outputs(10.0)
    assert out["outputs"]["a1"] == pytest.approx(np.interp(10.0, d.dp, 0.5 * (P[:, 1] + P[:, 2])))
    assert out["outputs"]["output 3"] == pytest.approx(np.interp(10.0, d.dp, 0.5 * (P[:, 0] + P[:, 1])))
    assert {"lumen", "axis", "bounds"} <= set(r) and len(r["bounds"]) == 5
    assert clean_outputs([{"name": " x ", "segment": 0}]) == [{"name": "x", "segment": 1}]


def test_the_single_output_of_the_earlier_version_becomes_a_named_output(design, tmp_path):
    import json
    from app.activation import ActivationProject
    data = json.loads(Path(design).read_text(encoding="utf-8"))
    data["output_segment"] = 3
    path = tmp_path / "old.mad"
    path.write_text(json.dumps(data), encoding="utf-8")
    project = ActivationProject.load(path)
    tube = next(q for q in project.parts if q.name == project.results["lumen"])
    assert tube.props["outputs"] == [{"name": "activation", "segment": 3}]


def test_output_segments_are_found_on_the_fluid_mesh():
    from types import SimpleNamespace
    from app.activation import segment_faces
    # a strip 0..10 along x, cut into 4 segments: every face in one segment
    from membrane_sim.mesh import rectangle_mesh
    V, F = rectangle_mesh([0, 0, 0], [10, 0, 0], [0, 0, 2], 8, 2)
    surface = SimpleNamespace(vertices=np.asarray(V, float), faces=np.asarray(F))
    parts = [segment_faces(surface, np.array([1.0, 0, 0]), 4, k) for k in (1, 2, 3, 4)]
    centers = surface.vertices[surface.faces].mean(axis=1)[:, 0]
    for k, faces in enumerate(parts):
        assert len(faces) and np.all((centers[faces] >= 2.5 * k - 1e-9) & (centers[faces] <= 2.5 * (k + 1) + 1e-9))
    assert set(np.concatenate(parts)) == set(range(len(surface.faces)))


@pytest.fixture(scope="module")
def linked_neuron(design, tmp_path_factory):
    import make_neuron_step
    from app.cad import CadModel
    from app.project import ACTIVATION_MEMBRANE, CHAMBER, INCOMPRESSIBLE, Project
    step = tmp_path_factory.mktemp("neuron") / "neuron.step"
    make_neuron_step.build(step)
    cad = CadModel()
    cad.load_step(step)
    project = Project(str(step), [b.name for b in cad.bodies])
    project.auto_assign_from_names()
    for part in project.parts:
        if part.role == CHAMBER:
            part.props.update({"Chamber_Left": dict(pressure=10.0), "Chamber_Right": dict(pressure=0.0),
                               "Chamber_Middle": dict(model=INCOMPRESSIBLE, stiffness=10.0)}[part.name])
        if part.name == "Membrane_Right":
            project.set_role(part, ACTIVATION_MEMBRANE)
            part.props["design"] = str(design)
    return cad, project, step


def test_activation_membrane_is_driven_by_the_activation_chamber(linked_neuron, tmp_path):
    from app.builder import build_environment, generate_mesh
    from app.project import Project
    cad, project, step = linked_neuron
    cad.load_step(step)
    index = {p.name: i for i, p in enumerate(project.parts)}
    build = build_environment(cad, generate_mesh(cad, project), project)
    link = build.activation[index["Membrane_Right"]]
    assert link.sides == {index["Chamber_Middle"]: 1, index["Chamber_Right"]: -1}
    assert index["Membrane_Right"] not in build.shells

    result = build.solve(project.solver)
    assert result.converged
    pressures = {c: v.P for c, v in build.volumes.items()}
    dp = link.pressure_difference(pressures)
    assert dp == pytest.approx(pressures[index["Chamber_Middle"]] / KPA - pressures[index["Chamber_Right"]] / KPA)
    assert dp > 0
    # at equilibrium the design's curve gives the same Δp at the volume the wall swept
    assert link.body.pressure_difference() / KPA == pytest.approx(dp, rel=1e-4, abs=1e-4)
    out = build.activation_outputs()[index["Membrane_Right"]]
    assert out["area"] == pytest.approx(link.design.area(dp)) and out["mdot"] > 0
    assert out["outputs"]["activation"] == pytest.approx(link.design.output("activation", dp))
    steps = build.activation_outputs(build.env.history[-1])
    assert steps[index["Membrane_Right"]]["dp"] == pytest.approx(dp)

    # the design path is saved next to the project and resolved on load
    saved = tmp_path / "linked.mns"
    project.save(saved)
    loaded = Project.load(saved)
    assert Path(loaded.parts[index["Membrane_Right"]].props["design"]) == Path(link.path).resolve()


def test_sweep_reports_the_activation_outputs_and_a_weight_for_the_design(linked_neuron):
    from app.sweep import run_sweep
    cad, project, step = linked_neuron
    cad.load_step(step)
    index = {p.name: i for i, p in enumerate(project.parts)}
    rows = []
    worker = SimpleNamespace(check=lambda: None, report=lambda *a: None, log=lambda *a: None,
                             item=SimpleNamespace(emit=rows.append))
    run_sweep(worker, cad, project, None, index["Chamber_Left"], np.array([5.0, 15.0]), None, None,
              index["Chamber_Middle"])
    assert all(r["converged"] for r in rows)
    act = [r["act"][index["Membrane_Right"]] for r in rows]
    assert act[1]["dp"] > act[0]["dp"] and act[1]["area"] <= act[0]["area"]
    for r in rows:  # the design's membrane is a path from the tube side into the activation chamber
        assert "Chamber_Right" in r["W"]
        assert r["p_a_rebuilt"] == pytest.approx(r["P"][index["Chamber_Middle"]], rel=1e-5)


def test_the_design_stores_its_fem_solution_at_every_point(design):
    from app.activation import ActivationDesign, decode_array, encode_array
    a = np.random.default_rng(0).normal(size=(3, 4, 3))
    assert np.allclose(decode_array(encode_array(a)), a, atol=1e-6)
    assert np.array_equal(decode_array(encode_array(np.arange(6).reshape(2, 3))), np.arange(6).reshape(2, 3))

    d = ActivationDesign.load(design)
    assert d.frames is not None
    kinds = {b["name"]: b["kind"] for b in d.frames.bodies}
    assert kinds["Tube"] == "solid" and kinds["Membrane"] == "shell"
    for b in d.frames.bodies:
        assert b["frames"].shape == (len(d.dp),) + b["rest"].shape and b["faces"].max() < len(b["rest"])
    tube = lambda dp: next(f for f in d.frames.at(dp) if f["name"] == "Tube")  # noqa: E731
    at0, at1 = tube(d.dp[1]), tube(d.dp[2])
    mid = tube(0.5 * (d.dp[1] + d.dp[2]))
    assert np.allclose(mid["x"], 0.5 * (at0["x"] + at1["x"]))            # linear between the points
    assert np.allclose(tube(d.dp[-1] + 50.0)["x"], tube(d.dp[-1])["x"])   # held at the end outside
    squeeze = [np.abs(tube(dp)["x"] - tube(dp)["rest"]).max() for dp in d.dp]
    assert squeeze[-1] > squeeze[0]                                        # the tube deforms more under more Δp


def test_leaving_the_simulated_range_gives_an_extrapolation_warning(design):
    from app.activation import ActivationDesign
    d = ActivationDesign.load(design)
    lo, hi = d.dp_range
    assert d.range_warning(0.5 * (lo + hi)) == "" and not d.outputs(0.5 * (lo + hi))["extrapolated"]
    above, below = d.range_warning(hi + 5.0, "Body14"), d.range_warning(lo - 5.0)
    assert "above" in above and "EXTRAPOLATING" in above and above.startswith("Body14:")
    assert "below" in below and d.outputs(lo - 5.0)["extrapolated"]
