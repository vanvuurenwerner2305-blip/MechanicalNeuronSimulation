"""Pipeline tests for the application layer (no GUI): STEP import -> roles -> mesh -> build."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from app.builder import build_environment, generate_mesh  # noqa: E402
from app.cad import CadModel, _signed_volume  # noqa: E402
from app.project import CHAMBER, IDEAL_GAS, MEMBRANE, RIGID, Project  # noqa: E402
import make_neuron_step  # noqa: E402


@pytest.fixture(scope="module")
def neuron(tmp_path_factory):
    path = tmp_path_factory.mktemp("cad") / "neuron.step"
    make_neuron_step.build(path)
    cad = CadModel()
    cad.load_step(path)
    project = Project(str(path), [b.name for b in cad.bodies])
    project.auto_assign_from_names()
    yield cad, project, path


def test_step_import_keeps_part_names_and_volumes(neuron):
    cad, _, _ = neuron
    volumes = {b.name: b.volume for b in cad.bodies}
    assert set(volumes) == {"Frame", "Membrane_Left", "Membrane_Right", "Obstacle_Top", "Obstacle_Bottom",
                            "Chamber_Left", "Chamber_Middle", "Chamber_Right"}
    assert volumes["Membrane_Left"] == pytest.approx(1.0 * 60 * 40)
    assert volumes["Chamber_Middle"] == pytest.approx(29 * 60 * 40 - 2 * 17200)


def test_auto_assign_roles_from_names(neuron):
    _, project, _ = neuron
    roles = {p.name: p.role for p in project.parts}
    assert roles["Membrane_Left"] == MEMBRANE
    assert roles["Chamber_Middle"] == CHAMBER
    assert roles["Frame"] == RIGID and roles["Obstacle_Top"] == RIGID


def test_surface_meshes_are_closed_and_outward(neuron):
    cad, project, _ = neuron
    data = generate_mesh(cad, project)
    for body in cad.bodies:  # includes the frame, whose inner cavity must face into the cavity
        mesh = data.surfaces[body.index]
        assert _signed_volume(mesh.vertices, mesh.faces) == pytest.approx(body.volume, rel=1e-6)


def test_midsurface_of_membrane(neuron):
    cad, project, _ = neuron
    data = generate_mesh(cad, project)
    left = next(b for b in cad.bodies if b.name == "Membrane_Left")
    mid = data.midsurfaces[left.index]
    assert mid.thickness == pytest.approx(1.0)
    assert np.allclose(mid.vertices[:, 0], -15.0)


def test_chambers_are_coupled_to_the_right_membranes(neuron):
    cad, project, _ = neuron
    build = build_environment(cad, generate_mesh(cad, project), project)
    name = {i: p.name for i, p in enumerate(project.parts)}
    couplings = {name[c]: sorted((name[k.shell_index], k.side) for k in ks) for c, ks in build.couplings.items()}
    # membrane normals point to -x (out of the largest face at x = -15.5 / 14.5)
    left_side = couplings["Chamber_Left"][0][1]
    assert couplings["Chamber_Left"] == [("Membrane_Left", left_side)]
    assert couplings["Chamber_Right"] == [("Membrane_Right", couplings["Chamber_Right"][0][1])]
    assert sorted(n for n, _ in couplings["Chamber_Middle"]) == ["Membrane_Left", "Membrane_Right"]
    middle_on_left = dict(couplings["Chamber_Middle"])["Membrane_Left"]
    assert middle_on_left == -left_side  # opposite sides of the same membrane
    assert not build.warnings


def test_project_round_trip(neuron, tmp_path):
    cad, project, _ = neuron
    chamber = next(p for p in project.parts if p.name == "Chamber_Middle")
    chamber.props["model"] = IDEAL_GAS
    project.save(tmp_path / "p.mns")
    loaded = Project.load(tmp_path / "p.mns")
    loaded.match_bodies(cad.bodies)
    assert [(p.name, p.role, p.props) for p in loaded.parts] == [(p.name, p.role, p.props) for p in project.parts]


def test_fusion_style_multibody_names():
    step = ROOT / "cad_models" / "Example1.step"
    if not step.exists():
        pytest.skip("example file not present")
    cad = CadModel()
    names = [b.name for b in cad.load_step(step)]
    assert names == [f"Body{i}" for i in range(1, 7)]


def test_incompressible_fraction_sets_the_gas_volume():
    from app.builder import P_ATM, chamber_model
    props = {"model": IDEAL_GAS, "pressure": 5.0, "incompressible": 60.0}
    kw = chamber_model(props, 1000.0)
    assert kw["gas_volume"] == pytest.approx(400.0)
    assert kw["P0"] == pytest.approx(5e-3)
    assert kw["atmospheric_pressure"] == P_ATM


def test_builder_creates_gas_chamber_with_liquid_share(neuron):
    cad, project, path = neuron
    cad.load_step(path)  # gmsh holds one model at a time; other tests may have loaded another file
    part = next(p for p in project.parts if p.name == "Chamber_Middle")
    part.props.update(model=IDEAL_GAS, incompressible=60.0)
    build = build_environment(cad, generate_mesh(cad, project), project)
    index = project.parts.index(part)
    volume = build.volumes[index]
    assert volume.gas_volume == pytest.approx(0.4 * cad.bodies[index].volume)
    part.props.update(model="Constant pressure (input)", incompressible=0.0)


def test_incompressible_and_vent_chamber_models():
    from app.builder import chamber_model
    from app.project import INCOMPRESSIBLE, VENT
    kw = chamber_model({"model": INCOMPRESSIBLE, "pressure": 2.0, "stiffness": 22000.0}, 1000.0)
    # 1 % of 1000 mm3 = 10 mm3 must raise the pressure by 22000 kPa
    assert kw["bulk_stiffness"] * 10.0 == pytest.approx(22.0)  # MPa
    assert kw["P0"] == pytest.approx(2e-3)
    assert chamber_model({"model": VENT, "pressure": 7.0}, 1000.0)["P0"] == 0.0


def test_incompressible_fluid_volume_defaults_to_the_body_and_can_differ():
    from app.builder import chamber_model
    from app.project import INCOMPRESSIBLE
    full = chamber_model({"model": INCOMPRESSIBLE, "stiffness": 10.0}, 100.0)
    assert full["liquid_volume"] == 100.0 and full["bulk_stiffness"] == pytest.approx(0.01)  # 10 kPa per mm3
    half = chamber_model({"model": INCOMPRESSIBLE, "stiffness": 10.0, "fluid_volume": 50.0}, 100.0)
    assert half["liquid_volume"] == 50.0 and half["bulk_stiffness"] == pytest.approx(0.02)   # per % of 50 mm3
    over = chamber_model({"model": INCOMPRESSIBLE, "fluid_volume": 150.0}, 100.0)
    assert over["liquid_volume"] == 150.0

def test_legacy_linear_chambers_load_as_incompressible(tmp_path):
    import json
    from app.project import INCOMPRESSIBLE, INCOMPRESSIBLE_STIFFNESS
    path = tmp_path / "old.mns"
    path.write_text(json.dumps({"step_path": "x.step", "parts": [
        {"name": "C", "role": CHAMBER, "props": {"model": "Closed: linear stiffness", "stiffness": 0.001}}]}))
    part = Project.load(path).parts[0]
    assert part.props["model"] == INCOMPRESSIBLE and part.props["stiffness"] == INCOMPRESSIBLE_STIFFNESS


def test_chamber_colours_follow_the_model():
    from app.project import CONSTANT, INCOMPRESSIBLE, PartSettings, VENT, part_color, part_opacity
    def chamber(**props):
        p = PartSettings("c")
        p.set_role(CHAMBER)
        p.props.update(props)
        return p
    brightness = lambda c: sum(int(c[k:k + 2], 16) for k in (1, 3, 5))
    assert part_color(chamber(model=CONSTANT)) == "#2ca02c"
    assert part_color(chamber(model=INCOMPRESSIBLE)) == "#7b2cbf"
    assert brightness(part_color(chamber(model=IDEAL_GAS, incompressible=60.0))) < \
        brightness(part_color(chamber(model=IDEAL_GAS, incompressible=0.0)))
    assert part_opacity(chamber(model=VENT)) < 0.1


def test_incompressible_chamber_keeps_its_volume(neuron):
    from app.project import INCOMPRESSIBLE
    cad, project, path = neuron
    cad.load_step(path)
    saved = {p.name: dict(p.props) for p in project.parts}
    for p in project.parts:
        if p.role == CHAMBER:
            p.props.update({"Chamber_Left": dict(pressure=20.0), "Chamber_Right": dict(pressure=5.0),
                            "Chamber_Middle": dict(model=INCOMPRESSIBLE, stiffness=1000.0)}[p.name])
    try:
        build = build_environment(cad, generate_mesh(cad, project), project)
        assert build.solve(project.solver).converged
        middle = next(v for c, v in build.volumes.items() if project.parts[c].name == "Chamber_Middle")
        assert abs(middle.delta_volume) / middle.initial_volume < 5e-4  # below 0.05 %
        assert 5.0 < middle.P / 1e-3 < 20.0  # between the two inputs
    finally:
        for p in project.parts:
            p.props = saved[p.name]


def test_mesh_density_in_elements_per_shortest_side():
    from types import SimpleNamespace
    from app.builder import element_size
    plate = SimpleNamespace(size=np.array([100.0, 100.0, 5.0]), diagonal=float(np.linalg.norm([100, 100, 5])))
    assert element_size(plate, MEMBRANE) == pytest.approx(10.0)       # default 10 per shortest in-plane side
    assert element_size(plate, MEMBRANE, 20) == pytest.approx(5.0)
    strip = SimpleNamespace(size=np.array([200.0, 40.0, 2.0]), diagonal=float(np.linalg.norm([200, 40, 2])))
    assert element_size(strip, MEMBRANE) == pytest.approx(4.0)        # the 40 mm side, not the thickness
    thin_rigid = SimpleNamespace(size=np.array([1.0, 100.0, 100.0]), diagonal=float(np.linalg.norm([1, 100, 100])))
    assert element_size(thin_rigid, RIGID) == pytest.approx(thin_rigid.diagonal / 30)  # guarded


def test_membrane_thickness_is_measured_from_cad(neuron):
    from app.builder import measure_thickness, mesh_sizes
    cad, project, path = neuron
    cad.load_step(path)
    surfaces = cad.mesh(mesh_sizes(cad, project))
    index = next(b.index for b in cad.bodies if b.name == "Membrane_Left")
    assert measure_thickness(cad, surfaces, index) == pytest.approx(1.0)


def test_ghost_volume_adds_to_the_chamber_and_liquid_share_applies_to_the_total():
    from app.builder import chamber_model
    kw = chamber_model({"model": IDEAL_GAS, "incompressible": 50.0, "ghost_volume": 20.0}, 10.0)
    assert kw["initial_volume"] == pytest.approx(30.0)   # 10 mm3 body + 20 mm3 ghost
    assert kw["gas_volume"] == pytest.approx(15.0)       # 15 mm3 of the 30 is incompressible


def test_sweep_records_the_chamber_pressures_at_every_input_pressure(neuron):
    from types import SimpleNamespace
    from app.project import INCOMPRESSIBLE
    from app.sweep_core import run_sweep
    cad, project, path = neuron
    cad.load_step(path)
    saved = {p.name: dict(p.props) for p in project.parts}
    rows = []
    worker = SimpleNamespace(check=lambda: None, report=lambda *a: None, log=lambda *a: None,
                             item=SimpleNamespace(emit=rows.append))
    index = {p.name: i for i, p in enumerate(project.parts)}
    for p in project.parts:
        if p.role == CHAMBER:
            p.props.update({"Chamber_Left": dict(pressure=5.0), "Chamber_Right": dict(pressure=5.0),
                            "Chamber_Middle": dict(model=INCOMPRESSIBLE, stiffness=10.0)}[p.name])
    try:
        run_sweep(worker, cad, project, None, index["Chamber_Left"], np.array([10.0, 15.0, 20.0]), None, None)
        middle = index["Chamber_Middle"]
        assert len(rows) == 3 and all(r["converged"] for r in rows)
        rows.sort(key=lambda r: r["a"])
        assert [r["P"][index["Chamber_Left"]] for r in rows] == pytest.approx([10.0, 15.0, 20.0])
        pre = [r["P"][middle] for r in rows]
        assert 0 < pre[0] < pre[1] < pre[2] < 20.0          # the closed chamber follows the input
        assert all("W" not in r for r in rows)
    finally:
        for p in project.parts:
            p.props = saved[p.name]
