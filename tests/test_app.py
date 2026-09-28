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
