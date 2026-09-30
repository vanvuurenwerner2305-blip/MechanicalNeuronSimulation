"""Shared fixtures: a simulated squeeze-valve design (*.mad) with its swept volume (about a minute)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))


@pytest.fixture(scope="session")
def design(tmp_path_factory):
    """A squeeze-valve design with its swept volume, saved as a .mad file."""
    import make_activation_step
    from app.activation import ActivationProject, build_activation, generate_activation_mesh, run_study
    from app.cad import CadModel
    from app.project import CHANNEL, MEMBRANE, ORIFICE
    folder = tmp_path_factory.mktemp("valve")
    step = folder / "valve.step"
    make_activation_step.build(step)
    cad = CadModel()
    cad.load_step(step)
    project = ActivationProject(str(step), [b.name for b in cad.bodies])
    project.auto_assign_from_names()
    for part in project.parts:
        if part.name == "InletFluid":
            part.props["pressure"] = 10.0
        if part.name == "TubeFluid":
            part.props["segments"] = 4
        if part.role == CHANNEL:
            part.props["elements_per_side"] = 4
        if part.role == MEMBRANE:
            part.props["thickness"] = 0.5
            part.props["elements_per_side"] = 6
    project.connection("OutletFluid ↔ TubeFluid")["type"] = ORIFICE
    project.study.dp_min, project.study.dp_max, project.study.points = 0.0, 30.0, 4
    build = build_activation(cad, generate_activation_mesh(cad, project), project)
    project.results, _ = run_study(build, project)
    assert all(project.results["converged"])
    path = folder / "valve.mad"
    project.save(path)
    return path
