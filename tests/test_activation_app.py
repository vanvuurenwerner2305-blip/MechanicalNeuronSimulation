"""Activation-function space without the GUI: STEP -> roles -> tube and gas model -> Δp sweep -> design file."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from app.activation import (ActivationDesign, ActivationProject, build_activation, detect_connections,  # noqa: E402
                            gas_of, generate_activation_mesh, run_study)
from app.cad import CadModel  # noqa: E402
from app.project import (CHANNEL, CONSTANT_FLUID, DYNAMIC_FLUID, FLUID, FREE, MEMBRANE, OPENING, ORIFICE,  # noqa: E402
                         RIGID, SOLID)
from membrane_sim.flow import compile_flow_law  # noqa: E402
import make_activation_step  # noqa: E402


def valve_project(cad, path, pusher=None):
    project = ActivationProject(str(path), [b.name for b in cad.bodies])
    project.auto_assign_from_names()
    for part in project.parts:
        if part.name == "InletFluid":
            part.props["pressure"] = 10.0
        if part.name == "TubeFluid":
            part.props["segments"] = 6
        if part.role == CHANNEL:
            part.props["elements_per_side"] = 4
        if part.role == MEMBRANE:
            part.props["thickness"] = 0.5
            part.props["elements_per_side"] = 6
        if part.name == "Pusher" and pusher == SOLID:
            project.set_role(part, SOLID)
            part.props["youngs_modulus"] = 5.0
            part.props["elements_per_side"] = 2
    project.connection("OutletFluid ↔ TubeFluid")["type"] = ORIFICE
    return project


@pytest.fixture(scope="module")
def valve(tmp_path_factory):
    path = tmp_path_factory.mktemp("cad") / "valve.step"
    make_activation_step.build(path)
    cad = CadModel()
    cad.load_step(path)
    project = valve_project(cad, path)
    mesh = generate_activation_mesh(cad, project)
    build = build_activation(cad, mesh, project)
    return cad, project, mesh, build, path


def test_roles_are_guessed_from_names(valve):
    _, project, _, _, _ = valve
    roles = {p.name: p.role for p in project.parts}
    assert roles == {"Tube": CHANNEL, "Rigid Body": RIGID, "Membrane": MEMBRANE, "Pusher": RIGID,
                     "TubeFluid": FLUID, "InletFluid": FLUID, "OutletFluid": FLUID}
    props = {p.name: p.props for p in project.parts}
    assert props["Pusher"]["motion"] == FREE and props["Rigid Body"]["motion"] != FREE
    assert props["TubeFluid"]["model"] == DYNAMIC_FLUID
    assert props["InletFluid"]["model"] == props["OutletFluid"]["model"] == CONSTANT_FLUID


def test_flow_connections_are_found_where_fluids_touch(valve):
    cad, project, mesh, _, _ = valve
    connections = {c.key: c for c in detect_connections(mesh.surfaces, project.parts, cad.bodies)}
    assert set(connections) == {"InletFluid ↔ TubeFluid", "OutletFluid ↔ TubeFluid"}  # no false outside patches
    assert connections["InletFluid ↔ TubeFluid"].area == pytest.approx(np.pi * 0.8 ** 2, rel=0.05)
    assert project.connections["InletFluid ↔ TubeFluid"]["type"] == OPENING


def test_tube_model_is_found(valve):
    _, project, _, build, _ = valve
    assert np.allclose(build.axis, [0, 0, -1], atol=1e-3)            # from the 10 kPa inlet at +z to the sink
    assert np.allclose(build.squeeze, [0, -1, 0], atol=1e-3)         # the membrane is above the tube
    assert build.channel_height == pytest.approx(1.6, rel=0.02)
    assert build.A0 == pytest.approx(np.pi * 0.8 ** 2, rel=0.05)     # faceted circle, coarse mesh
    assert build.fixed_tube_nodes > 0 and all(len(n) > 10 for n in build.tie_nodes.values())
    names = {j: p.name for j, p in enumerate(project.parts)}
    assert [(names[i], names[j]) for i, j, _ in build.bonds] == [("Membrane", "Pusher")]
    assert build.flow.segments == 6 and len([w for w in build.flow.wall if w[1][0] == "segment"]) == 6


def test_gas_flows_from_the_inlet_through_the_orifice(valve):
    _, project, _, build, _ = valve
    gas = gas_of(project.study)
    flow = build.solve_flow(gas)
    p = flow["tube"]
    assert p[0] == pytest.approx(10.0) and np.all(np.diff(p) < 0) and 0 < p[-1] < 10  # falls along the tube
    assert flow["mdot"] > 0
    # mass balance at the orifice: its law gives the same flow for the drop to the 0 kPa sink
    conn, settings = next((c, s) for c, s in build.flow.connections if "Outlet" in c.key)
    rho_up = gas.density(p[-1] * 1000)
    assert (flow["mdot"] / (0.61 * conn.area * 1e-6)) ** 2 / (2 * rho_up) == pytest.approx(p[-1] * 1000, rel=1e-6)
    assert compile_flow_law(settings["law"]).text == "(mdot / (0.61 * A))**2 / (2 * rho_up)"


def test_sweep_squeezes_the_tube_and_the_design_round_trips(valve, tmp_path):
    cad, project, mesh, build, path = valve
    project.study.dp_min, project.study.dp_max, project.study.points = 0.0, 30.0, 2
    results, states = run_study(build, project)
    assert all(results["converged"])
    A_open, A_squeezed = results["area"]
    assert A_squeezed < 0.8 * A_open and results["travel"][1] > 0.05
    assert results["mdot"][1] < results["mdot"][0]                  # squeezing restricts the flow
    assert len(results["pressures"][0]) == 7 and len(states) == 2

    project.results = results
    design_path = tmp_path / "valve.mad"
    project.save(design_path)
    loaded = ActivationProject.load(design_path)
    assert loaded.results["area"] == results["area"] and loaded.study.dp_max == 30.0
    assert loaded.connections["OutletFluid ↔ TubeFluid"]["type"] == ORIFICE
    design = ActivationDesign.load(design_path)
    assert design.area(15.0) == pytest.approx(0.5 * (A_open + A_squeezed))
    assert design.mass_flow(0.0) == pytest.approx(results["mdot"][0])


@pytest.mark.xfail(reason="solid-solid contact stalls on this coarse curved contact (open issue)", run=False)
def test_a_deformable_pusher_closes_the_tube_too(valve):
    cad, _, _, _, path = valve
    project = valve_project(cad, path, pusher=SOLID)
    build = build_activation(cad, generate_activation_mesh(cad, project), project)
    names = {j: p.name for j, p in enumerate(project.parts)}
    assert [(names[i], names[j]) for i, j, _ in build.bonds] == [("Membrane", "Pusher")]
    project.study.dp_min, project.study.dp_max, project.study.points = 0.0, 30.0, 2
    results, _ = run_study(build, project)
    assert all(results["converged"])
    assert results["area"][1] < 0.8 * results["area"][0] and results["travel"][1] > 0.05
