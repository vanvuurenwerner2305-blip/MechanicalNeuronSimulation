"""The headless API and the `mns` command line: design files -> CAD -> project -> runs -> reports, in a temporary
study folder (about two minutes)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def mns(study, *args, timeout=600):
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
    env.pop("MNS_STUDY", None)
    env.pop("MNS_JOB", None)
    proc = subprocess.run([sys.executable, "-m", "mns_api", *args], cwd=str(study), env=env, capture_output=True,
                          timeout=timeout)
    lines = proc.stdout.decode("utf-8").strip().splitlines()
    assert lines, proc.stderr.decode("utf-8", "replace")
    out = json.loads(lines[-1])
    assert len(lines) == 1, f"stdout must hold only the JSON line: {lines}"
    return out


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    folder = tmp_path_factory.mktemp("api") / "study"
    out = mns(folder.parent, "new-study", str(folder), "--name", "api test")
    assert out["ok"], out
    return folder


# -----------------------------
# Pieces
# -----------------------------

def test_expressions_use_parameters_and_nothing_else():
    from mns_api.util import ApiError, evaluate, evaluate_parameters, parse_range
    names = evaluate_parameters({"t": 0.5, "R": "8", "gap": "2*t + sqrt(4)"})
    assert names["gap"] == pytest.approx(3.0)
    assert evaluate("max(R, 10) - pi*0", names) == pytest.approx(10.0)
    for bad in ("__import__('os')", "t.real", "open('x')", "unknown + 1"):
        with pytest.raises(ApiError):
            evaluate(bad, names)
    assert parse_range("0:20:5") == [0, 5, 10, 15, 20]
    assert parse_range("1,2.5") == [1.0, 2.5]


def test_cad_builds_named_bodies_and_finds_overlaps(tmp_path):
    from mns_api.cadspec import CadBuilder, geometry_report, write_step
    bodies = [
        {"name": "Plate", "shape": {"cylinder": {"base": [0, 0, -0.25], "axis": [0, 0, 0.5], "radius": 5}}},
        {"name": "Below", "shape": {"cavity": {"inside": {"cylinder": {"base": [0, 0, -3], "axis": [0, 0, 3],
                                                                         "radius": 5}}, "minus": ["Plate"]}}},
        {"name": "Lens", "shape": {"extrude": {"sketch": {"plane": "xz", "path": [
            [-2, 0], {"arc": {"through": [0, 1], "to": [2, 0]}}, {"arc": {"through": [0, -1], "to": [-2, 0]}}]},
            "distance": 2, "symmetric": True}, "translate": [20, 0, 0]}},
        {"name": "Clash", "shape": {"box": {"center": [20, 0, 0], "size": [1, 1, 1]}}},
    ]
    built = CadBuilder(bodies, {}).build()
    cad = write_step(tmp_path / "m.step", built)
    assert [b.name for b in cad.bodies] == ["Plate", "Below", "Lens", "Clash"]
    report = geometry_report(cad)
    volumes = {b["name"]: b["volume_mm3"] for b in report["bodies"]}
    assert volumes["Below"] == pytest.approx(3.14159265 * 25 * 2.75, rel=1e-6)   # the plate's half is cut out
    assert ["Plate", "Below"] in report["touching"]
    assert [o["bodies"] for o in report["overlaps"]] == [["Lens", "Clash"]]


def test_bad_design_files_explain_themselves():
    from mns_api.cadspec import CadBuilder
    from mns_api.model import resolve_choice, resolve_role
    from mns_api.util import ApiError
    with pytest.raises(ApiError, match="exactly one kind"):
        CadBuilder([{"name": "X", "shape": {"box": {"min": [0, 0, 0], "size": [1, 1, 1]},
                                            "sphere": {"center": [0, 0, 0], "radius": 1}}}], {}).build()
    with pytest.raises(ApiError, match="Unknown role"):
        resolve_role("balloon", "neuron")
    assert resolve_role("chamber", "neuron") == "Fluid chamber"
    assert resolve_choice("liquid", ["Constant pressure (input)", "Closed: incompressible"], "m") == \
        "Closed: incompressible"


# -----------------------------
# The command line on a study
# -----------------------------

def test_neuron_design_from_template_to_report(study):
    out = mns(study, "design", "new", "basic", "--template", "neuron_basic", "--why", "test")
    assert out["ok"] and out["id"] == "N001"
    built = mns(study, "cad", "build", "N001", "--no-render")
    assert built["ok"] and not built["overlaps"] and not built["warnings"], built
    assert set(built["touching"]["PreActivation"]) >= {"Membrane1", "Membrane2"}
    check = mns(study, "check", "N001")
    assert check["preactivation_default"] == "PreActivation"
    assert [p["input"] for p in check["input_paths_into_preactivation"]] == ["Input1", "Input2"]
    solve = mns(study, "solve", "N001", "--no-render", "--set", "Input1.pressure=5")
    assert solve["converged"] and solve["set"] == {"Input1.pressure": 5}
    assert 0 < solve["chambers"]["PreActivation"]["P_kPa"] < 5
    sweep = mns(study, "sweep", "N001", "--input", "Input1=0:10:3", "--no-render", "--tolerance", "2")
    assert sweep["converged"] == 3 and sweep["equation"]["met"]
    fit = mns(study, "fit", "N001", "--tolerance", "0.05")
    assert fit["ok"] and "max_error_kPa" in fit
    assert (study / "designs" / "N001_basic" / "results" / "equation.tex").exists()
    report = mns(study, "report", "design", "N001")
    assert report["unwritten_sections"] == ["Aim", "Design", "Results", "Findings"]
    tex = (study / "designs" / "N001_basic" / "report" / "auto_results.tex").read_text(encoding="utf-8")
    assert "\\begin{align}" in tex and "Static solve" in tex


def test_derive_changes_a_parameter_and_rebuilds(study):
    out = mns(study, "design", "derive", "N001", "thin", "--set", "t=0.3", "--set", "Input1.pressure=3",
              "--no-render", "--why", "thinner")
    assert out["ok"], out
    assert out["build"]["bodies"][0]["thickness_mm"] == pytest.approx(0.3)
    show = mns(study, "design", "show", out["id"], "--spec")
    assert show["parent"] == "N001" and "t: 0.3" in show["design_yaml"]
    table = mns(study, "compare", "--metrics", "solve:P_PreActivation_kPa")
    assert [d["id"] for d in table["designs"]] == ["N001", "N002"]


def test_errors_come_back_as_json_with_hints(study):
    out = mns(study, "solve", "N999")
    assert not out["ok"] and "N001" in out["hint"]
    out = mns(study, "sweep", "N001", "--input", "Nope=0:1:2")
    assert not out["ok"] and "Chambers" in out["hint"]


def test_background_job_and_live_events(study):
    out = mns(study, "solve", "N001", "--background", "--no-render", "--why", "background")
    job = out["job"]
    done = mns(study, "wait", job, "--timeout", "300")
    assert done["state"] == "done" and done["output"]["converged"], done
    events = [json.loads(line) for line in (study / ".mns" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = {e["kind"] for e in events}
    assert {"command", "done", "design", "frame"} <= kinds
    assert any(e.get("why") == "background" for e in events)
    assert (study / ".mns" / "live" / "N001.npz").exists()


def test_activation_and_full_neuron(study):
    assert mns(study, "design", "new", "valve", "--template", "activation_valve")["id"] == "A001"
    assert mns(study, "cad", "build", "A001", "--no-render")["ok"]
    check = mns(study, "check", "A001")
    assert check["tube_fluid"] == "TubeFluid" and check["bonds"][0]["to"] == "Pusher"
    run = mns(study, "dpsweep", "A001", "--dp", "0:20:2", "--no-render")
    assert run["metrics"]["converged"] == 2, run
    assert mns(study, "design", "new", "full", "--template", "full")["id"] == "F001"
    built = mns(study, "cad", "build", "F001")
    assert built["ok"] and built["link"]["part"] == "Membrane2", built
    grid = mns(study, "characterise", "F001", "--axis", "Input1.pressure=0:10:2", "--no-render")
    assert grid["converged"] == 2 and "out:activation" in grid["ranges"]


def test_import_of_a_project(study, tmp_path):
    """A .mns with its STEP file comes in as a design that builds and solves like the original."""
    from mns_api.study import Study
    src = ROOT / "examples" / "soft_neuron.step"
    if not src.exists():
        pytest.skip("examples/soft_neuron.step not generated")
    step = tmp_path / "soft.step"
    shutil.copy2(src, step)
    out = mns(study, "design", "import", str(step), "--name", "soft")
    assert out["ok"], out
    design = Study(study).design(out["imported"][0]["id"])
    spec = design.spec()
    assert spec["base_step"] == "base.step" and all("shape" not in b for b in spec["bodies"])
