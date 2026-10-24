"""Neuron network: characterised neurons as blocks, wired and evaluated from the stored data (no GUI, synthetic data)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.network import (CLAMP, CONSTANT, CURRENT, GAIN, INPUT, MISSING, NEURON, OUTDATED, OUTPUT, ROLE_CONSTANT, ROLE_PARAMETER, SUM,
                         VALVE, Network, component_from_design, component_from_mfn, source_status, sweep_csv)

P = [-20.0, -10.0, 0.0, 10.0, 20.0]
T = [0.0, 0.2]


def neuron_mfn(path, w=0.4, bias=0.0):
    """A full-neuron file whose characterisation is p_a = w (p1 + p2) + bias - 10 T (exact under interpolation)."""
    values, dv = [], []
    for p1 in P:
        for p2 in [20.0, 0.0, -20.0]:            # a decreasing axis
            for t in T:
                values.append(w * (p1 + p2) + bias - 10 * t)
                dv.append(p1 - p2)
    c = {"axes": [{"part": "In1", "field": "pressure", "label": "P In1", "unit": "kPa", "values": P},
                  {"part": "In2", "field": "pressure", "label": "P In2", "unit": "kPa", "values": [20.0, 0.0, -20.0]},
                  {"part": "A", "also": ["B"], "field": "pretension", "label": "T", "unit": "N/mm", "values": T}],
         "keys": [{"key": "P:Pre-Activation", "label": "P Pre-Activation", "unit": "kPa"},
                  {"key": "dV:In1", "label": "dV In1", "unit": "mm3"}],
         "values": {"P:Pre-Activation": values, "dV:In1": dv}, "converged": [True] * len(values),
         "extrapolated": [False] * len(values), "complete": True, "shapes": [{"name": "big"}]}
    path.write_text(json.dumps({"type": "full_neuron", "version": 2, "characterisation": c}), encoding="utf-8")
    return path


@pytest.fixture
def net(tmp_path):
    n = Network()
    cid = n.add_component(component_from_mfn(neuron_mfn(tmp_path / "lin.mfn")))
    return n, cid


def test_component_and_default_ports(net):
    n, cid = net
    c = n.components[cid]
    assert [a["name"] for a in c["axes"]] == ["In1.pressure", "In2.pressure", "A+B.pretension"]
    assert "shapes" not in c
    b = n.add_block(NEURON, cid)
    ins, outs = n.ports(b)
    assert ins == ["In1.pressure", "In2.pressure"]          # pressure axes are inputs, the rest parameters
    assert b["axes"]["A+B.pretension"]["role"] == ROLE_CONSTANT
    assert outs == ["P:Pre-Activation"]


def test_feed_forward_chain(net):
    n, cid = net
    a, b = n.add_block(INPUT, name="a"), n.add_block(INPUT, name="b")
    n1, n2 = n.add_block(NEURON, cid), n.add_block(NEURON, cid)
    k = n.add_block(CONSTANT)
    k["value"] = 5.0
    out = n.add_block(OUTPUT, name="y")
    for blk in (n1, n2):
        blk["axes"]["A+B.pretension"]["value"] = 0.1
    n.connect((a["id"], "out"), (n1["id"], "In1.pressure"))
    n.connect((b["id"], "out"), (n1["id"], "In2.pressure"))
    n.connect((n1["id"], "P:Pre-Activation"), (n2["id"], "In1.pressure"))
    n.connect((k["id"], "out"), (n2["id"], "In2.pressure"))
    n.connect((n2["id"], "P:Pre-Activation"), (out["id"], "in"))
    r = n.solve({"a": 7.0, "b": -3.0})
    h = 0.4 * (7.0 - 3.0) - 1.0
    assert r["converged"] and not r["loop"] and not r["flags"]
    assert r["probes"]["y"] == pytest.approx(0.4 * (h + 5.0) - 1.0, abs=1e-12)


def test_unconnected_input_port_uses_its_value(net):
    n, cid = net
    b = n.add_block(NEURON, cid)
    b["axes"]["In2.pressure"]["value"] = 10.0
    b["axes"]["A+B.pretension"]["value"] = 0.0
    out = n.add_block(OUTPUT)
    n.connect((b["id"], "P:Pre-Activation"), (out["id"], "in"))
    assert n.solve()["probes"][out["name"]] == pytest.approx(4.0)


def test_extrapolate_or_clamp(net):
    n, cid = net
    i = n.add_block(INPUT, name="x")
    b = n.add_block(NEURON, cid)
    b["axes"]["A+B.pretension"]["value"] = 0.0
    out = n.add_block(OUTPUT, name="y")
    n.connect((i["id"], "out"), (b["id"], "In1.pressure"))
    n.connect((b["id"], "P:Pre-Activation"), (out["id"], "in"))
    r = n.solve({"x": 30.0})
    assert r["probes"]["y"] == pytest.approx(12.0) and r["flags"][b["id"]] == ["In1.pressure"]
    b["range"] = CLAMP
    assert n.solve({"x": 30.0})["probes"]["y"] == pytest.approx(8.0)


def test_feedback_loop(net):
    """y = 0.4 (x + y) - T*10 with T = 0: y = 0.4 x / 0.6."""
    n, cid = net
    i = n.add_block(INPUT, name="x")
    b = n.add_block(NEURON, cid)
    b["axes"]["A+B.pretension"]["value"] = 0.0
    out = n.add_block(OUTPUT, name="y")
    n.connect((i["id"], "out"), (b["id"], "In1.pressure"))
    n.connect((b["id"], "P:Pre-Activation"), (b["id"], "In2.pressure"))
    n.connect((b["id"], "P:Pre-Activation"), (out["id"], "in"))
    r = n.solve({"x": 9.0})
    assert r["loop"] and r["converged"]
    assert r["probes"]["y"] == pytest.approx(0.4 * 9.0 / 0.6, abs=1e-8)


def test_sum_gain_and_sweep(net):
    n, cid = net
    x, y = n.add_block(INPUT, name="x"), n.add_block(INPUT, name="y")
    s, g = n.add_block(SUM), n.add_block(GAIN)
    s["signs"] = "+-"
    g["k"] = 2.0
    out = n.add_block(OUTPUT, name="z")
    n.connect((x["id"], "out"), (s["id"], "in1"))
    n.connect((y["id"], "out"), (s["id"], "in2"))
    n.connect((s["id"], "out"), (g["id"], "in"))
    n.connect((g["id"], "out"), (out["id"], "in"))
    sw = n.run_sweep("x", np.linspace(0, 4, 5), "y", [1.0, 2.0, 3.0])
    assert sw["probes"]["z"].shape == (3, 5)
    assert np.allclose(sw["probes"]["z"], 2 * (np.linspace(0, 4, 5)[None, :] - np.array([1.0, 2.0, 3.0])[:, None]))
    assert sweep_csv(sw).count("\n") == 16


def test_ports_follow_roles_and_prune(net):
    n, cid = net
    i = n.add_block(INPUT)
    b = n.add_block(NEURON, cid)
    n.connect((i["id"], "out"), (b["id"], "In1.pressure"))
    b["axes"]["In1.pressure"]["role"] = ROLE_PARAMETER
    assert n.prune_wires() == 1 and not n.wires
    with pytest.raises(ValueError):
        n.connect((i["id"], "out"), (b["id"], "In1.pressure"))


def test_valve_block():
    design = SimpleNamespace(output_names=["activation"], p_out={"activation": np.array([10.0, 5.0, 0.0])},
                             dp=np.array([0.0, 10.0, 20.0]), A=np.array([1.0, 0.5, 0.0]), mdot=np.zeros(3),
                             p_end=np.zeros(3), volume=None)
    n = Network()
    cid = n.add_component(component_from_design(design, "valve"))
    i, v, out = n.add_block(INPUT, name="dp"), n.add_block(VALVE, cid), n.add_block(OUTPUT, name="act")
    assert n.ports(v) == (["dp"], ["out:activation"])
    n.connect((i["id"], "out"), (v["id"], "dp"))
    n.connect((v["id"], "out:activation"), (out["id"], "in"))
    assert n.solve({"dp": 15.0})["probes"]["act"] == pytest.approx(2.5)
    assert n.solve({"dp": 40.0})["probes"]["act"] == pytest.approx(0.0)     # a valve is held at its end values


def test_save_load_and_source_status(net, tmp_path):
    n, cid = net
    b = n.add_block(NEURON, cid)
    o = n.add_block(OUTPUT)
    n.connect((b["id"], "P:Pre-Activation"), (o["id"], "in"))
    n.save(tmp_path / "net.mnn")
    m = Network.load(tmp_path / "net.mnn")
    assert m.components[cid]["source"] == str((tmp_path / "lin.mfn").resolve())
    assert source_status(m.components[cid]) == CURRENT
    assert m.solve()["probes"] == pytest.approx(n.solve()["probes"])
    neuron_mfn(tmp_path / "lin.mfn", bias=1.0)                    # the source is characterised again
    assert source_status(m.components[cid]) == OUTDATED
    m.update_component(cid)
    assert source_status(m.components[cid]) == CURRENT
    assert m.wires and m.solve()["probes"][o["name"]] == pytest.approx(n.solve()["probes"][o["name"]] + 1.0)
    (tmp_path / "lin.mfn").unlink()
    assert source_status(m.components[cid]) == MISSING


def test_mfn_without_characterisation(tmp_path):
    p = tmp_path / "x.mfn"
    p.write_text(json.dumps({"type": "full_neuron", "characterisation": None}), encoding="utf-8")
    with pytest.raises(ValueError, match="no stored characterisation"):
        component_from_mfn(p)


def test_window_builds_wires_evaluates_and_sweeps(tmp_path):
    """The Network tab without showing it: blocks, a wire through the canvas API, live values, a sweep, save/open."""
    import app  # noqa: F401  (sets QT_API)
    from qtpy.QtCore import QPointF
    from qtpy.QtWidgets import QApplication
    qa = QApplication.instance() or QApplication([])
    from app.network_window import NetworkWindow
    w = NetworkWindow()
    cid = w.add_component(str(neuron_mfn(tmp_path / "lin.mfn")))
    x, n1, y = w.add_block(INPUT, None, QPointF(0, 0)), w.add_block(NEURON, cid, QPointF(200, 0)), \
        w.add_block(OUTPUT, None, QPointF(500, 0))
    n1["axes"]["A+B.pretension"]["value"] = 0.0
    w.connect_ports((x["id"], "out"), (n1["id"], "In1.pressure"))
    w.connect_ports((n1["id"], "P:Pre-Activation"), (y["id"], "in"))
    w._set_input(x, 10.0)
    assert w.result["probes"]["Output"] == pytest.approx(4.0)
    assert w.items[n1["id"]].value_items["P:Pre-Activation"].text().startswith("4 kPa")
    w.run_sweep()
    assert w.sweep_result["probes"]["Output"].shape == (1, 41)
    w.network.save(tmp_path / "net.mnn")
    w.open_project(str(tmp_path / "net.mnn"))
    assert len(w.wire_items) == 2 and w.result["probes"]["Output"] == pytest.approx(4.0)
    w.close()
    qa.processEvents()
