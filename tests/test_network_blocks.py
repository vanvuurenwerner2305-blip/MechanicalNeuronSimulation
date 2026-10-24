"""Neuron network: the block library (math, activation, logic, lookup) and subsystems (no GUI, synthetic data)."""
import numpy as np
import pytest

from app.network import (GAIN, INPORT, INPUT, NEURON, OUTPORT, OUTPUT, SUBSYSTEM, TYPES, Network,
                         component_from_mfn)
from test_network import neuron_mfn


@pytest.fixture
def net(tmp_path):
    n = Network()
    cid = n.add_component(component_from_mfn(neuron_mfn(tmp_path / "lin.mfn")))
    return n, cid


def chain(n, kind, settings, value, port="in"):
    """Input -> block -> Output; the output for one input value."""
    i, b, o = n.add_block(INPUT, name="u"), n.add_block(kind), n.add_block(OUTPUT, name="y")
    b.update(settings)
    n.connect((i["id"], "out"), (b["id"], port))
    n.connect((b["id"], "out"), (o["id"], "in"))
    return n.solve({"u": value})["probes"]["y"]


@pytest.mark.parametrize("kind, settings, u, expected", [
    ("sigmoid", {"k": 0.5, "x0": 2.0, "low": 0.0, "high": 20.0}, 2.0, 10.0),
    ("sigmoid", {"k": 0.5, "x0": 0.0, "low": 0.0, "high": 1.0}, 4.0, 1 / (1 + np.exp(-2.0))),
    ("tanh", {"k": 0.1, "x0": 0.0, "amplitude": 3.0}, 5.0, 3 * np.tanh(0.5)),
    ("relu", {"threshold": 1.0, "slope": 2.0}, 4.0, 6.0),
    ("relu", {"threshold": 1.0, "slope": 2.0}, -4.0, 0.0),
    ("saturation", {"lower": -1.0, "upper": 2.0}, 5.0, 2.0),
    ("deadzone", {"start": -1.0, "end": 1.0}, 3.0, 2.0),
    ("step", {"threshold": 0.0, "low": -1.0, "high": 5.0}, 0.0, 5.0),
    ("bias", {"b": -2.5}, 1.0, -1.5),
    ("abs", {}, -3.0, 3.0),
    ("power", {"n": 3.0}, 2.0, 8.0),
    ("log", {}, -1.0, float("nan")),
    ("lookup", {"x": "0, 10, 20", "y": "0, 5, 20"}, 15.0, 12.5),
    ("lookup", {"x": "0, 10, 20", "y": "0, 5, 20"}, 40.0, 20.0),
    ("compare_const", {"op": ">=", "c": 2.0}, 2.0, 1.0),
    ("not", {}, 1.0, 0.0),
    ("function", {"inputs": 1, "expr": "20*sigmoid(0.3*(u1 - 5)) + pi*0"}, 5.0, 10.0),
])
def test_single_input_blocks(kind, settings, u, expected):
    y = chain(Network(), kind, settings, u)
    assert (np.isnan(y) and np.isnan(expected)) or y == pytest.approx(expected)


def test_multi_input_logic_and_switch():
    n = Network()
    a, b, c = (n.add_block(INPUT, name=x) for x in "abc")
    blocks = {k: n.add_block(k) for k in ("and", "or", "xor", "nand", "compare", "switch", "product", "max")}
    blocks["product"]["ops"] = "*/"
    blocks["compare"]["op"] = "<"
    for k, blk in blocks.items():
        for src, port in zip((a, b, c), n.ports(blk)[0]):
            n.connect((src["id"], "out"), (blk["id"], port))
        out = n.add_block(OUTPUT, name=k)
        n.connect((blk["id"], "out"), (out["id"], "in"))
    p = n.solve({"a": 1.0, "b": 0.0, "c": 7.0})["probes"]
    assert (p["and"], p["or"], p["xor"], p["nand"], p["compare"]) == (0.0, 1.0, 1.0, 1.0, 0.0)
    assert p["switch"] == 7.0 and np.isnan(p["product"]) and p["max"] == 1.0      # 1 / 0 -> NaN
    p = n.solve({"a": 3.0, "b": 2.0, "c": 7.0})["probes"]
    assert (p["and"], p["switch"], p["product"], p["compare"]) == (1.0, 3.0, 1.5, 0.0)


def test_every_registered_type_builds_and_evaluates():
    n = Network()
    for kind in TYPES:
        if kind in (NEURON, "valve"):
            continue
        b = n.add_block(kind)
        assert isinstance(n.ports(b), tuple)
    n.solve()


def full_neuron(n, cid):
    """p1, p2 -> neuron -> sigmoid -> act, as on the canvas before grouping."""
    p1, p2 = n.add_block(INPUT, name="p1"), n.add_block(INPUT, name="p2")
    neuron, sig, out = n.add_block(NEURON, cid), n.add_block("sigmoid"), n.add_block(OUTPUT, name="act")
    neuron["axes"]["A+B.pretension"]["value"] = 0.0
    sig.update(k=0.4, high=20.0)
    n.connect((p1["id"], "out"), (neuron["id"], "In1.pressure"))
    n.connect((p2["id"], "out"), (neuron["id"], "In2.pressure"))
    n.connect((neuron["id"], "P:Pre-Activation"), (sig["id"], "in"))
    n.connect((sig["id"], "out"), (out["id"], "in"))
    return p1, p2, neuron, sig, out


def test_group_into_subsystem_keeps_the_result(net):
    """Neuron + sigmoid grouped into one block: same output, ports made from the crossing wires."""
    n, cid = net
    p1, p2, neuron, sig, out = full_neuron(n, cid)
    before = n.solve({"p1": 6.0, "p2": -2.0})["probes"]["act"]
    sub = n.group([neuron["id"], sig["id"]], name="Full neuron")
    d = n.definition_of(sub)
    ins, outs = n.ports(sub)
    assert len(ins) == 2 and len(outs) == 1 and len(n.blocks) == 4 and len(n.wires) == 3
    assert [b["name"] for b in d["blocks"] if b["type"] == INPORT] == ["p1", "p2"]
    r = n.solve({"p1": 6.0, "p2": -2.0})
    assert r["probes"]["act"] == pytest.approx(before) == pytest.approx(20 / (1 + np.exp(-0.4 * 0.4 * 4)))
    assert r["values"][(sub["id"], outs[0])] == pytest.approx(before)          # shown on the subsystem's port
    # a second instance shares the definition; a loop through the subsystem boundary is solved
    second = n.add_block(SUBSYSTEM, definition=sub["definition"])
    n.connect((p1["id"], "out"), (second["id"], ins[0]))
    n.connect((second["id"], outs[0]), (second["id"], ins[1]))
    o2 = n.add_block(OUTPUT, name="loop")
    n.connect((second["id"], outs[0]), (o2["id"], "in"))
    r = n.solve({"p1": 6.0, "p2": -2.0})
    y = r["probes"]["loop"]
    assert r["converged"] and y == pytest.approx(20 / (1 + np.exp(-0.4 * 0.4 * (6.0 + y))), abs=1e-7)
    assert len(n.instances(sub["definition"])) == 2


def test_nested_subsystems_ports_and_recursion(net):
    n, _ = net
    did = n.new_definition("Inner")
    d = n.definitions[did]
    i = n.add_block(INPORT, container=d, name="x")
    g = n.add_block(GAIN, container=d)
    g["k"] = 3.0
    o = n.add_block(OUTPORT, container=d, name="y")
    n.connect((i["id"], "out"), (g["id"], "in"), d)
    n.connect((g["id"], "out"), (o["id"], "in"), d)
    outer = n.new_definition("Outer")
    od = n.definitions[outer]
    oi = n.add_block(INPORT, container=od)
    inner = n.add_block(SUBSYSTEM, container=od, definition=did)
    oo = n.add_block(OUTPORT, container=od)
    n.connect((oi["id"], "out"), (inner["id"], i["id"]), od)
    n.connect((inner["id"], o["id"]), (oo["id"], "in"), od)
    with pytest.raises(ValueError, match="itself"):
        n.add_block(SUBSYSTEM, container=d, definition=outer)
    src, top, probe = n.add_block(INPUT, name="u"), n.add_block(SUBSYSTEM, definition=outer), n.add_block(OUTPUT)
    n.connect((src["id"], "out"), (top["id"], oi["id"]))
    n.connect((top["id"], oo["id"]), (probe["id"], "in"))
    assert n.solve({"u": 2.0})["probes"][probe["name"]] == pytest.approx(6.0)
    n.remove_block(o["id"], d)                        # a port removed: the wires to it go, at every level
    assert not any(w["from"][0] == inner["id"] for w in od["wires"])
    assert np.isnan(n.solve({"u": 2.0})["probes"][probe["name"]])


def test_library_block_export_import(net, tmp_path):
    n, cid = net
    _, _, neuron, sig, _ = full_neuron(n, cid)
    sub = n.group([neuron["id"], sig["id"]], name="Full neuron")
    n.export_definition(sub["definition"], tmp_path / "full.mnb")
    m = Network()
    did = m.import_definition(tmp_path / "full.mnb")
    assert m.definitions[did]["name"] == "Full neuron" and len(m.components) == 1
    b = m.add_block(SUBSYSTEM, definition=did)
    x, y, out = m.add_block(INPUT, name="a"), m.add_block(INPUT, name="b"), m.add_block(OUTPUT, name="z")
    ins, outs = m.ports(b)
    m.connect((x["id"], "out"), (b["id"], ins[0]))
    m.connect((y["id"], "out"), (b["id"], ins[1]))
    m.connect((b["id"], outs[0]), (out["id"], "in"))
    z = m.solve({"a": 6.0, "b": -2.0})["probes"]["z"]
    assert z == pytest.approx(n.solve({"p1": 6.0, "p2": -2.0})["probes"]["act"])
    m.save(tmp_path / "m.mnn")
    assert Network.load(tmp_path / "m.mnn").solve({"a": 6.0, "b": -2.0})["probes"]["z"] == pytest.approx(z)
    assert m.import_definition(tmp_path / "full.mnb") != did and len(m.components) == 1   # same data: reused


def test_inputs_parameters_and_constants(net):
    """A variable is an input port, a trainable parameter (bounded by the characterised range) or a constant."""
    n, cid = net
    b = n.add_block(NEURON, cid)
    assert b["axes"]["A+B.pretension"]["role"] == "constant" and n.parameters() == []
    b["axes"]["A+B.pretension"]["role"] = "parameter"
    sig = n.add_block("sigmoid")
    sig["trainable"] = ["k", "x0"]
    did = n.new_definition("Inner")
    g = n.add_block(GAIN, container=n.definitions[did])
    g["trainable"] = ["k"]
    params = {p["name"]: p for p in n.parameters()}
    assert set(params) == {f"{b['name']}.A+B.pretension", "Sigmoid.k", "Sigmoid.x0", "Gain.k"}
    assert params["Gain.k"]["where"] == "Inner" and params["Sigmoid.k"]["min"] is None
    p = params[f"{b['name']}.A+B.pretension"]
    assert (p["min"], p["max"]) == (0.0, 0.2)
    assert n.set_parameter(p, 0.5) == 0.2 and b["axes"]["A+B.pretension"]["value"] == 0.2   # kept in range
    n.set_parameter(params["Sigmoid.k"], 2.0)
    assert sig["k"] == 2.0
    # held either way: a constant and a parameter at the same value give the same output
    o = n.add_block(OUTPUT)
    n.connect((b["id"], "P:Pre-Activation"), (o["id"], "in"))
    y = n.solve()["probes"][o["name"]]
    b["axes"]["A+B.pretension"]["role"] = "constant"
    assert n.solve()["probes"][o["name"]] == y == pytest.approx(-2.0)


def test_analyse_a_block_on_its_own(net):
    """A subsystem (neuron -> sigmoid) evaluated alone on a grid of its inputs and a parameter inside it, as a dataset
    the Analysis tab reads; the network is left as it was."""
    from app.full_neuron import Characterisation
    n, cid = net
    _, _, neuron, sig, _ = full_neuron(n, cid)
    sub = n.group([neuron["id"], sig["id"]], name="Full neuron")
    inner_sig = next(b for b in n.definition_of(sub)["blocks"] if b["type"] == "sigmoid")
    inner_sig["trainable"] = ["k"]
    ins, outs = n.ports(sub)
    assert n.port_range(sub, ins[0]) == (-20.0, 20.0)              # followed to the neuron's variable
    assert n.port_range(inner_sig, "in") == pytest.approx((-15.0, 15.0))   # the sigmoid's typical range, k 0.4
    (prm,) = n.block_parameters(sub)
    sweeps = [{"kind": "input", "port": ins[0], "label": "p1", "unit": "kPa", "values": [-10.0, 0.0, 10.0]},
              {"kind": "parameter", "parameter": prm, "label": "k", "unit": "", "values": [0.2, 0.4]}]
    data = n.analyse_block(sub, sweeps, held={ins[1]: 5.0})
    c = Characterisation(data)
    assert c.shape == (3, 2) and c.names == ["p1.input", "k.parameter"] and c.keys == outs
    pa = 0.4 * (10.0 + 5.0)
    assert c.evaluate({"p1.input": 10.0, "k.parameter": 0.2})[outs[0]] == pytest.approx(20 / (1 + np.exp(-0.2 * pa)))
    assert inner_sig["k"] == 0.4 and len(n.blocks) == 4             # nothing changed
    with pytest.raises(ValueError):
        n.analyse_block(n.blocks[0], [])                              # an Input block


def test_window_groups_opens_and_edits_a_subsystem(tmp_path):
    """The Network tab without showing it: Ctrl+G on a neuron and a sigmoid, open it, change the sigmoid, go up."""
    import app  # noqa: F401  (sets QT_API)
    from qtpy.QtCore import QPointF
    from qtpy.QtWidgets import QApplication
    qa = QApplication.instance() or QApplication([])
    from app.network_window import NetworkWindow
    w = NetworkWindow()
    cid = w.add_component(str(neuron_mfn(tmp_path / "lin.mfn")))
    p1, neuron = w.add_block(INPUT, None, QPointF(0, 0)), w.add_block(NEURON, cid, QPointF(200, 0))
    sig, out = w.add_block("sigmoid", None, QPointF(500, 0)), w.add_block(OUTPUT, None, QPointF(700, 0))
    neuron["axes"]["A+B.pretension"]["value"] = 0.0
    w.connect_ports((p1["id"], "out"), (neuron["id"], "In1.pressure"))
    w.connect_ports((neuron["id"], "P:Pre-Activation"), (sig["id"], "in"))
    w.connect_ports((sig["id"], "out"), (out["id"], "in"))
    w._set_input(p1, 5.0)
    before = w.result["probes"]["Output"]
    w.scene.clearSelection()
    for b in (neuron, sig):
        w.items[b["id"]].setSelected(True)
    w.group_selected("Full neuron")
    assert len(w.network.blocks) == 3 and w.result["probes"]["Output"] == pytest.approx(before)
    sub = next(b for b in w.network.blocks if b["type"] == SUBSYSTEM)
    w.enter(sub)
    assert w.path == [sub["id"]] and {b["type"] for b in w.container()["blocks"]} == {INPORT, NEURON, "sigmoid",
                                                                                      OUTPORT}
    inner_sig = next(b for b in w.container()["blocks"] if b["type"] == "sigmoid")
    w._set(inner_sig, "high", 2.0)                    # edit inside: the top-level output follows
    assert w.result["probes"]["Output"] == pytest.approx(2.0 * before)   # default high level 1 -> 2
    w.go_up()
    assert w.path == [] and sub["id"] in w.items
    w.dirty = False
    w.close()
    qa.processEvents()
