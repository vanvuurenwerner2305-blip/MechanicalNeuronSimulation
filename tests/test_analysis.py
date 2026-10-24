"""Analysis of a stored characterisation: slices, counted axes, interpolation on/off (no GUI, synthetic data)."""
import itertools

import numpy as np
import pytest

from app.analysis import axis_weights, evaluate_point, panels, samples, slice_grid
from app.full_neuron import Characterisation

AXES = [("A", [0.0, 10.0, 20.0]), ("B", [5.0, 0.0]), ("C", [0.0, 1.0, 2.0, 3.0]), ("D", [1.0, 2.0]),
        ("E", [0.0, 4.0, 8.0])]


def z_of(a, b, c, d, e):
    return a + 10 * b + 100 * c + 1000 * d + 10000 * e      # linear: interpolation is exact


@pytest.fixture
def dataset():
    shape = [len(v) for _, v in AXES]
    values, converged, extrapolated = [], [], []
    for idx in np.ndindex(*shape):
        x = [AXES[k][1][i] for k, i in enumerate(idx)]
        values.append(z_of(*x))
        converged.append(not (idx[0] == 2 and idx[1] == 0))   # A=20, B=5 did not converge
        extrapolated.append(idx[4] == 2)                      # E=8 extrapolated
    values[0] = None                                          # one point not solved
    converged[0] = None
    data = {"axes": [{"part": n, "field": "pressure", "label": n, "unit": "kPa", "values": v} for n, v in AXES],
            "keys": [{"key": "P:Out", "label": "P Out", "unit": "kPa"}], "values": {"P:Out": values},
            "converged": converged, "extrapolated": extrapolated, "complete": True}
    return Characterisation(data)


def test_samples_and_weights():
    assert samples([0, 10, 20, 30, 40], 3, False).tolist() == [0, 20, 40]
    assert samples([0, 10, 20, 30, 40], 9, False).tolist() == [0, 10, 20, 30, 40]
    assert samples([0, 10, 20, 30, 40], 3, True).tolist() == [0, 20, 40]
    assert samples([0, 10, 20, 30, 40], 2, True).tolist() == [0, 40]
    assert axis_weights([5.0, 0.0], 1.0, True) == [(1, pytest.approx(0.8)), (0, pytest.approx(0.2))]
    assert axis_weights([5.0, 0.0], 1.0, False) == [(1, 1.0)]
    assert axis_weights([0.0, 10.0], 99.0, True) == [(1, 1.0)]          # clamped, no extrapolation


def test_a_surface_slice_is_sorted_and_held_at_the_other_values(dataset):
    values = [a["values"] for a in dataset.axes]
    xs, z = slice_grid(dataset.grid("P:Out"), values, [1, 0], {2: 3.0, 3: 2.0, 4: 4.0}, False)
    assert xs[0].tolist() == [0.0, 5.0] and xs[1].tolist() == [0.0, 10.0, 20.0]
    assert z.shape == (2, 3)
    for (i, b), (j, a) in itertools.product(enumerate(xs[0]), enumerate(xs[1])):
        assert z[i, j] == z_of(a, b, 3.0, 2.0, 4.0)
    _, z = slice_grid(dataset.grid("P:Out"), values, [0], {1: 2.5, 2: 1.5, 3: 1.0, 4: 2.0}, True)
    assert np.allclose(z, [z_of(a, 2.5, 1.5, 1.0, 2.0) for a in (0.0, 10.0, 20.0)])


def test_panels_for_one_to_five_axes(dataset):
    fixed = {k: a["values"][0] for k, a in enumerate(dataset.axes)}
    one = panels(dataset, "P:Out", [2], fixed, {}, False)
    assert one["y"] is None and len(one["layers_data"]) == 1 and one["layers_data"][0][0][0]["z"].shape == (4,)
    five = panels(dataset, "P:Out", [0, 2, 1, 4, 3], fixed, {4: 2}, False)
    assert five["columns"][1].tolist() == [0.0, 5.0] and five["rows"][1].tolist() == [0.0, 8.0]
    assert five["layers"][1].tolist() == [1.0, 2.0]
    layers = five["layers_data"]
    assert len(layers) == 2 and len(layers[0]) == 2 and len(layers[0][0]) == 2
    p = layers[1][0][1]                                       # layer D=2, row E=0, column B=5
    assert p["at"] == {1: 5.0, 3: 2.0, 4: 0.0} and p["z"].shape == (3, 4)
    assert p["z"][2, 3] == z_of(20.0, 5.0, 3.0, 2.0, 0.0)
    assert p["unconverged"][2].all() and not p["unconverged"][:2].any()
    assert layers[1][1][0]["extrapolated"].all()              # E=8 row
    assert np.isnan(layers[0][0][1]["z"][0, 0])               # the unsolved point (B=5 column) is a gap
    assert five["zlim"] == (pytest.approx(z_of(0, 0, 0, 1, 0)), pytest.approx(z_of(20, 5, 3, 2, 8)))


def test_a_point_between_the_simulated_ones(dataset):
    point = {0: 15.0, 1: 2.5, 2: 2.5, 3: 1.5, 4: 6.0}
    values, _, extrapolated, converged = evaluate_point(dataset, point, True)
    assert values["P:Out"] == pytest.approx(z_of(15.0, 2.5, 2.5, 1.5, 6.0))
    assert extrapolated and not converged
    values, _, extrapolated, converged = evaluate_point(dataset, point, False)  # nearest (ties: the first)
    assert values["P:Out"] == z_of(10.0, 5.0, 2.0, 1.0, 4.0) and not extrapolated and converged
