"""
Analysis of a stored characterisation (no GUI): slices of the swept grid for plotting and the response at any
point, either on the simulated points only (nearest) or interpolated (multilinear) between them.

The plot axes are given in the order they were chosen; their roles are ROLES: the first two span each panel (a curve
for one, a surface for two), the third sets the columns of a row of panels, the fourth the rows of a grid and the
fifth the layers (one grid shown at a time). Every other swept parameter is held at one value (its slider).
"""
import itertools
import math

import numpy as np

ROLES = ("x", "y", "columns", "rows", "layers")
COUNTED = ROLES[2:]          # roles shown at a number of values along the parameter (instead of all of them)


def sorted_values(values):
    return np.sort(np.asarray(values, float))


def samples(values, count, interpolate):
    """`count` values along an axis: evenly spaced in its range (interpolate) or evenly spaced simulated values."""
    v = sorted_values(values)
    count = max(int(count), 1)
    if len(v) == 1:
        return v[:1]
    if interpolate:
        return np.linspace(v[0], v[-1], count) if count > 1 else v[:1]
    if count >= len(v):
        return v
    return v[np.unique(np.round(np.linspace(0, len(v) - 1, count)).astype(int))]


def axis_weights(values, x, interpolate):
    """[(index into `values`, weight)] that give the value at x along one axis: the nearest simulated value, or the
    two around it (linear; clamped to the range, no extrapolation)."""
    values = np.asarray(values, float)
    if len(values) == 1:
        return [(0, 1.0)]
    if not interpolate:
        return [(int(np.argmin(np.abs(values - x))), 1.0)]
    order = np.argsort(values)
    s = values[order]
    x = min(max(float(x), s[0]), s[-1])
    j = int(np.clip(np.searchsorted(s, x), 1, len(s) - 1))
    t = (x - s[j - 1]) / (s[j] - s[j - 1]) if s[j] > s[j - 1] else 0.0
    out = [(int(order[j - 1]), 1.0 - t), (int(order[j]), t)]
    return [(i, w) for i, w in out if w > 1e-12] or [(int(order[j]), 1.0)]


def point_weights(axes_values, point, interpolate):
    """[(grid index tuple, weight)] of the response at `point` {axis index: value} (missing axes: first value)."""
    per_axis = [axis_weights(v, point.get(k, v[0]), interpolate) for k, v in enumerate(axes_values)]
    out = []
    for combo in itertools.product(*per_axis):
        out.append((tuple(i for i, _ in combo), math.prod(w for _, w in combo)))
    return out


def at_point(grid, weights):
    """A grid's value at a point from point_weights (NaN if any corner is not solved)."""
    return float(sum(w * grid[idx] for idx, w in weights))


def slice_grid(grid, axes_values, free, fixed, interpolate):
    """The grid over the `free` axes (in that order, each sorted by value) with every other axis at its value in
    `fixed` {axis: value}. Returns (list of the free axes' sorted values, array)."""
    held = [k for k in range(len(axes_values)) if k not in free]
    per_axis = [axis_weights(axes_values[k], fixed.get(k, axes_values[k][0]), interpolate) for k in held]
    total = None
    for combo in itertools.product(*per_axis):
        index = [slice(None)] * len(axes_values)
        for k, (i, _) in zip(held, combo):
            index[k] = i
        part = math.prod(w for _, w in combo) * np.asarray(grid[tuple(index)], float)
        total = part if total is None else total + part
    remaining = sorted(free)              # the order the free axes are left in after indexing
    total = np.transpose(total, [remaining.index(k) for k in free])
    xs = []
    for d, k in enumerate(free):
        order = np.argsort(np.asarray(axes_values[k], float))
        total = np.take(total, order, axis=d)
        xs.append(np.asarray(axes_values[k], float)[order])
    return xs, total


def panels(dataset, key, plot, fixed, counts, interpolate):
    """Everything to draw for Z = `key`: plot = axis indices in role order (1 to 5), fixed = {axis: value} of the
    other axes, counts = {axis: number of values} of the counted roles.

    Returns {"x": (axis, values), "y": (axis, values) or None, "columns"/"rows"/"layers": (axis, values) or None,
    "layers_data": [grid of panels per layer], "zlim": (lo, hi)}; a panel is {"at": {axis: value}, "z", "extrapolated",
    "unconverged"} with z over (x) or (x, y).
    """
    if not 1 <= len(plot) <= len(ROLES):
        raise ValueError(f"Choose 1 to {len(ROLES)} parameters to plot.")
    values = [a["values"] for a in dataset.axes]
    grid = dataset.grid(key)
    extrapolated = dataset.extrapolated.astype(float)
    unconverged = (dataset.solved & ~dataset.converged).astype(float)
    free = list(plot[:2])
    roles = {}
    for role, k in zip(COUNTED, plot[2:]):
        roles[role] = (k, samples(values[k], counts.get(k, len(values[k])), interpolate))
    layers = []
    lo, hi = math.inf, -math.inf
    for layer in (roles["layers"][1] if "layers" in roles else [None]):
        rows = []
        for r in (roles["rows"][1] if "rows" in roles else [None]):
            row = []
            for c in (roles["columns"][1] if "columns" in roles else [None]):
                at = dict(fixed)
                for role, v in (("columns", c), ("rows", r), ("layers", layer)):
                    if v is not None:
                        at[roles[role][0]] = float(v)
                at = {k: v for k, v in at.items() if k not in free}
                xs, z = slice_grid(grid, values, free, at, interpolate)
                _, ex = slice_grid(extrapolated, values, free, at, interpolate)
                _, un = slice_grid(unconverged, values, free, at, interpolate)
                if np.isfinite(z).any():
                    lo, hi = min(lo, float(np.nanmin(z))), max(hi, float(np.nanmax(z)))
                row.append({"at": at, "z": z, "extrapolated": ex > 0, "unconverged": un > 0})
            rows.append(row)
        layers.append(rows)
    if not math.isfinite(lo):
        lo, hi = 0.0, 1.0
    if hi <= lo:
        pad = max(abs(lo) * 0.05, 1e-9)
        lo, hi = lo - pad, hi + pad
    xs = [sorted_values(values[k]) for k in free]
    return {"x": (free[0], xs[0]), "y": (free[1], xs[1]) if len(free) > 1 else None,
            "columns": roles.get("columns"), "rows": roles.get("rows"), "layers": roles.get("layers"),
            "layers_data": layers, "zlim": (lo, hi)}


def evaluate_point(dataset, point, interpolate, shapes=None):
    """(values {key: value}, coords [per shape body] or None, extrapolated, converged) at `point` {axis: value}."""
    values = [a["values"] for a in dataset.axes]
    weights = point_weights(values, point, interpolate)
    out = {key: at_point(dataset.grid(key), weights) for key in dataset.keys}
    extrapolated = any(dataset.extrapolated[idx] for idx, w in weights if w > 0)
    converged = all(dataset.converged[idx] for idx, w in weights if w > 0)
    coords = None
    if shapes:
        flat = [(int(np.ravel_multi_index(idx, dataset.shape)) if dataset.shape else 0, w) for idx, w in weights]
        coords = [sum(w * b["frames"][f].astype(float) for f, w in flat) for b in shapes]
    return out, coords, extrapolated, converged
