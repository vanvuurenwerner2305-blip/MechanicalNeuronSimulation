"""
Static figures (PNG, for the agent to look at and for the LaTeX reports). One look throughout: series colours in
a fixed order (never cycled), one sequential blue ramp for magnitudes, thin lines, light grid, one y-axis per
plot, a legend whenever there are two or more series.
"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUES = LinearSegmentedColormap.from_list("mns_blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
INK, MUTED = "#1f1f1f", "#6b6b6b"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": "#b0b0b0", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2.0, "lines.markersize": 5, "legend.frameon": False,
    "legend.fontsize": 8,
})


def color(k):
    return SERIES[k % len(SERIES)] if k < len(SERIES) else MUTED


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _grid(n, width=4.2, height=3.0):
    cols = 1 if n == 1 else 2 if n <= 4 else 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    return fig, axes.ravel()[:n]


def lines(path, x, series, xlabel, ylabel, title=""):
    """series: [(label, y values)] over the same x."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for k, (label, y) in enumerate(series):
        ax.plot(x, y, "o-", color=color(k), label=label, markersize=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if len(series) > 1:
        ax.legend()
    return _save(fig, path)


def panels(path, x, panels_, xlabel, title=""):
    """panels_: [(ylabel, [(label, y)])], one small plot each, all over x."""
    fig, axes = _grid(len(panels_))
    for ax, (ylabel, series) in zip(axes, panels_):
        for k, (label, y) in enumerate(series):
            ax.plot(x, y, "o-", color=color(k), label=label, markersize=3.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if len(series) > 1:
            ax.legend()
    if title:
        fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def heatmap(path, x, y, z, xlabel, ylabel, zlabel, title="", diverging=False):
    """z[j, i] at (x[i], y[j])."""
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    z = np.asarray(z, float)
    if diverging:
        lim = np.nanmax(np.abs(z)) or 1.0
        cmap = LinearSegmentedColormap.from_list("mns_div", ["#1c5cab", "#86b6ef", "#efefef", "#f2a07f", "#b8441a"])
        image = ax.pcolormesh(x, y, z, cmap=cmap, vmin=-lim, vmax=lim, shading="nearest")
    else:
        image = ax.pcolormesh(x, y, z, cmap=BLUES, shading="nearest")
    fig.colorbar(image, ax=ax, label=zlabel)
    for j in range(len(y)):
        for i in range(len(x)):
            if np.isfinite(z[j, i]) and len(x) * len(y) <= 49:
                ax.text(x[i], y[j], f"{z[j, i]:.3g}", ha="center", va="center", fontsize=7, color=INK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    if title:
        ax.set_title(title)
    return _save(fig, path)


def scatter_compare(path, points, xlabel, ylabel, title=""):
    """points: [(label, x, y)] - one dot per design, labelled."""
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    ax.plot(xs, ys, "o", color=SERIES[0], markersize=7)
    for label, x, y in points:
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8, color=INK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return _save(fig, path)


def weight_fits(path, weights):
    """weights: [(title, dp samples, W samples, [(side label, dp curve, W curve)], xlabel)]."""
    fig, axes = _grid(len(weights), 4.4, 3.2)
    for ax, (title, dp, W, curves, xlabel) in zip(axes, weights):
        dp, W = np.asarray(dp, float), np.asarray(W, float)
        ok = np.isfinite(W)
        main = np.abs(dp) >= 0.05 * np.nanmax(np.abs(dp)) if len(dp) else ok
        ax.plot(dp[ok & main], W[ok & main], "o", color=SERIES[0], label="sampled W", markersize=4.5)
        for k, (label, x, y) in enumerate(curves):
            ax.plot(x, y, "-", color=SERIES[1 + k], label=label)
        shown = np.concatenate([W[ok & main]] + [np.asarray(y) for _, _, y in curves]) if (ok & main).any() else W
        shown = shown[np.isfinite(shown)]
        if len(shown):
            lo, hi = shown.min(), shown.max()
            pad = 0.15 * (hi - lo or abs(hi) or 1.0)
            ax.set_ylim(min(lo - pad, 0.0) if lo >= 0 else lo - pad, hi + pad)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("W [mm³/kPa]")
        ax.legend()
    return _save(fig, path)
