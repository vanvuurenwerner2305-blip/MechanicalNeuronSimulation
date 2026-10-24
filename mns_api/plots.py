"""
Static figures (PNG, for the agent to look at and for the LaTeX reports). One look throughout: series colours in
a fixed order (never cycled), one sequential blue ramp for magnitudes, thin lines, light grid, one y-axis per
plot, a legend whenever there are two or more series.
"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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

