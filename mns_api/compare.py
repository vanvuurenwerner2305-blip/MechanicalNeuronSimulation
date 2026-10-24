"""Comparing designs: their parameters next to the metrics their runs recorded, optionally one plotted against
another (results go to <study>/report/figures so the study report can use them)."""
import math

from . import plots
from .util import ApiError, rounded


def design_row(design):
    spec, state = design.spec(), design.state()
    try:
        params = design.parameters() if design.space != "full" else {}
    except Exception:
        params = {}
    row = {"id": design.id, "name": design.name, "space": design.space, "parent": spec.get("parent"),
           "status": state.get("status")}
    row.update({f"p:{k}": v for k, v in params.items()})
    if design.space == "full":
        row.update({"neuron": spec.get("neuron"), "activation": spec.get("activation")})
        row.update({f"set:{k}": v for k, v in (spec.get("set") or {}).items()})
    row.update({f"m:{k}": v for k, v in (state.get("metrics") or {}).items()})
    return row


def compare(study, ids=(), metrics=None, x=None, y=None):
    designs = [study.design(i) for i in ids] if ids else study.designs()
    rows = [design_row(d) for d in designs]
    if metrics:
        wanted = set(metrics)
        rows = [{k: v for k, v in r.items() if not k.startswith("m:") or k[2:] in wanted} for r in rows]
    out = {"designs": rounded(rows, 4)}
    if x and y:
        def value(row, key):
            for k in (key, f"p:{key}", f"m:{key}", f"set:{key}"):
                if k in row and isinstance(row[k], (int, float)) and not isinstance(row[k], bool):
                    return float(row[k])
            return math.nan
        full = [design_row(d) for d in designs]
        points = [(r["id"], value(r, x), value(r, y)) for r in full]
        points = [p for p in points if math.isfinite(p[1]) and math.isfinite(p[2])]
        if len(points) < 2:
            raise ApiError(f"Fewer than two designs have both {x!r} and {y!r}",
                           "Use a parameter name (e.g. t) or a metric name from the table (without m:).")
        folder = study.root / "report" / "figures"
        folder.mkdir(parents=True, exist_ok=True)
        safe = lambda s: "".join(c if c.isalnum() else "_" for c in s)  # noqa: E731
        path = folder / f"compare_{safe(y)}_vs_{safe(x)}.png"
        plots.scatter_compare(path, points, x, y, title=f"{y} against {x}")
        out["plot"] = str(path)
        study.events.emit("file", f"comparison {path.name}", files=[str(path)])
    return out
