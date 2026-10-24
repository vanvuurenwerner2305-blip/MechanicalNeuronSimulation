"""
CAD from a design file: every body's shape is described with primitives, sketches (extruded, revolved),
pipes, booleans and transforms, all in millimetres and in terms of the design's parameters; the result is a
STEP file with one named solid per body (what the simulator imports, like a Fusion 360 export).

    shape kinds (exactly one per shape):
      box:       {min: [x,y,z], size: [dx,dy,dz]}   or {center: [x,y,z], size: [...]}
      cylinder:  {base: [x,y,z], axis: [ax,ay,az], radius: r}          (axis vector = height and direction)
      cone:      {base: [...], axis: [...], radius1: r1, radius2: r2}
      sphere:    {center: [...], radius: r}
      torus:     {center: [...], radius: R, tube_radius: r, axis: [0,0,1]}
      extrude:   {sketch: SKETCH, distance: d, symmetric: false}       (along the sketch plane's normal)
      revolve:   {sketch: SKETCH, axis_point: [...], axis: [...], angle: 360}   (degrees)
      pipe:      {path: [[x,y,z], ...], radius: r, inner_radius: 0, smooth: false}
      union:     [SHAPE, SHAPE, ...]
      cut:       {from: SHAPE, remove: [SHAPE, ...]}
      intersect: [SHAPE, SHAPE, ...]
      ref:       BodyName                  (a copy of another body's shape)
      cavity:    {inside: SHAPE, minus: [BodyName, ...]}   (the region inside SHAPE not taken by those bodies)
      import:    BodyName                  (a body of the design's base STEP file, `base_step`)
    optional on any shape (applied in this order): scale: s | [sx,sy,sz] (about scale_origin, default origin),
      rotate: {axis: [...], angle: deg, origin: [...]} (or a list of them), mirror: {normal: [...], origin: [...]},
      translate: [dx,dy,dz], fillet: r (rounds every edge)

    SKETCH = {plane: xy | yz | zx | xz | {origin: [...], normal: [...], u: [...]}, offset: d,
              and one outline: circle: {center: [u,v], radius: r}
                             | rectangle: {min: [u,v], size: [du,dv]} or {center: [u,v], size: [...]}, corner_radius
                             | polygon: [[u,v], ...]
                             | path: [[u,v], [u,v], {arc: {through: [u,v], to: [u,v]}},
                                      {arc: {center: [u,v], to: [u,v]}}, {spline: [[u,v], ..., [u,v]]}, ...]
              holes: [outline, ...] (optional: each {circle: ...} / {rectangle: ...} / {polygon: ...} / {path: ...})}

Every number may be an expression of the design's parameters (strings such as "2*t + gap").
"""
import math
import re
from pathlib import Path

import gmsh
import numpy as np

from .util import ApiError, as_number, as_vector

SHAPE_KINDS = ("box", "cylinder", "cone", "sphere", "torus", "extrude", "revolve", "pipe", "union", "cut",
               "intersect", "ref", "cavity", "import")
MODIFIERS = ("scale", "scale_origin", "rotate", "mirror", "translate", "fillet")
PLANES = {"xy": ([1, 0, 0], [0, 1, 0]), "yz": ([0, 1, 0], [0, 0, 1]), "zx": ([0, 0, 1], [1, 0, 0]),
          "xz": ([1, 0, 0], [0, 0, 1])}


class CadBuilder:
    """Builds the bodies of a design file into gmsh's OpenCASCADE model and writes them as a STEP file."""

    def __init__(self, bodies, names, base_step=None, folder="."):
        self.specs = {}
        for k, body in enumerate(bodies):
            name = str(body.get("name") or "").strip()
            if not name:
                raise ApiError(f"Body number {k + 1} has no name.")
            if name in self.specs:
                raise ApiError(f"Two bodies are called {name!r}.", "Every body needs its own name.")
            self.specs[name] = body
        self.names = names
        self.base_step = base_step
        self.folder = Path(folder)
        self.built = {}        # body name -> volume tag
        self._building = []
        self.imported = {}     # base STEP body name -> volume tag

    # -----------------------------

    def build(self):
        """Every body in the gmsh model; returns {name: volume tag} in the design's order."""
        from app.cad import CadModel
        if not gmsh.isInitialized():
            gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        if self._uses_import():
            if not self.base_step:
                raise ApiError("A body uses 'import' but the design has no base_step.",
                               "Set base_step: <file.step> in the design file (next to it in the design folder).")
            path = self.folder / self.base_step
            if not path.exists():
                raise ApiError(f"Base STEP file not found: {path}")
            cad = CadModel()
            cad.load_step(str(path))
            self.imported = {b.name: b.tag for b in cad.bodies}
        else:
            gmsh.clear()
            gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
            gmsh.model.add("design")
        for name in self.specs:
            self._body(name)
        occ = gmsh.model.occ
        occ.synchronize()
        keep = set(self.built.values())
        extra = [(3, t) for _, t in gmsh.model.getEntities(3) if t not in keep]
        if extra:
            occ.remove(extra, recursive=True)
        occ.synchronize()
        # sketches, pipe sections and other construction entities left over would be written as well
        used = {(3, t) for t in keep}
        level = [(3, t) for t in keep]
        for _ in range(3):
            level = [(d, abs(t)) for d, t in gmsh.model.getBoundary(level, combined=False, oriented=False)]
            used.update(level)
        for dim in (2, 1, 0):
            loose = [e for e in gmsh.model.getEntities(dim) if e not in used]
            if loose:
                occ.remove(loose)
        occ.synchronize()
        return dict(self.built)

    def _uses_import(self):
        found = []

        def walk(x):
            if isinstance(x, dict):
                if "import" in x:
                    found.append(True)
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        for spec in self.specs.values():
            walk(spec.get("shape"))
        return bool(found)

    def _body(self, name):
        if name in self.built:
            return self.built[name]
        if name not in self.specs:
            raise ApiError(f"Unknown body {name!r}", f"Bodies: {', '.join(self.specs)}")
        if name in self._building:
            raise ApiError(f"Bodies refer to each other in a circle: {' -> '.join(self._building + [name])}")
        spec = self.specs[name]
        if "shape" not in spec:
            raise ApiError(f"Body {name!r} has no shape.", "Give it a shape (box, cylinder, extrude, ..., or "
                                                           "import: <name> to take it from base_step).")
        self._building.append(name)
        try:
            tags = self._shape(spec["shape"], name)
        except ApiError as exc:
            raise ApiError(f"{name}: {exc}", exc.hint)
        finally:
            self._building.pop()
        tags = self._one_solid(tags, name)
        self.built[name] = tags
        return tags

    def _one_solid(self, tags, name):
        occ = gmsh.model.occ
        if len(tags) > 1:
            out, _ = occ.fuse(tags[:1], tags[1:])
            occ.synchronize()
            tags = [t for t in out if t[0] == 3]
        if len(tags) != 1:
            raise ApiError(f"Body {name!r} is {len(tags)} separate solids, not one.",
                           "A body must be one connected solid: make the pieces overlap or touch, or split them "
                           "into separate bodies.")
        return tags[0][1]

    # -----------------------------
    # Shapes
    # -----------------------------

    def _shape(self, spec, where):
        """[(3, tag)] of a shape description."""
        if isinstance(spec, list):
            raise ApiError("A shape must be a dictionary with one kind (box, cylinder, ...), not a list.")
        if not isinstance(spec, dict):
            raise ApiError(f"Not a shape: {spec!r}")
        kinds = [k for k in spec if k in SHAPE_KINDS]
        unknown = [k for k in spec if k not in SHAPE_KINDS and k not in MODIFIERS]
        if unknown:
            raise ApiError(f"Unknown key(s) in a shape: {', '.join(unknown)}",
                           f"Shape kinds: {', '.join(SHAPE_KINDS)}; modifiers: {', '.join(MODIFIERS)}.")
        if len(kinds) != 1:
            raise ApiError(f"A shape needs exactly one kind, found {kinds or 'none'}",
                           f"Shape kinds: {', '.join(SHAPE_KINDS)}.")
        kind = kinds[0]
        tags = getattr(self, "_" + kind)(spec[kind])
        gmsh.model.occ.synchronize()
        return self._modify(tags, spec)

    def n(self, x, what):
        return as_number(x, self.names, what)

    def v(self, x, what, n=3):
        return as_vector(x, self.names, n, what)

    def _box(self, p):
        size = self.v(p.get("size"), "box size")
        if "center" in p:
            c = self.v(p["center"], "box center")
            lo = [c[i] - size[i] / 2 for i in range(3)]
        else:
            lo = self.v(p.get("min"), "box min")
        if min(abs(s) for s in size) <= 0:
            raise ApiError("box size must be non-zero in x, y and z")
        return [(3, gmsh.model.occ.addBox(*lo, *size))]

    def _cylinder(self, p):
        r = self.n(p.get("radius"), "cylinder radius")
        return [(3, gmsh.model.occ.addCylinder(*self.v(p.get("base"), "cylinder base"),
                                               *self.v(p.get("axis"), "cylinder axis"), r))]

    def _cone(self, p):
        return [(3, gmsh.model.occ.addCone(*self.v(p.get("base"), "cone base"), *self.v(p.get("axis"), "cone axis"),
                                           self.n(p.get("radius1"), "cone radius1"),
                                           self.n(p.get("radius2"), "cone radius2")))]

    def _sphere(self, p):
        return [(3, gmsh.model.occ.addSphere(*self.v(p.get("center"), "sphere center"),
                                             self.n(p.get("radius"), "sphere radius")))]

    def _torus(self, p):
        return [(3, gmsh.model.occ.addTorus(*self.v(p.get("center"), "torus center"),
                                            self.n(p.get("radius"), "torus radius"),
                                            self.n(p.get("tube_radius"), "torus tube_radius"),
                                            zAxis=self.v(p.get("axis", [0, 0, 1]), "torus axis")))]

    def _extrude(self, p):
        surface, normal = self._sketch(p.get("sketch"))
        d = self.n(p.get("distance"), "extrude distance")
        occ = gmsh.model.occ
        if p.get("symmetric"):
            occ.translate([(2, surface)], *(-0.5 * d * normal))
        out = occ.extrude([(2, surface)], *(d * normal))
        return [t for t in out if t[0] == 3]

    def _revolve(self, p):
        surface, _ = self._sketch(p.get("sketch"))
        angle = math.radians(self.n(p.get("angle", 360), "revolve angle"))
        out = gmsh.model.occ.revolve([(2, surface)], *self.v(p.get("axis_point", [0, 0, 0]), "revolve axis_point"),
                                     *self.v(p.get("axis"), "revolve axis"), angle)
        return [t for t in out if t[0] == 3]

    def _pipe(self, p):
        occ = gmsh.model.occ
        path = p.get("path")
        if not isinstance(path, list) or len(path) < 2:
            raise ApiError("pipe path needs at least two points [[x,y,z], ...]")
        pts = np.array([self.v(q, "pipe path point") for q in path])
        tags = [occ.addPoint(*q) for q in pts]
        if p.get("smooth") and len(pts) > 2:
            curves = [occ.addSpline(tags)]
        else:
            curves = [occ.addLine(tags[i], tags[i + 1]) for i in range(len(tags) - 1)]
        wire = occ.addWire(curves)
        direction = pts[1] - pts[0]
        r = self.n(p.get("radius"), "pipe radius")
        ri = self.n(p.get("inner_radius", 0), "pipe inner_radius")
        disk = occ.addDisk(*pts[0], r, r, zAxis=list(direction))
        if ri > 0:
            hole = occ.addDisk(*pts[0], ri, ri, zAxis=list(direction))
            disk = occ.cut([(2, disk)], [(2, hole)])[0][0][1]
        out = occ.addPipe([(2, disk)], wire)
        return [t for t in out if t[0] == 3]

    def _union(self, items):
        shapes = self._list(items, "union")
        out, _ = gmsh.model.occ.fuse(shapes[0], [t for s in shapes[1:] for t in s])
        return [t for t in out if t[0] == 3]

    def _intersect(self, items):
        shapes = self._list(items, "intersect")
        result = shapes[0]
        for s in shapes[1:]:
            result, _ = gmsh.model.occ.intersect(result, s)
            result = [t for t in result if t[0] == 3]
        if not result:
            raise ApiError("intersect: the shapes do not overlap (empty result)")
        return result

    def _cut(self, p):
        if not isinstance(p, dict) or "from" not in p or "remove" not in p:
            raise ApiError("cut needs {from: SHAPE, remove: [SHAPE, ...]}")
        base = self._shape(p["from"], "cut")
        tools = [t for s in self._list(p["remove"], "cut remove", minimum=1) for t in s]
        out, _ = gmsh.model.occ.cut(base, tools)
        out = [t for t in out if t[0] == 3]
        if not out:
            raise ApiError("cut removed everything (empty result)")
        return out

    def _ref(self, name):
        tag = self._body(str(name))
        return [t for t in gmsh.model.occ.copy([(3, tag)]) if t[0] == 3]

    def _cavity(self, p):
        if not isinstance(p, dict) or "inside" not in p:
            raise ApiError("cavity needs {inside: SHAPE, minus: [BodyName, ...]}")
        region = self._shape(p["inside"], "cavity")
        minus = p.get("minus") or []
        if isinstance(minus, str):
            minus = [minus]
        tools = [t for name in minus for t in self._ref(name)]
        if not tools:
            return region
        out, _ = gmsh.model.occ.cut(region, tools)
        out = [t for t in out if t[0] == 3]
        if not out:
            raise ApiError("cavity: nothing is left of the region")
        return out

    def _import(self, name):
        name = str(name)
        if name not in self.imported:
            raise ApiError(f"import: {name!r} is not a body of {self.base_step}",
                           f"Its bodies: {', '.join(self.imported)}")
        return [t for t in gmsh.model.occ.copy([(3, self.imported[name])]) if t[0] == 3]

    def _list(self, items, what, minimum=2):
        if not isinstance(items, list) or len(items) < minimum:
            raise ApiError(f"{what} needs a list of at least {minimum} shapes")
        return [self._shape(s, what) for s in items]

    # -----------------------------
    # Sketches
    # -----------------------------

    def _plane(self, sketch):
        plane = sketch.get("plane", "xy")
        offset = self.n(sketch.get("offset", 0.0), "sketch offset")
        if isinstance(plane, str):
            if plane not in PLANES:
                raise ApiError(f"Unknown sketch plane {plane!r}", "Use xy, yz, zx, xz or {origin, normal, u}.")
            e1, e2 = (np.array(a, float) for a in PLANES[plane])
            normal = np.cross(e1, e2)
            origin = offset * normal
        else:
            origin = np.array(self.v(plane.get("origin", [0, 0, 0]), "plane origin"))
            normal = np.array(self.v(plane.get("normal"), "plane normal"))
            normal /= np.linalg.norm(normal)
            u = np.array(self.v(plane.get("u"), "plane u")) if "u" in plane else None
            if u is None:
                u = np.cross(normal, [0, 0, 1] if abs(normal[2]) < 0.9 else [1, 0, 0])
            e1 = u - (u @ normal) * normal
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(normal, e1)
            origin = origin + offset * normal
        return origin, e1, e2, normal

    def _sketch(self, sketch):
        if not isinstance(sketch, dict):
            raise ApiError("sketch must be {plane: ..., circle|rectangle|polygon|path: ...}")
        frame = self._plane(sketch)
        outer = self._loop(sketch, frame)
        holes = [self._loop(h, frame) for h in sketch.get("holes", [])]
        surface = gmsh.model.occ.addPlaneSurface([outer] + holes)
        return surface, frame[3]

    def _loop(self, outline, frame):
        origin, e1, e2, _ = frame
        occ = gmsh.model.occ
        at = lambda uv: origin + uv[0] * e1 + uv[1] * e2  # noqa: E731
        kinds = [k for k in ("circle", "rectangle", "polygon", "path") if k in outline]
        if len(kinds) != 1:
            raise ApiError("a sketch outline needs exactly one of circle, rectangle, polygon, path")
        kind = kinds[0]
        p = outline[kind]
        if kind == "circle":
            c, r = self.v(p.get("center", [0, 0]), "circle center", 2), self.n(p.get("radius"), "circle radius")
            circle = occ.addCircle(*at(c), r, zAxis=list(np.cross(e1, e2)), xAxis=list(e1))
            return occ.addCurveLoop([circle])
        if kind == "rectangle":
            size = self.v(p.get("size"), "rectangle size", 2)
            if "center" in p:
                c = self.v(p["center"], "rectangle center", 2)
                lo = [c[0] - size[0] / 2, c[1] - size[1] / 2]
            else:
                lo = self.v(p.get("min"), "rectangle min", 2)
            r = self.n(outline.get("corner_radius", p.get("corner_radius", 0)), "corner_radius")
            corners = [lo, [lo[0] + size[0], lo[1]], [lo[0] + size[0], lo[1] + size[1]], [lo[0], lo[1] + size[1]]]
            if r <= 0:
                return self._polyline([at(q) for q in corners])
            return self._rounded(corners, r, at)
        if kind == "polygon":
            if not isinstance(p, list) or len(p) < 3:
                raise ApiError("polygon needs at least three points [[u,v], ...]")
            return self._polyline([at(self.v(q, "polygon point", 2)) for q in p])
        return self._path(p, at)

    def _polyline(self, points):
        occ = gmsh.model.occ
        tags = [occ.addPoint(*q) for q in points]
        lines = [occ.addLine(tags[i], tags[(i + 1) % len(tags)]) for i in range(len(tags))]
        return occ.addCurveLoop(lines)

    def _rounded(self, corners, r, at):
        occ = gmsh.model.occ
        curves = []
        c = [np.array(q, float) for q in corners]
        pts = []
        for i in range(4):
            prev, cur, nxt = c[i - 1], c[i], c[(i + 1) % 4]
            a = cur + r * (prev - cur) / np.linalg.norm(prev - cur)
            b = cur + r * (nxt - cur) / np.linalg.norm(nxt - cur)
            centre = a + (b - cur)
            pts.append((occ.addPoint(*at(a)), occ.addPoint(*at(centre)), occ.addPoint(*at(b))))
        for i in range(4):
            a, centre, b = pts[i]
            curves.append(occ.addCircleArc(a, centre, b))
            curves.append(occ.addLine(b, pts[(i + 1) % 4][0]))
        return occ.addCurveLoop(curves)

    def _path(self, items, at):
        occ = gmsh.model.occ
        if not isinstance(items, list) or len(items) < 2 or not isinstance(items[0], list):
            raise ApiError("path must start with a point [u,v] and have at least two items")
        start = self.v(items[0], "path point", 2)
        first = occ.addPoint(*at(start))
        scale = max(1.0, *[abs(c) for c in start])

        def point(uv):  # back at the start: close the outline on the first point
            if abs(uv[0] - start[0]) + abs(uv[1] - start[1]) <= 1e-9 * scale:
                return first
            return occ.addPoint(*at(uv))

        current, curves = first, []
        for item in items[1:]:
            if isinstance(item, list):
                nxt = point(self.v(item, "path point", 2))
                curves.append(occ.addLine(current, nxt))
            elif isinstance(item, dict) and "arc" in item:
                arc = item["arc"]
                nxt = point(self.v(arc.get("to"), "arc to", 2))
                if "through" in arc:
                    mid = occ.addPoint(*at(self.v(arc["through"], "arc through", 2)))
                    curves.append(occ.addCircleArc(current, mid, nxt, center=False))
                elif "center" in arc:
                    mid = occ.addPoint(*at(self.v(arc["center"], "arc center", 2)))
                    curves.append(occ.addCircleArc(current, mid, nxt))
                else:
                    raise ApiError("arc needs {to: [u,v]} and either through: [u,v] or center: [u,v]")
            elif isinstance(item, dict) and "spline" in item:
                pts = [current] + [point(self.v(q, "spline point", 2)) for q in item["spline"]]
                nxt = pts[-1]
                curves.append(occ.addSpline(pts))
            else:
                raise ApiError(f"Unknown path item {item!r}", "Use [u,v], {arc: {...}} or {spline: [...]}.")
            current = nxt
        if current != first:
            curves.append(occ.addLine(current, first))
        return occ.addCurveLoop(curves)

    # -----------------------------
    # Modifiers
    # -----------------------------

    def _modify(self, tags, spec):
        occ = gmsh.model.occ
        if "scale" in spec:
            s = spec["scale"]
            f = self.v(s, "scale") if isinstance(s, list) else [self.n(s, "scale")] * 3
            o = self.v(spec.get("scale_origin", [0, 0, 0]), "scale_origin")
            occ.dilate(tags, *o, *f)
        rotations = spec.get("rotate")
        for r in ([rotations] if isinstance(rotations, dict) else rotations or []):
            occ.rotate(tags, *self.v(r.get("origin", [0, 0, 0]), "rotate origin"), *self.v(r.get("axis"), "rotate axis"),
                       math.radians(self.n(r.get("angle"), "rotate angle")))
        if "mirror" in spec:
            m = spec["mirror"]
            normal = np.array(self.v(m.get("normal"), "mirror normal"))
            origin = np.array(self.v(m.get("origin", [0, 0, 0]), "mirror origin"))
            occ.mirror(tags, *normal, -float(normal @ origin))
        if "translate" in spec:
            occ.translate(tags, *self.v(spec["translate"], "translate"))
        if "fillet" in spec:
            r = self.n(spec["fillet"], "fillet")
            occ.synchronize()
            surfaces = gmsh.model.getBoundary(tags, combined=False, oriented=False)
            edges = sorted({abs(e[1]) for e in gmsh.model.getBoundary(surfaces, combined=False, oriented=False)})
            try:
                tags = [t for t in occ.fillet([t for _, t in tags], edges, [r]) if t[0] == 3]
            except Exception as exc:
                raise ApiError(f"fillet of {r} mm failed ({exc})", "Use a smaller radius than the thinnest wall.")
        occ.synchronize()
        return tags


# -----------------------------
# STEP output with body names
# -----------------------------

def write_step(path, built, model_name="Design"):
    """Write the built bodies (name -> volume tag) to a STEP file whose products carry the body names (as a CAD
    export would), then check the names read back."""
    from app.cad import CadModel
    path = Path(path)
    gmsh.model.occ.synchronize()
    gmsh.write(str(path))
    written = [tag for _, tag in gmsh.model.getEntities(3)]
    tag_to_name = {tag: name for name, tag in built.items()}
    text = path.read_text(errors="ignore")
    text = re.sub(r"Open CASCADE STEP translator [\d.]+ \d+\.(\d+)",
                  lambda m: tag_to_name.get(written[int(m.group(1)) - 1], m.group(0)), text)
    text = re.sub(r"'Open CASCADE STEP translator [\d.]+ \d+'", f"'{model_name}'", text)
    path.write_text(text)
    cad = CadModel()
    bodies = cad.load_step(str(path))
    names = [b.name for b in bodies]
    if sorted(names) != sorted(built):
        raise ApiError(f"The STEP file's body names came back as {names}, expected {list(built)}")
    return cad


# -----------------------------
# Geometry checks
# -----------------------------

def geometry_report(cad, roles=None, overlap_tolerance=1e-4):
    """Per body: volume, bounding box, size, thickness estimate; between bodies: overlaps (shared volume - an
    error for touching parts such as a membrane and its chamber) and which bodies touch."""
    occ = gmsh.model.occ
    cad._ensure_active()
    roles = roles or {}
    bodies = []
    for b in cad.bodies:
        size = b.size
        bodies.append({"name": b.name, "role": roles.get(b.name), "volume_mm3": b.volume, "area_mm2": b.area,
                       "bbox_min": b.bbox[:3], "bbox_max": b.bbox[3:], "size": size,
                       "thin_thickness_estimate": b.thickness_estimate if np.sort(size)[0] < 0.2 * np.sort(size)[1]
                       else None})
    overlaps, touching = [], []
    tol = 1e-6 * max(max(b.diagonal for b in cad.bodies), 1.0)
    for i, a in enumerate(cad.bodies):
        for b in cad.bodies[i + 1:]:
            lo = np.maximum(a.bbox[:3], b.bbox[:3])
            hi = np.minimum(a.bbox[3:], b.bbox[3:])
            if np.any(lo > hi + 10 * tol):
                continue
            distance = occ.getDistance(3, a.tag, 3, b.tag)[0]
            if distance > 10 * tol:
                continue
            shared = 0.0
            if np.all(hi - lo > 10 * tol):
                copies_a, copies_b = occ.copy([(3, a.tag)]), occ.copy([(3, b.tag)])
                try:
                    out, _ = occ.intersect(copies_a, copies_b)
                    occ.synchronize()
                    shared = sum(occ.getMass(3, t) for d, t in out if d == 3)
                    if out:
                        occ.remove(out, recursive=True)
                except Exception:
                    shared = 0.0
                occ.synchronize()
                leftover = [t for t in copies_a + copies_b if t in gmsh.model.getEntities(3)]
                if leftover:
                    occ.remove(leftover, recursive=True)
                    occ.synchronize()
            if shared > overlap_tolerance * min(a.volume, b.volume):
                overlaps.append({"bodies": [a.name, b.name], "shared_volume_mm3": shared,
                                 "share_of_smaller": shared / min(a.volume, b.volume)})
            else:
                touching.append([a.name, b.name])
    return {"bodies": bodies, "overlaps": overlaps, "touching": touching}
