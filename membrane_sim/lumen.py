"""
Cross-sectional area of a channel (the inside of a tube) as it deforms.

The channel wall is a set of oriented triangles of a solid (outward normals of the solid point
into the channel). Sections are planes normal to the channel axis, placed in the *rest*
configuration: every wall triangle that a plane cuts contributes one segment, whose end points
are fixed points of the material (on the triangle's edges) and follow the deformation. The
section area is the area the deformed loop encloses, projected on the plane normal to the axis,

    A = 1/2 sum_segments (p x q) . a,

with every segment oriented from its triangle's normal, so the segments need not be put in
order and the sum is the signed area of the closed loop(s). Walls pressed through each other
(penalty contact) count negatively, and the area is clipped at 0.
"""
import numpy as np


class ChannelSections:
    def __init__(self, X, faces, axis, stations=None, n_stations: int = 60, margin: float = 0.02):
        """
        X        : (N, 3) rest coordinates of the solid's nodes.
        faces    : (F, 3) the channel wall triangles (normals pointing into the channel).
        axis     : direction of the channel (input -> output).
        stations : positions along the axis (measured from the origin of X) of the sections;
                   default n_stations evenly over the wall's extent, leaving `margin` at each end.
        """
        X = np.asarray(X, float)
        F = np.asarray(faces, np.int64)
        a = np.asarray(axis, float)
        self.axis = a / np.linalg.norm(a)
        s = X @ self.axis
        lo, hi = s[F].min(), s[F].max()
        if stations is None:
            pad = margin * (hi - lo)
            stations = np.linspace(lo + pad, hi - pad, n_stations)
        self.stations = np.asarray(stations, float)
        self.start = lo

        # For every section: segment end points as (node i, node j, weight of j) on triangle edges
        self.segments = []
        tri_s = s[F]
        e = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])
        for z in self.stations:
            # a vertex on the plane counts as above it, so every cut triangle has exactly two cut edges
            above = tri_s >= z
            cut = above.any(axis=1) & ~above.all(axis=1)
            ends = []
            for f in np.nonzero(cut)[0]:
                pts = []
                for i, j in ((0, 1), (1, 2), (2, 0)):
                    if above[f, i] != above[f, j]:
                        si, sj = tri_s[f, i], tri_s[f, j]
                        pts.append((F[f, i], F[f, j], (z - si) / (sj - si)))
                # orient the segment so that (q - p) = axis x normal direction (consistent loops)
                p = _point(X, pts[0])
                q = _point(X, pts[1])
                if np.dot(np.cross(self.axis, e[f]), q - p) < 0:
                    pts = pts[::-1]
                ends.append(pts)
            if ends:
                arr = np.array([[[i, j, w] for i, j, w in seg] for seg in ends])
                self.segments.append((arr[:, :, 0].astype(np.int64), arr[:, :, 1].astype(np.int64), arr[:, :, 2]))
            else:
                self.segments.append(None)
        raw = self._raw_areas(X)
        self._sign = 1.0 if np.median(raw) >= 0 else -1.0   # rest loops count positive
        self.rest_areas = self._sign * raw

    def _raw_areas(self, x) -> np.ndarray:
        x = np.asarray(x, float)
        out = np.zeros(len(self.stations))
        for k, seg in enumerate(self.segments):
            if seg is None:
                continue
            i, j, w = seg
            pts = (1.0 - w)[..., None] * x[i] + w[..., None] * x[j]        # (segments, 2, 3)
            out[k] = 0.5 * np.einsum("sk,k->s", np.cross(pts[:, 0], pts[:, 1]), self.axis).sum()
        return out

    def signed(self, x) -> np.ndarray:
        """Signed section areas (negative where the walls are pressed through each other)."""
        return self._sign * self._raw_areas(x)

    def geometry(self, x, across) -> dict:
        """Per station: area A (clipped at 0), wetted perimeter P, height h (extent along `across`, e.g. the
        squeeze direction) and width w (extent along axis x across) of the deformed section."""
        x = np.asarray(x, float)
        across = np.asarray(across, float) - np.dot(across, self.axis) * self.axis
        across /= np.linalg.norm(across)
        lateral = np.cross(self.axis, across)
        n = len(self.stations)
        out = {"A": np.clip(self.signed(x), 0.0, None), "P": np.zeros(n), "h": np.zeros(n), "w": np.zeros(n)}
        for k, seg in enumerate(self.segments):
            if seg is None:
                continue
            i, j, wt = seg
            pts = (1.0 - wt)[..., None] * x[i] + wt[..., None] * x[j]
            out["P"][k] = np.linalg.norm(pts[:, 1] - pts[:, 0], axis=1).sum()
            flat = pts.reshape(-1, 3)
            out["h"][k] = np.ptp(flat @ across)
            out["w"][k] = np.ptp(flat @ lateral)
        return out

    def profile(self, x) -> np.ndarray:
        """Section areas (>= 0) along the channel at node coordinates x."""
        return np.clip(self.signed(x), 0.0, None)

    def minimum(self, x) -> float:
        """Smallest section area: the constriction that sets the flow."""
        return float(self.profile(x).min())


def _point(X, spec):
    i, j, w = spec
    return (1.0 - w) * X[i] + w * X[j]
