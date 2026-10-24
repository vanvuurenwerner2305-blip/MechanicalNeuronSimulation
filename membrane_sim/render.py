"""Plotly 3D rendering of shells and obstacles, for a single state or along the load path."""
import os
import webbrowser

import numpy as np
import plotly.graph_objects as go


class Renderer:
    def __init__(self, title: str = "Membrane Simulation"):
        self.title = title

    # -----------------------------
    # Traces
    # -----------------------------

    @staticmethod
    def shell_trace(shell, coords, color_by_displacement=False, name=None):
        coords = np.asarray(coords)
        f = shell.faces.cpu().numpy()
        kwargs = dict(color=shell.color)
        if color_by_displacement:
            u = np.linalg.norm(coords - shell.X.cpu().numpy(), axis=1)
            kwargs = dict(intensity=u, colorscale="Viridis", colorbar=dict(title="|u|"))
        return go.Mesh3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                         i=f[:, 0], j=f[:, 1], k=f[:, 2],
                         flatshading=True, name=name or shell.name or "Membrane", **kwargs)

    @staticmethod
    def obstacle_traces(obstacles):
        traces = []
        for obs in obstacles:
            v = obs.vertices.cpu().numpy()
            f = obs.faces.cpu().numpy()
            if obs.inverted:  # containers: draw the feature edges only so the inside stays visible
                x, y, z = [], [], []
                for a, b in _feature_edges(v, f):
                    for coord, lst in zip(np.stack([v[a], v[b]]).T, (x, y, z)):
                        lst.extend([coord[0], coord[1], None])
                traces.append(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="black", width=2),
                                           name="Container", showlegend=False, hoverinfo="skip"))
            else:
                traces.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                        color=obs.color, opacity=0.35, flatshading=True,
                                        name="Obstacle", hoverinfo="skip"))
        return traces

    def _layout(self, title):
        return go.Layout(title=title, scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=40, b=0))

    @staticmethod
    def _pressure_text(pressures, volumes):
        return "  |  ".join(f"{v.name or f'V{i}'}: P = {p:.4g}" for i, (v, p) in enumerate(zip(volumes, pressures)))

    # -----------------------------
    # Output
    # -----------------------------

    def render_scene(self, env, filename=None, open_browser=True, color_by_displacement=False):
        data = self.obstacle_traces(env.obstacle_list)
        data += [self.shell_trace(s, s.x.detach().cpu().numpy(), color_by_displacement) for s in env.membrane_list]
        title = f"{self.title}<br><sub>{self._pressure_text([v.P for v in env.fluid_volume_list], env.fluid_volume_list)}</sub>"
        fig = go.Figure(data=data, layout=self._layout(title))
        return self._show(fig, filename, open_browser)

    def render_load_path(self, env, filename="membrane_simulation.html", open_browser=True):
        """Animated figure with one frame per converged load step (slider = load factor)."""
        history = env.history
        if not history:
            raise RuntimeError("Nothing to render: call solve() first.")
        static = self.obstacle_traces(env.obstacle_list)
        n_static = len(static)

        def shells_at(step):
            return [self.shell_trace(s, c.numpy()) for s, c in zip(env.membrane_list, step["shell_coords"])]

        frames = [go.Frame(data=shells_at(step), name=str(k),
                           traces=list(range(n_static, n_static + len(env.membrane_list))),
                           layout=go.Layout(title=f"{self.title}  (load factor {step['load_factor']:.3f})<br><sub>"
                                                  f"{self._pressure_text(step['pressures'], env.fluid_volume_list)}</sub>"))
                  for k, step in enumerate(history)]

        slider = [dict(active=len(frames) - 1, currentvalue={"prefix": "Load factor: "}, pad={"t": 30},
                       steps=[dict(method="animate", label=f"{step['load_factor']:.2f}",
                                   args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                        transition=dict(duration=0))])
                              for k, step in enumerate(history)])]
        buttons = [dict(type="buttons", direction="right", x=0, y=0, xanchor="left", yanchor="top",
                        buttons=[dict(label="Play", method="animate",
                                      args=[None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True}]),
                                 dict(label="Pause", method="animate",
                                      args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])])]

        layout = self._layout(frames[-1].layout.title.text)
        layout.update(sliders=slider, updatemenus=buttons)
        fig = go.Figure(data=static + shells_at(history[-1]), frames=frames, layout=layout)
        return self._show(fig, filename, open_browser)

    @staticmethod
    def _show(fig, filename, open_browser):
        if filename is None:
            if open_browser:
                fig.show()
            return fig
        fig.write_html(filename)
        if open_browser:
            webbrowser.open("file://" + os.path.realpath(filename))
        return fig


def _feature_edges(vertices, faces, angle_tol=1e-6):
    """Edges between non-coplanar faces (drops triangulation diagonals of flat walls)."""
    n = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    owners = {}
    for f, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            owners.setdefault((min(a, b), max(a, b)), []).append(f)
    return [e for e, fs in owners.items() if len(fs) != 2 or abs(np.dot(n[fs[0]], n[fs[1]])) < 1 - angle_tol]
