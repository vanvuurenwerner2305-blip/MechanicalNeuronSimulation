"""
3D version of the SoftNeuron used in the Sim_Build_v2 notebooks.

A box-shaped cavity is split into three chambers by two clamped membranes (planes x = -+m).
The left/right chambers are held at the input pressures I0, I1; the middle chamber is closed
(P = -K dV). Two obstacles, shaped by the weight tensor W exactly as in make_spline, hang from
the top and bottom walls into the middle chamber and are extruded through the full depth.
The output is the middle-chamber pressure (x 1e3, as in the notebooks).

Run:  python examples/soft_neuron_3d.py [--show]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import membrane_sim as ms  # noqa: E402


class SoftNeuron3D:
    def __init__(self,
                 half_width=2.5,        # cavity x extent: [-half_width, half_width]
                 half_height=3.0,       # cavity y extent
                 depth=4.0,             # cavity z extent (out-of-plane direction of the 2D model)
                 membrane_x=1.5,        # membranes at x = -membrane_x and +membrane_x
                 divisions=(24, 16),    # membrane mesh (along y, along z)
                 thickness=0.1,
                 youngs_modulus=1.0,
                 contact_stiffness=10.0,
                 device="cpu"):
        self.half_width, self.half_height, self.depth = half_width, half_height, depth
        self.membrane_x = membrane_x
        self.divisions = divisions
        self.membrane_kwargs = dict(thickness=thickness, youngs_modulus=youngs_modulus, material="neo_hookean")
        self.contact_stiffness = contact_stiffness
        self.device = device

    # Obstacle outline from the weight tensor (the old make_spline: linear spline through 4 points)
    def obstacle_outline(self, W, x_end=1.4, overlap=0.5):
        W = np.asarray(W, dtype=float)
        h = self.half_height
        return np.array([[-x_end, h], W[0], W[1], [x_end, h], [x_end, h + overlap], [-x_end, h + overlap]])

    def build_simulation(self, W, bulk_stiffness=1e-3):
        env = ms.Environment(contact_stiffness=self.contact_stiffness, device=self.device)
        h, d, m = self.half_height, self.depth, self.membrane_x

        # Membranes: normal +x, so "behind" is the chamber on the -x side
        self.left_membrane = env.add_rectangular_membrane((-m, -h, -d / 2), (0, 2 * h, 0), (0, 0, d),
                                                          self.divisions, color="red", name="left",
                                                          **self.membrane_kwargs)
        self.right_membrane = env.add_rectangular_membrane((m, -h, -d / 2), (0, 2 * h, 0), (0, 0, d),
                                                           self.divisions, color="red", name="right",
                                                           **self.membrane_kwargs)

        outline = self.obstacle_outline(W)
        inside = outline[:4]  # the part of the obstacle below the wall
        obstacle_area = 0.5 * abs(np.sum(inside[:, 0] * np.roll(inside[:, 1], -1) - inside[:, 1] * np.roll(inside[:, 0], -1)))

        self.fvl = env.add_fluid_volume(P0=0.0, name="left", color="blue")
        self.fvm = env.add_fluid_volume(P0=0.0, bulk_stiffness=bulk_stiffness, name="middle", color="green",
                                        initial_volume=(2 * m * 2 * h - 2 * obstacle_area) * d)
        self.fvr = env.add_fluid_volume(P0=0.0, name="right", color="blue")
        self.left_membrane.fluid_volume_contacts((self.fvl, self.fvm))
        self.right_membrane.fluid_volume_contacts((self.fvm, self.fvr))

        # Cavity walls and the two W-shaped obstacles (bottom one mirrored in y)
        w = self.half_width
        env.add_obstacle(*ms.box_mesh((-w, -h, -d / 2), (w, h, d / 2)), inverted=True)
        z0, z1 = -d / 2 - 0.5, d / 2 + 0.5
        env.add_obstacle(*ms.extrude_polygon(outline, z0, z1))
        env.add_obstacle(*ms.extrude_polygon(outline * np.array([1.0, -1.0]), z0, z1))

        self.env = env
        return env

    def forward(self, I0, I1, load_steps=10, verbose=False):
        self.fvl.P0 = float(I0) * 1e-3
        self.fvr.P0 = float(I1) * 1e-3
        self.env.reset()
        result = self.env.solve(load_steps=load_steps, verbose=verbose)
        if not result.converged:
            raise RuntimeError(f"Equilibrium not found for I0={I0}, I1={I1}: {result}")
        return self.fvm.P * 1e3


if __name__ == "__main__":
    W = torch.tensor([[-1.0, 1.0], [0.5, 1.0]])
    model = SoftNeuron3D()
    model.build_simulation(W)

    output = model.forward(I0=10, I1=15, verbose=False)
    res = model.env.last_result
    print(res)
    print(f"Output (middle chamber pressure x 1e3): {output:.4f}")
    print(f"Middle chamber volume change: {model.fvm.delta_volume:.4f} (of {model.fvm.initial_volume:.1f})")

    show = "--show" in sys.argv
    model.env.render_scene(filename="soft_neuron_3d.html", open_browser=show, color_by_displacement=False)
    model.env.render_load_path(filename="soft_neuron_3d_load_path.html", open_browser=show)
    print("Wrote soft_neuron_3d.html and soft_neuron_3d_load_path.html")
