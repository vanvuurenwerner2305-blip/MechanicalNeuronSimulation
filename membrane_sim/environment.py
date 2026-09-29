"""
Environment: collects shells, obstacles and fluid volumes and solves for static equilibrium.
Mirrors the 2D `Enviorment` API (add_membrane / add_obstacle / add_fluid_volume / reset /
render_scene), with `solve` replacing `run_simulation`.
"""
import time

import torch

from . import mesh
from .contact import Obstacle
from .fluid import FluidVolume
from .render import Renderer
from .shell import Shell
from .solver import NewtonSolver, SolveResult


class Environment:
    def __init__(self, contact_stiffness: float = None, contact_offset: float = 0.0, device="cpu"):
        """
        contact_stiffness : penalty stiffness (pressure per unit penetration). Default:
                            1e3 * max(E t) / L^2 with L the size of the shell bounding box.
        contact_offset    : keep shell nodes this far outside obstacles (e.g. half the thickness).
        """
        self.device = device
        self.contact_stiffness = contact_stiffness
        self.contact_offset = contact_offset
        self.renderer = Renderer()

        self.membrane_list = []
        self.obstacle_list = []
        self.fluid_volume_list = []
        self.couplings = []          # extra energy terms (RigidTie, MovingContact, ...)
        self.surface_contacts = []   # explicit ShellContact pairs (e.g. inside a tube)
        self.history = []  # one entry per converged load step
        self.last_result = None
        self.solver = None  # NewtonSolver of the last solve (for post-processing the solved state)

    # -----------------------------
    # Building the model
    # -----------------------------

    def add_membrane(self, vertices, faces, thickness, youngs_modulus, **shell_kwargs) -> Shell:
        """Add a shell from any triangle mesh. See Shell for the keyword arguments."""
        shell = Shell(vertices, faces, thickness, youngs_modulus, device=self.device, **shell_kwargs)
        self.membrane_list.append(shell)
        return shell

    def add_rectangular_membrane(self, corner, edge_u, edge_v, divisions=(20, 20), **kwargs) -> Shell:
        """Membrane on the parallelogram corner + s*edge_u + t*edge_v; normal = edge_u x edge_v."""
        vertices, faces = mesh.rectangle_mesh(corner, edge_u, edge_v, *divisions)
        return self.add_membrane(vertices, faces, **kwargs)

    def add_disk_membrane(self, center, normal, radius, rings=10, **kwargs) -> Shell:
        vertices, faces = mesh.disk_mesh(center, normal, radius, rings)
        return self.add_membrane(vertices, faces, **kwargs)

    def add_obstacle(self, vertices, faces, inverted: bool = False, color: str = "grey") -> Obstacle:
        obstacle = Obstacle(vertices, faces, inverted=inverted, color=color, device=self.device)
        self.obstacle_list.append(obstacle)
        return obstacle

    def add_fluid_volume(self, boundaries=(), P0=0.0, bulk_stiffness=0.0, pressure_law=None,
                         initial_volume=None, gas_volume=None, liquid_volume=None, atmospheric_pressure=0.101325,
                         color="blue", name=None) -> FluidVolume:
        """
        boundaries: iterable of (shell, side) with side = +1 if the shell normal points out of
        this volume. Boundaries can also be attached later with shell.fluid_volume_contacts.
        """
        volume = FluidVolume(P0=P0, bulk_stiffness=bulk_stiffness, pressure_law=pressure_law,
                             initial_volume=initial_volume, gas_volume=gas_volume, liquid_volume=liquid_volume,
                             atmospheric_pressure=atmospheric_pressure, color=color, name=name)
        for shell, side in boundaries:
            volume.add_boundary(shell, side)
        self.fluid_volume_list.append(volume)
        return volume

    # -----------------------------
    # Solving
    # -----------------------------

    def default_contact_stiffness(self) -> float:
        X = torch.cat([s.X for s in self.membrane_list])
        L = (X.max(0).values - X.min(0).values).norm().item()
        return 1e3 * max(s.youngs_modulus * s.thickness for s in self.membrane_list) / L ** 2

    def solve(self, load_steps: int = 10, max_iterations: int = 40, rtol: float = 1e-8,
              atol: float = 1e-12, step_tol: float = 1e-10, max_step: float = None, min_load_increment: float = 1e-3,
              verbose: bool = False, callback=None, warm_start: bool = False) -> SolveResult:
        """
        Find static equilibrium with all fluid pressures applied, ramping them from 0 to full
        over `load_steps` increments (cut automatically when Newton fails). Starts from the
        current shell positions, so call reset() first for a fresh solve.
        callback(load_factor, iteration, residual_norm) is called every Newton iteration; raise
        an exception from it to abort.
        warm_start: continue from the last converged solution and ramp only the change in P0
        (much faster for sweeps, where contact is already established).
        """
        start = time.time()
        k = self.contact_stiffness if self.contact_stiffness is not None else self.default_contact_stiffness()
        solver = NewtonSolver(self.membrane_list, self.fluid_volume_list, self.obstacle_list,
                              contact_stiffness=k, contact_offset=self.contact_offset,
                              rtol=rtol, atol=atol, step_tol=step_tol, max_iterations=max_iterations,
                              max_step=max_step, verbose=verbose, callback=callback,
                              couplings=self.couplings, surface_contacts=self.surface_contacts)
        self.solver = solver

        lam, increment = 0.0, 1.0 / load_steps
        relaxed = False
        result = SolveResult(converged=False, load_factor=0.0)
        for volume in self.fluid_volume_list:
            volume.start_P0 = volume.solved_P0 if warm_start else None
        if not self.history or not warm_start:
            self.history = []
            self._record(0.0)

        while lam < 1.0 - 1e-12:
            target = min(1.0, lam + increment)
            if verbose:
                print(f"load factor {target:.4f}")
            u_saved = solver.get_u().clone()
            ok, iterations = solver.newton(target)
            if ok:
                lam = target
                result.load_factors.append(lam)
                result.iterations.append(iterations)
                self._record(lam)
                if iterations <= 4:
                    increment = min(1.5 * increment, 1.0 / load_steps)
            else:
                solver.set_u(u_saved)
                if not relaxed and increment < 0.25 / load_steps:
                    # Load stepping keeps failing: typically a snap-through, where no nearby
                    # equilibrium exists. Let Newton (an energy descent with damping) run much
                    # longer at the target load so the structure can move to the new state.
                    relaxed = True
                    solver.max_iterations = 5 * max_iterations
                    ok, iterations = solver.newton(target)
                    solver.max_iterations = max_iterations
                    if ok:
                        lam = target
                        result.load_factors.append(lam)
                        result.iterations.append(iterations)
                        self._record(lam)
                        continue
                    solver.set_u(u_saved)
                increment *= 0.5
                if increment < min_load_increment:
                    result.message = f"load increment below {min_load_increment} at load factor {lam:.4g}"
                    break
        else:
            result.converged = True
            result.message = f"solved in {time.time() - start:.2f} s"

        result.load_factor = lam
        for volume in self.fluid_volume_list:
            volume.update(lam)
            if result.converged:
                volume.solved_P0, volume.start_P0 = volume.P0, None
        self.last_result = result
        return result

    def _record(self, load_factor):
        for volume in self.fluid_volume_list:
            volume.update(load_factor)
            volume.pressure_hist.append(volume.P)
            volume.volume_hist.append(volume.delta_volume)
        self.history.append({
            "load_factor": load_factor,
            "shell_coords": [s.x.detach().cpu().clone() for s in self.membrane_list],
            # step 0 is the undeformed state with the full chamber pressures acting on it (e.g. the
            # suction of an under-filled liquid chamber), not the zero pressures of the load ramp
            "pressures": [v.pressure(v.delta_volume) if load_factor == 0.0 else v.P for v in self.fluid_volume_list],
            "delta_volumes": [v.delta_volume for v in self.fluid_volume_list],
        })

    def reset(self):
        self.history = []
        for shell in self.membrane_list:
            shell.reset()
        for volume in self.fluid_volume_list:
            volume.reset()

    # -----------------------------
    # Rendering
    # -----------------------------

    def render_scene(self, filename=None, open_browser=True, color_by_displacement=False):
        return self.renderer.render_scene(self, filename=filename, open_browser=open_browser,
                                          color_by_displacement=color_by_displacement)

    def render_load_path(self, filename="membrane_simulation.html", open_browser=True):
        return self.renderer.render_load_path(self, filename=filename, open_browser=open_browser)
