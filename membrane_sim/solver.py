"""
Static Newton-Raphson solver.

The equilibrium state minimises the total potential
    Pi(u) = sum W_shell(u) + sum E_contact(u) - lambda * sum_volumes int_0^dV P(s) ds
over the free dofs u (node coordinates and mid-edge rotations). The residual is R = dPi/du
and the tangent
    K = K_elements + K_contact - lambda * sum_v [ P_v d2V_v/du2 + P'_v g_v g_v^T ].
The g g^T terms are dense (they couple every node of a chamber) so they are kept out of
the sparse matrix and handled with the Sherman-Morrison-Woodbury identity on top of a
sparse LU factorisation.

Robustness: pressures are ramped with a load factor lambda in [0, 1] with automatic
step cutting; every Newton step is limited in size and followed by a backtracking line
search (energy Armijo or residual decrease); rejected steps and singular tangents (e.g. a
flat membrane without bending or pre-tension has no out-of-plane stiffness) fall back to
Levenberg-Marquardt damping K + mu I.
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .contact import ObstacleField
from .fluid import cone_volume_grad, cone_volume_hess

DTYPE = torch.float64


@dataclass
class SolveResult:
    converged: bool
    load_factor: float
    load_factors: list = field(default_factory=list)
    iterations: list = field(default_factory=list)
    message: str = ""

    def __repr__(self):
        return (f"SolveResult(converged={self.converged}, load_factor={self.load_factor:.4g}, "
                f"load_steps={len(self.load_factors)}, newton_iterations={sum(self.iterations)}, "
                f"message='{self.message}')")


class NewtonSolver:
    def __init__(self, shells, fluid_volumes, obstacles,
                 contact_stiffness: float, contact_offset: float = 0.0,
                 rtol: float = 1e-8, atol: float = 1e-12, step_tol: float = 1e-10,
                 max_iterations: int = 40, max_step: float = None, verbose: bool = False, callback=None):
        self.shells = list(shells)
        self.fluid_volumes = list(fluid_volumes)
        self.field = ObstacleField(obstacles) if obstacles else None
        self.contact_stiffness = contact_stiffness
        self.contact_offset = contact_offset
        self.rtol, self.atol, self.step_tol = rtol, atol, step_tol
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.callback = callback  # called as callback(load_factor, iteration, residual_norm)
        self.device = self.shells[0].device

        # Global dof layout: one block per shell, [node coords..., edge rotations...]
        sizes = [s.n_dof for s in self.shells]
        self.dof_offset = dict(zip(map(id, self.shells), np.cumsum([0] + sizes[:-1]).tolist()))
        self.n_dof = int(sum(sizes))
        fixed = torch.cat([s.fixed_dofs for s in self.shells]).cpu().numpy()
        self.free_dofs = np.nonzero(~fixed)[0]
        self.n_free = len(self.free_dofs)
        self.global_to_free = np.full(self.n_dof, -1, dtype=np.int64)
        self.global_to_free[self.free_dofs] = np.arange(self.n_free)
        self._free_t = torch.as_tensor(self.free_dofs, device=self.device)

        is_coord = np.concatenate([np.r_[np.ones(3 * s.n_nodes, bool), np.zeros(s.n_edges, bool)]
                                   for s in self.shells])
        self._free_is_coord = torch.as_tensor(is_coord[self.free_dofs], device=self.device)

        # Contact offset per free node: the shell's own `contact_offset` (e.g. half its thickness)
        # or the global one, reduced where the rest geometry is already closer than that
        # (nodes next to the part a shell is attached to must not start in contact).
        self._contact_nodes, self._contact_offsets = {}, {}
        for s in self.shells:
            nodes = torch.nonzero(~s.fixed).flatten()
            offset = torch.full((len(nodes),), float(getattr(s, "contact_offset", None) or contact_offset),
                                dtype=DTYPE, device=self.device)
            if self.field is not None and len(nodes):
                rest_sd, _ = self.field.signed_distance(s.X[nodes], max_distance=offset.max().item() + 1e-9)
                offset = torch.minimum(offset, rest_sd.clamp_min(0.0))
            self._contact_nodes[id(s)], self._contact_offsets[id(s)] = nodes, offset

        X = torch.cat([s.X for s in self.shells])
        self.length_scale = (X.max(0).values - X.min(0).values).norm().item()
        self.max_step = max_step if max_step is not None else 0.1 * self.length_scale

    # -----------------------------
    # State
    # -----------------------------

    def get_u(self) -> torch.Tensor:
        return torch.cat([s.get_state() for s in self.shells])

    def set_u(self, u: torch.Tensor):
        for s in self.shells:
            off = self.dof_offset[id(s)]
            s.set_state(u[off:off + s.n_dof])

    # -----------------------------
    # Assembly
    # -----------------------------

    def evaluate(self, load_factor: float, tangent: bool = True):
        """Energy, residual (free dofs) and, optionally, the tangent pieces at the current state."""
        R = torch.zeros(self.n_dof, dtype=DTYPE, device=self.device)
        rows, cols, vals = [], [], []
        energy = 0.0

        def add_block(dofs, H, scale=1.0):
            d = dofs.shape[1]
            rows.append(dofs[:, :, None].expand(-1, d, d).reshape(-1))
            cols.append(dofs[:, None, :].expand(-1, d, d).reshape(-1))
            vals.append((scale * H).reshape(-1))

        # Shell elements
        for s in self.shells:
            off = self.dof_offset[id(s)]
            for dofs, e, g, H in s.element_terms(tangent):
                R.index_add_(0, (dofs + off).reshape(-1), g.reshape(-1))
                energy += e.sum().item()
                if tangent:
                    add_block(dofs + off, H)
        f_internal = R[self._free_t].norm().item()

        # Fluid pressure
        U, c = [], []
        f_pressure = torch.zeros_like(R)
        for v in self.fluid_volumes:
            dV = v.compute_delta_volume()
            P, dP, work = v.load_state(dV, load_factor)
            energy -= work

            g_vol = torch.zeros_like(R)
            for shell, side, _ in v.boundaries:
                xe = shell.x[shell.faces] - shell.volume_origin
                dofs = shell.face_dofs + self.dof_offset[id(shell)]
                g_vol.index_add_(0, dofs.reshape(-1), side * cone_volume_grad(xe).reshape(-1))
                if tangent and P != 0.0:
                    add_block(dofs, cone_volume_hess(xe).reshape(-1, 9, 9), -side * P)
            f_pressure += P * g_vol
            if tangent and dP != 0.0:
                U.append(g_vol[self._free_t].cpu().numpy())
                c.append(-dP)
        R -= f_pressure

        # Contact
        f_contact = torch.zeros_like(R)
        if self.field is not None and self.contact_stiffness > 0:
            for s in self.shells:
                nodes, offset = self._contact_nodes[id(s)], self._contact_offsets[id(s)]
                if not len(nodes):
                    continue
                sd, normal, curvature = self.field.signed_distance(
                    s.x[nodes], max_distance=offset.max().item() + 1e-9, hessian=True)
                gap = sd - offset
                active = gap < 0
                if not active.any():
                    continue
                nodes, gap, normal = nodes[active], gap[active], normal[active]
                ka = self.contact_stiffness * s.nodal_area[nodes]
                energy += (0.5 * ka * gap ** 2).sum().item()
                dofs = self.dof_offset[id(s)] + 3 * nodes[:, None] + torch.arange(3, device=self.device)
                f_contact.index_add_(0, dofs.reshape(-1), ((ka * gap)[:, None] * normal).reshape(-1))
                if tangent:
                    H = normal[:, :, None] * normal[:, None, :] + gap[:, None, None] * curvature[active]
                    add_block(dofs, ka[:, None, None] * H)
        R += f_contact

        out = {
            "energy": energy,
            "residual": R[self._free_t],
            "reference_force": max(f_internal, f_pressure[self._free_t].norm().item(),
                                   f_contact[self._free_t].norm().item()),
        }
        if tangent:
            r = self.global_to_free[torch.cat(rows).cpu().numpy()]
            col = self.global_to_free[torch.cat(cols).cpu().numpy()]
            val = torch.cat(vals).cpu().numpy()
            keep = (r >= 0) & (col >= 0)
            out["K"] = sp.coo_matrix((val[keep], (r[keep], col[keep])), shape=(self.n_free, self.n_free)).tocsc()
            out["U"] = np.stack(U, axis=1) if U else np.zeros((self.n_free, 0))
            out["c"] = np.asarray(c)
        return out

    # -----------------------------
    # Linear solve
    # -----------------------------

    @staticmethod
    def _solve_linear(K, U, c, rhs, damping):
        A = K + damping * sp.identity(K.shape[0], format="csc") if damping > 0 else K
        lu = spla.splu(A.tocsc())
        y = lu.solve(rhs)
        if U.shape[1]:
            Z = lu.solve(U)
            S = np.diag(1.0 / c) + U.T @ Z
            y = y - Z @ np.linalg.solve(S, U.T @ y)
        if not np.all(np.isfinite(y)):
            raise np.linalg.LinAlgError("non-finite Newton step")
        return y

    # -----------------------------
    # Newton iterations for one load level
    # -----------------------------

    def newton(self, load_factor: float):
        """Returns (converged, iterations) and leaves the shells in the final state."""
        damping_rel = 0.0
        for iteration in range(1, self.max_iterations + 1):
            state = self.evaluate(load_factor, tangent=True)
            R = state["residual"]
            r_norm = R.norm().item()
            tol = self.atol + self.rtol * state["reference_force"]
            if self.verbose:
                print(f"    it {iteration:2d}  |R| = {r_norm:.3e}  (tol {tol:.1e})  damping {damping_rel:.1e}")
            if self.callback is not None:
                self.callback(load_factor, iteration, r_norm)
            if r_norm <= tol:
                return True, iteration - 1

            K, U, c = state["K"], state["U"], state["c"]
            diag_scale = max(np.abs(K.diagonal()).mean(), 1e-300)
            rhs = -R.cpu().numpy()
            u0 = self.get_u().clone()

            while True:
                try:
                    du = self._solve_linear(K, U, c, rhs, damping_rel * diag_scale)
                except (RuntimeError, np.linalg.LinAlgError):
                    du = None
                if du is not None and damping_rel == 0.0 and np.abs(du).max() <= self.step_tol * self.length_scale:
                    return True, iteration  # residual is at its round-off floor
                if du is not None and self._line_search(u0, du, state, load_factor):
                    damping_rel = damping_rel * 0.1 if damping_rel > 1e-10 else 0.0
                    break
                self.set_u(u0)
                damping_rel = max(10.0 * damping_rel, 1e-8)
                if damping_rel > 1e8:
                    return False, iteration
        return False, self.max_iterations

    def _line_search(self, u0, du, state, load_factor, max_halvings: int = 8):
        du = torch.as_tensor(du, dtype=DTYPE, device=self.device)
        biggest = du[self._free_is_coord].abs().max().item() if self._free_is_coord.any() else 0.0
        if biggest > self.max_step:
            du = du * (self.max_step / biggest)

        # The potential energy is the merit function. Accepting on residual decrease as well would
        # let the iteration cycle (e.g. a node flipping in and out of contact), so the residual is
        # only used where energy differences are lost in round-off, close to the solution.
        slope = torch.dot(state["residual"], du).item()
        e0, r0 = state["energy"], state["residual"].norm().item()
        noise = 1e-12 * (abs(e0) + state["reference_force"] * self.length_scale)
        alpha = 1.0
        for _ in range(max_halvings + 1):
            u = u0.clone()
            u[self._free_t] += alpha * du
            self.set_u(u)
            trial = self.evaluate(load_factor, tangent=False)
            if slope < 0 and trial["energy"] <= e0 + 1e-4 * alpha * slope:
                return True
            if abs(trial["energy"] - e0) <= noise and trial["residual"].norm().item() < r0:
                return True
            alpha *= 0.5
        return False
