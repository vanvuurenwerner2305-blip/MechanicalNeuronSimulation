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

Each Newton iteration tries, in order, until the potential energy decreases:
  1. the Newton step, with a second-order correction (SOC) for closed chambers: chamber volume
     is quadratic in the displacements, so along a straight step a stiff chamber sees a volume
     error that its stiffness turns into a huge energy (the Maratos effect of SQP). The SOC moves
     the trial point back onto the linearised volumes, so stiff and soft chambers converge alike;
  2. Levenberg-Marquardt damped Newton steps (K + mu I), for indefinite or singular tangents;
  3. a Jacobi-preconditioned gradient step, which always decreases the energy.
Every candidate step is capped in size and accepted by a line search on the potential energy
(Armijo condition, quadratic interpolation). Pressures are ramped with a load factor lambda in
[0, 1] with automatic step cutting (see Environment.solve).
"""
import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .contact import CachedSignedDistance, ObstacleField
from .shell_contact import ShellContact

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
                 max_iterations: int = 40, max_step: float = None, verbose: bool = False, callback=None,
                 couplings=(), surface_contacts=()):
        """
        shells           : the bodies with dofs: Shell, Solid (membrane_sim.solid) or RigidBody
                           (membrane_sim.rigid); anything with the same interface.
        couplings        : extra energy terms between bodies, e.g. RigidTie / MovingContact; each has
                           bind(solver) and terms(tangent) -> [(global dofs, energy, gradient, Hessian)].
        surface_contacts : extra ShellContact objects built from explicit node/face pairs (e.g. the
                           inside walls of a tube touching each other).
        """
        self.shells = list(shells)
        self.fluid_volumes = list(fluid_volumes)
        self.field = ObstacleField(obstacles) if obstacles else None
        self.contact_stiffness = contact_stiffness
        self.contact_offset = contact_offset
        self.rtol, self.atol, self.step_tol = rtol, atol, step_tol
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.callback = callback  # called as callback(load_factor, iteration, residual_norm)
        self._damping = 0.0       # Levenberg-Marquardt damping carried between iterations
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

        is_coord = np.concatenate([np.asarray(s.coord_mask, bool) if hasattr(s, "coord_mask") else
                                   np.r_[np.ones(3 * s.n_nodes, bool), np.zeros(s.n_edges, bool)]
                                   for s in self.shells])
        self._free_is_coord = torch.as_tensor(is_coord[self.free_dofs], device=self.device)

        # Contact offset per free node: the shell's own `contact_offset` (e.g. half its thickness)
        # or the global one, reduced where the rest geometry is already closer than that.
        # Nodes sharing an element with a fixed node that start at a wall (within 1.25x their offset,
        # i.e. at the wall the shell is clamped to) get no rigid contact: they sit right at the
        # penalty's on/off kink, and flipping in and out of contact made Newton cycle
        # (NeuronTest.mns stalled at 1% residual).
        self._contact_nodes, self._contact_offsets = {}, {}
        for s in self.shells:
            if getattr(s, "is_rigid", False):
                continue
            fixed = s.fixed.cpu().numpy()
            next_to_clamp = np.zeros_like(fixed)
            next_to_clamp[s.faces_np[fixed[s.faces_np].any(axis=1)]] = True
            next_to_clamp &= ~fixed
            candidates = torch.zeros_like(s.fixed)
            candidates[getattr(s, "surface_nodes", torch.arange(s.n_nodes, device=self.device))] = True
            nodes = torch.nonzero(candidates & ~s.fixed).flatten()
            offset = torch.full((len(nodes),), float(getattr(s, "contact_offset", None) or contact_offset),
                                dtype=DTYPE, device=self.device)
            if self.field is not None and len(nodes):
                overlap = getattr(s, "allow_initial_overlap", False)
                reach = 1.25 * offset.max().item() + (0.01 * _size(s.X) if overlap else 1e-9)
                rest_sd, _ = self.field.signed_distance(s.X[nodes], max_distance=reach)
                at_clamp_wall = torch.as_tensor(next_to_clamp, device=self.device)[nodes] & (rest_sd <= 1.25 * offset)
                nodes, offset, rest_sd = nodes[~at_clamp_wall], offset[~at_clamp_wall], rest_sd[~at_clamp_wall]
                # Solids may start exactly on an obstacle (or overlap it by the faceting of curved CAD
                # faces): that rest contact is neutral. Sheets keep at least a zero offset.
                offset = torch.minimum(offset, rest_sd if overlap else rest_sd.clamp_min(0.0))
            self._contact_nodes[id(s)], self._contact_offsets[id(s)] = nodes, offset

        X = torch.cat([s.X for s in self.shells])
        self.length_scale = (X.max(0).values - X.min(0).values).norm().item()
        self.max_step = max_step if max_step is not None else 0.1 * self.length_scale

        # Candidate lists for the rigid contact queries, rebuilt when a node has moved more than
        # the skin (exact: the same result as searching every obstacle triangle each time)
        self._rigid_cache = {}
        if self.field is not None:
            for s in self.shells:
                offset = self._contact_offsets.get(id(s), ())
                if len(offset):
                    reach = offset.max().item() + 1e-9
                    self._rigid_cache[id(s)] = CachedSignedDistance(
                        self.field, reach, skin=max(reach, 0.01 * self.length_scale))

        # Contact between membranes/shells (and any explicit surface contact pairs)
        sheets = [s for s in self.shells if getattr(s, "sheet_contact", True)]
        self.surface_contacts = []
        if contact_stiffness > 0 and len(sheets) > 1:
            contact = ShellContact(sheets, contact_stiffness, search_distance=2 * self.max_step)
            if contact:
                self.surface_contacts.append(contact)
        for contact in surface_contacts:
            if contact:
                contact.search_distance = max(contact.search_distance, 2 * self.max_step)
                self.surface_contacts.append(contact)
        self.shell_contact = self.surface_contacts[0] if self.surface_contacts else None
        self.couplings = list(couplings)
        for coupling in self.couplings:
            coupling.bind(self)

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
        U, c, closed = [], [], []
        f_pressure = torch.zeros_like(R)
        for v in self.fluid_volumes:
            dV = v.compute_delta_volume()
            P, dP, work = v.load_state(dV, load_factor)
            energy -= work

            g_vol = torch.zeros_like(R)
            for body, dofs, g, H in v.volume_terms(tangent and P != 0.0):
                dofs = dofs + self.dof_offset[id(body)]
                g_vol.index_add_(0, dofs.reshape(-1), g.reshape(-1))
                if H is not None:
                    add_block(dofs, H, -P)
            f_pressure += P * g_vol
            if tangent and dP != 0.0:
                U.append(g_vol[self._free_t].cpu().numpy())
                c.append(-dP)
                closed.append((v, dV))
        R -= f_pressure

        # Contact
        f_contact = torch.zeros_like(R)
        if self.field is not None and self.contact_stiffness > 0:
            for s in self.shells:
                if id(s) not in self._contact_nodes:
                    continue
                nodes, offset = self._contact_nodes[id(s)], self._contact_offsets[id(s)]
                if not len(nodes):
                    continue
                sd, normal, curvature = self._rigid_cache[id(s)].signed_distance(s.x[nodes], hessian=True)
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
        for contact in self.surface_contacts:
            for A, B, nodes, tri_nodes, e, g, H in contact.terms(tangent):
                arange = torch.arange(3, device=self.device)
                dofs = torch.cat([(self.dof_offset[id(A)] + 3 * nodes[:, None] + arange),
                                  (self.dof_offset[id(B)] + 3 * tri_nodes[:, :, None] + arange).reshape(-1, 9)],
                                 dim=1)
                energy += e.sum().item()
                f_contact.index_add_(0, dofs.reshape(-1), g.reshape(-1))
                if tangent:
                    add_block(dofs, H)
        for coupling in self.couplings:
            for dofs, e, g, H in coupling.terms(tangent):
                energy += e.sum().item()
                f_contact.index_add_(0, dofs.reshape(-1), g.reshape(-1))
                if tangent:
                    add_block(dofs, H)
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
            out["closed"] = closed  # (volume, dV) for each column of U
        return out

    # -----------------------------
    # Newton iterations for one load level
    # -----------------------------

    def newton(self, load_factor: float):
        """Returns (converged, iterations) and leaves the shells in the final state."""
        damping = self._damping
        best, stalled = np.inf, 0
        for iteration in range(1, self.max_iterations + 1):
            state = self.evaluate(load_factor, tangent=True)
            R = state["residual"]
            r_norm = R.norm().item()
            tol = self.atol + self.rtol * state["reference_force"]
            if self.verbose:
                print(f"    it {iteration:2d}  |R| = {r_norm:.3e}  (tol {tol:.1e})  damping {damping:.1e}")
            if self.callback is not None:
                self.callback(load_factor, iteration, r_norm)
            if r_norm <= tol:
                self._damping = damping
                return True, iteration - 1
            # Contact energies are only C1 where the closest point crosses a triangle edge, which
            # puts a floor under the attainable residual: accept a stalled iteration that is
            # already within 1000x of the tolerance.
            if r_norm < 0.5 * best:
                best, stalled = r_norm, 0
            else:
                stalled += 1
            if stalled >= 3 and r_norm <= 1e3 * tol:
                self._damping = damping
                return True, iteration - 1

            K, U, c = state["K"], state["U"], state["c"]
            diag = K.diagonal()
            scale = max(np.abs(diag).mean(), 1e-300)
            gradient = R.cpu().numpy()
            u0 = self.get_u().clone()

            # 1-2) Newton step, then increasingly damped Newton steps
            accepted = False
            for attempt in range(4):
                try:
                    tangent = _Tangent(K, U, c, damping * scale)
                    step = tangent.solve(-gradient)
                except (RuntimeError, np.linalg.LinAlgError):
                    tangent = step = None
                if step is not None:
                    if damping == 0.0 and np.abs(step).max() <= self.step_tol * self.length_scale:
                        self._damping = damping
                        return True, iteration  # residual is at its round-off floor
                    alpha = self._line_search(u0, step, state, load_factor, tangent)
                    if alpha:
                        accepted = True
                        if attempt == 0 and alpha == 1.0:
                            damping = damping * 0.1 if damping > 1e-9 else 0.0
                        break
                damping = max(10.0 * damping, 1e-6)

            # 3) first-order fallback: preconditioned steepest descent
            if not accepted:
                self.set_u(u0)
                precondition = np.abs(diag) + (U ** 2 @ np.abs(c) if U.shape[1] else 0.0) + 1e-8 * scale
                if not self._line_search(u0, -gradient / precondition, state, load_factor, None,
                                         max_backtracks=40):
                    self.set_u(u0)
                    self._damping = damping
                    return False, iteration
                if self.verbose:
                    print("      (gradient step)")
        self._damping = damping
        return False, self.max_iterations

    def _line_search(self, u0, step, state, load_factor, tangent, max_backtracks: int = 16):
        """
        Backtracking line search on the potential energy. Returns the accepted step length, or 0.
        With a tangent factorisation, trial points get the second-order correction that restores
        the closed-chamber volumes to their linearised values.
        """
        step = torch.as_tensor(step, dtype=DTYPE, device=self.device)
        biggest = step[self._free_is_coord].abs().max().item() if self._free_is_coord.any() else 0.0
        if biggest > self.max_step:
            step = step * (self.max_step / biggest)
        gradient = state["residual"]
        slope = torch.dot(gradient, step).item()
        if not slope < 0:
            return 0.0

        e0, r0 = state["energy"], gradient.norm().item()
        noise = 1e-12 * (abs(e0) + state["reference_force"] * self.length_scale)
        closed = state.get("closed", []) if tangent is not None else []
        if closed:
            step_np = step.cpu().numpy()
            linear_rate = state["U"].T @ step_np  # d(dV)/d(alpha) of each closed chamber

        alpha = 1.0
        if self.surface_contacts:  # never let a node pass through another sheet
            full = torch.zeros(self.n_dof, dtype=DTYPE, device=self.device)
            full[self._free_t] = step
            motion = {id(s): full[self.dof_offset[id(s)]:self.dof_offset[id(s)] + 3 * s.n_nodes].reshape(-1, 3)
                      for s in self.shells}
            alpha = min([1.0] + [c.safe_step(motion) for c in self.surface_contacts])
            if alpha < 1e-10:
                return 0.0
        for _ in range(max_backtracks):
            u = u0.clone()
            u[self._free_t] += alpha * step
            self.set_u(u)
            if closed:
                error = np.array([v.compute_delta_volume() - (dV + alpha * rate)
                                  for (v, dV), rate in zip(closed, linear_rate)])
                correction = tangent.volume_correction(error)
                if np.all(np.isfinite(correction)):
                    u[self._free_t] += torch.as_tensor(correction, dtype=DTYPE, device=self.device)
                    self.set_u(u)
            trial = self.evaluate(load_factor, tangent=False)
            energy = trial["energy"]
            if energy <= e0 + 1e-4 * alpha * slope:
                return alpha
            if abs(energy - e0) <= noise and trial["residual"].norm().item() < r0:
                return alpha
            # minimiser of the quadratic through phi(0), phi'(0) and phi(alpha), safeguarded
            if np.isfinite(energy):
                curvature = energy - e0 - slope * alpha
                guess = -slope * alpha ** 2 / (2.0 * curvature) if curvature > 0 else 0.5 * alpha
                alpha = min(max(guess, 0.1 * alpha), 0.5 * alpha)
            else:
                alpha *= 0.1
            if alpha < 1e-10:
                break
        return 0.0


def _size(X):
    return (X.max(0).values - X.min(0).values).norm().item() if len(X) else 0.0


class _Tangent:
    """
    Factorised tangent: sparse LU of K (+ damping I), with the dense closed-chamber terms
    U diag(c) U^T added through the Sherman-Morrison-Woodbury identity.
    """

    def __init__(self, K, U, c, damping=0.0):
        A = K + damping * sp.identity(K.shape[0], format="csc") if damping > 0 else K
        self.lu = spla.splu(A.tocsc())
        self.U, self.c = U, c
        if U.shape[1]:
            self.Z = self.lu.solve(U)                   # A^-1 U
            self.UZ = U.T @ self.Z
            self.S = np.diag(1.0 / c) + self.UZ

    def solve(self, rhs):
        y = self.lu.solve(rhs)
        if self.U.shape[1]:
            y = y - self.Z @ np.linalg.solve(self.S, self.U.T @ y)
        if not np.all(np.isfinite(y)):
            raise np.linalg.LinAlgError("non-finite Newton step")
        return y

    def volume_correction(self, error):
        """Smallest correction (in the A-norm) that changes the closed-chamber volumes by -error."""
        return -self.Z @ np.linalg.solve(self.UZ, error)
