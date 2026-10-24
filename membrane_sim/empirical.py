"""
A membrane replaced by its pre-simulated (empirical) response.

An activation-function device (membrane -> pusher -> squeezed tube) is simulated once on its own;
what the rest of a model needs from its membrane is only how much volume it sweeps at a pressure
difference: V(dp), dp = pressure on the driving side - pressure on the tube side. In a neuron the
device's membrane is one wall of the pre-activation chamber, so it is replaced by that curve:

  one dof q (mm): the mean deflection towards the tube, swept volume v = A q (A = membrane area);
  stored energy E(v) = int_0^v dp(s) ds, with dp(v) the inverse of the simulated V(dp);
  as a chamber wall it adds side * v to the chamber's volume change (side = +1 for the driving
  chamber, -1 for a chamber on the tube side), so at equilibrium dp(v) = P_drive - P_tube.

dp(v) is a monotone cubic (PCHIP) through the simulated points (plus (0, 0) when dp = 0 was not
simulated), extended linearly with the end slopes. Without simulated points below dp = 0 the curve is
mirrored about its dp = 0 point (pulled away from the tube the membrane is free on both sides).
"""
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

DTYPE = torch.float64


def empirical_curve(dp, volume):
    """Sorted (dp, V) through the rest state, strictly increasing in both; mirrored when it has no dp < 0.
    Points where the volume does not increase with dp (solver noise once the tube is shut) are dropped."""
    dp, volume = np.asarray(dp, float), np.asarray(volume, float)
    ok = np.isfinite(dp) & np.isfinite(volume)
    dp, volume = dp[ok], volume[ok]
    if not np.any(np.abs(dp) <= 1e-12 * max(np.abs(dp).max(initial=0.0), 1.0)):
        dp, volume = np.r_[dp, 0.0], np.r_[volume, 0.0]
    order = np.argsort(dp, kind="stable")
    dp, volume = dp[order], volume[order]
    keep = [int(np.argmin(np.abs(dp)))]  # the rest state, then outwards while both keep growing
    for k in range(keep[0] + 1, len(dp)):
        if dp[k] > dp[keep[-1]] and volume[k] > volume[keep[-1]]:
            keep.append(k)
    for k in range(keep[0] - 1, -1, -1):
        if dp[k] < dp[keep[0]] and volume[k] < volume[keep[0]]:
            keep.insert(0, k)
    dp, volume = dp[keep], volume[keep]
    if not (dp < 0).any():  # mirrored about the dp = 0 point (the tube's gas may hold the membrane off 0 there)
        pos, v0 = dp > 0, volume[0]
        dp = np.r_[-dp[pos][::-1], dp[~pos], dp[pos]]
        volume = np.r_[2 * v0 - volume[pos][::-1], volume[~pos], volume[pos]]
    if len(dp) < 2:
        raise ValueError("The design's volume does not change with the pressure difference.")
    return dp, volume


class EmpiricalMembrane:
    sheet_contact = False
    is_rigid = True      # no nodes: the solver's per-node contact bookkeeping skips it
    n_nodes = 0
    n_edges = 0
    n_dof = 1
    thickness = 0.0
    youngs_modulus = 0.0
    material = "empirical"

    def __init__(self, dp, volume, area: float, name: str = None, color: str = "#e4572e", device="cpu"):
        """
        dp, volume : the simulated response, pressure difference (model units) and swept volume.
        area       : membrane area; scales the dof to a mean deflection (a length, like the other dofs).
        """
        self.device = torch.device(device)
        self.name, self.color = name, color
        self.area = float(area)
        if self.area <= 0:
            raise ValueError("The membrane area must be positive.")
        self.length_scale = self.area ** 0.5  # model size when nothing else is in the model
        self.curve_dp, self.curve_volume = empirical_curve(dp, volume)
        self._dp = PchipInterpolator(self.curve_volume, self.curve_dp, extrapolate=False)
        self._slope = self._dp.derivative()
        self._energy = self._dp.antiderivative()
        self._e0 = float(self._energy(0.0))
        v, p = self.curve_volume, self.curve_dp
        self._ends = [(v[0], p[0], float(self._slope(v[0])), float(self._energy(v[0])) - self._e0),
                      (v[-1], p[-1], float(self._slope(v[-1])), float(self._energy(v[-1])) - self._e0)]

        self.X = torch.zeros((0, 3), dtype=DTYPE, device=self.device)
        self.x = self.X
        self.q = torch.zeros(1, dtype=DTYPE, device=self.device)
        self.fixed = torch.zeros(0, dtype=torch.bool, device=self.device)
        self.fixed_dofs = torch.zeros(1, dtype=torch.bool, device=self.device)
        self.coord_mask = np.array([True])  # a mean deflection: a length
        self.fluid_volumes = []
        self._dofs = torch.zeros((1, 1), dtype=torch.long, device=self.device)

    # -----------------------------
    # Response
    # -----------------------------

    def response(self, v: float):
        """(energy, dp, d(dp)/dv) at swept volume v."""
        for k, (ve, pe, ke, ee) in enumerate(self._ends):
            if (k == 0 and v < ve) or (k == 1 and v > ve):
                d = v - ve
                return ee + pe * d + 0.5 * ke * d * d, pe + ke * d, ke
        return float(self._energy(v)) - self._e0, float(self._dp(v)), float(self._slope(v))

    def pressure_difference(self, v=None) -> float:
        return self.response(self.swept_volume() if v is None else v)[1]

    def swept_volume(self) -> float:
        return self.area * float(self.q[0])

    # -----------------------------
    # Solver interface
    # -----------------------------

    def get_state(self):
        return self.q

    def set_state(self, u):
        self.q = u

    def reset(self):
        self.q = torch.zeros_like(self.q)

    def element_terms(self, tangent=True):
        e, dp, k = self.response(self.swept_volume())
        A = self.area
        t = lambda value: torch.tensor(value, dtype=DTYPE, device=self.device)  # noqa: E731
        yield (self._dofs, t([e]), t([[A * dp]]), t([[[A * A * k]]]) if tangent else None)

    def volume_terms(self, side: int, tangent: bool):
        """(local dofs, d(side v)/dq, d2(side v)/dq2) as a chamber wall."""
        g = torch.tensor([[side * self.area]], dtype=DTYPE, device=self.device)
        return self._dofs, g, torch.zeros((1, 1, 1), dtype=DTYPE, device=self.device) if tangent else None
