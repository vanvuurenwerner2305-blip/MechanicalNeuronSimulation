"""
Steady flow of a gas through a network of flow resistances (SI units: Pa, kg/s, m).

Nodes carry a gauge pressure: fixed (a constant-pressure fluid) or unknown (a dynamic fluid, or a
point between two segments of a channel). Edges are flow resistances between two nodes with a law

    dp = f(mdot, rho, mu, A, P, h, w, L, Dh, p, p_up, p_down, rho_up)
                                                    (Pa, for flow from the higher to the lower pressure)

that the user writes for a positive mass flow; the sign follows the pressure difference. The gas
density comes from the ideal gas law, rho = p_abs / (R_specific T): `rho` at the edge's mean pressure
(for friction along a passage), `rho_up` at the upstream pressure (for an orifice). `p`, `p_up` and
`p_down` are absolute pressures.
In steady state the mass flows into every unknown node add up to zero; those equations are solved
for the unknown pressures. Each edge's flow for a given pressure difference is found by inverting
its law (it must increase with mdot).
"""
import math

import numpy as np
from scipy.optimize import brentq, root

_NAMES = {name: getattr(np, name) for name in
          ("exp", "log", "log10", "sqrt", "sin", "cos", "tan", "tanh", "arctan", "minimum", "maximum", "clip",
           "abs", "where", "pi", "e")}
_NAMES.update(min=min, max=max, abs=abs)
LAW_VARIABLES = ("mdot", "rho", "mu", "A", "P", "h", "w", "L", "Dh", "p", "p_up", "p_down", "rho_up")

# Laminar flow through a passage of hydraulic diameter Dh = 4A/P (Hagen-Poiseuille for a circle)
SEGMENT_LAW = "32 * mu * L * mdot / (rho * A * Dh**2)"
# Sharp-edged orifice, discharge coefficient 0.61, upstream density
ORIFICE_LAW = "(mdot / (0.61 * A))**2 / (2 * rho_up)"


def compile_flow_law(text, default=SEGMENT_LAW):
    """dp (Pa) as a function of the variables in LAW_VARIABLES (SI units), from the user's expression."""
    text = (text or "").strip() or default
    code = compile(text, "<flow resistance>", "eval")
    unknown = [n for n in code.co_names if n not in _NAMES and n not in LAW_VARIABLES]
    if unknown:
        raise ValueError(f"Unknown name(s) in the flow resistance: {', '.join(unknown)}. "
                         f"Use {', '.join(LAW_VARIABLES)} and numpy functions.")

    def law(**variables):
        return float(eval(code, {"__builtins__": {}}, dict(_NAMES, **variables)))
    law.text = text
    return law


class Gas:
    def __init__(self, gas_constant=287.05, temperature=293.15, atmospheric_pressure=101325.0, viscosity=1.81e-5):
        self.R, self.T, self.p_atm, self.mu = gas_constant, temperature, atmospheric_pressure, viscosity

    def density(self, p_gauge):
        return max(p_gauge + self.p_atm, 1.0) / (self.R * self.T)


class FlowNetwork:
    def __init__(self, gas: Gas):
        self.gas = gas
        self.nodes = []       # {"name", "fixed": pressure or None}
        self.edges = []       # {"a", "b", "law", "geometry": dict, "name"}
        self.openings = []    # (a, b): joined without resistance (equal pressure)

    def add_node(self, name, pressure=None):
        self.nodes.append({"name": name, "fixed": pressure})
        return len(self.nodes) - 1

    def add_edge(self, a, b, law, name="", geometry=None):
        self.edges.append({"a": a, "b": b, "law": law, "name": name, "geometry": dict(geometry or {})})
        return len(self.edges) - 1

    def add_opening(self, a, b):
        self.openings.append((a, b))

    def _groups(self):
        """Union of the nodes joined by openings: representative of every node."""
        parent = list(range(len(self.nodes)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for a, b in self.openings:
            parent[find(a)] = find(b)
        return [find(i) for i in range(len(self.nodes))]

    # -----------------------------

    def edge_flow(self, edge, pa, pb):
        """Mass flow from a to b (kg/s) for gauge pressures pa, pb (Pa)."""
        dp = pa - pb
        if dp == 0.0:
            return 0.0
        g = edge["geometry"]
        p_mean, p_up, p_down = 0.5 * (pa + pb), max(pa, pb), min(pa, pb)
        variables = dict(rho=self.gas.density(p_mean), mu=self.gas.mu, p=p_mean + self.gas.p_atm,
                         p_up=p_up + self.gas.p_atm, p_down=p_down + self.gas.p_atm, rho_up=self.gas.density(p_up),
                         A=g.get("A", 0.0), P=g.get("P", 0.0), h=g.get("h", 0.0), w=g.get("w", 0.0),
                         L=g.get("L", 0.0), Dh=g.get("Dh", 0.0))
        if variables["A"] <= 0.0:
            return 0.0  # closed passage

        def residual(m):
            return edge["law"](mdot=m, **variables) - abs(dp)

        hi = 1e-12
        with np.errstate(all="ignore"):
            while residual(hi) < 0:
                hi *= 10.0
                if hi > 1e6:
                    raise ValueError(f"Flow resistance '{edge['name']}' never reaches the pressure difference "
                                     f"{abs(dp):.4g} Pa: its law must grow with mdot.")
            m = brentq(residual, 0.0, hi, xtol=1e-18, rtol=1e-12, maxiter=200)
        return math.copysign(m, dp)

    def solve(self, guess=None):
        """Node pressures (Pa gauge) and edge flows (kg/s, from a to b)."""
        group = self._groups()
        pressure = {}
        for i, n in enumerate(self.nodes):
            if n["fixed"] is not None:
                g = group[i]
                if g in pressure and abs(pressure[g] - n["fixed"]) > 1e-9 * max(1.0, abs(n["fixed"])):
                    raise ValueError(f"'{n['name']}' is joined by openings to a fluid held at a different constant "
                                     f"pressure: make one of the connections an orifice.")
                pressure[g] = n["fixed"]
        if len(set(group)) < len(group):  # solve on the joined nodes, then spread the pressures back
            reps = sorted(set(group))
            index = {r: k for k, r in enumerate(reps)}
            reduced = FlowNetwork(self.gas)
            for r in reps:
                reduced.add_node(self.nodes[r]["name"], pressure.get(r))
            for e in self.edges:
                reduced.add_edge(index[group[e["a"]]], index[group[e["b"]]], e["law"], e["name"], e["geometry"])
            g = None if guess is None else np.asarray(guess, float)[reps]
            p_red, flows = reduced.solve(g)
            return np.array([p_red[index[group[i]]] for i in range(len(self.nodes))]), flows
        fixed = np.array([n["fixed"] is not None for n in self.nodes])
        p = np.array([n["fixed"] if n["fixed"] is not None else 0.0 for n in self.nodes], float)
        free = np.nonzero(~fixed)[0]
        if len(free):
            known = p[fixed]
            if guess is not None and len(guess) == len(p):
                x0 = np.asarray(guess, float)[free]
            else:
                x0 = np.full(len(free), known.mean() if len(known) else 0.0)
            scale = max(np.abs(known).max() if len(known) else 0.0, 1.0)

            def balance(x):
                q = p.copy()
                q[free] = x * scale
                net = np.zeros(len(q))
                for e in self.edges:
                    m = self.edge_flow(e, q[e["a"]], q[e["b"]])
                    net[e["a"]] -= m
                    net[e["b"]] += m
                return net[free]

            # the flows can be tiny: scale the balance by the typical flow at the start
            ref = max(np.abs([self.edge_flow(e, p[e["a"]] if fixed[e["a"]] else x0.mean(),
                                             p[e["b"]] if fixed[e["b"]] else x0.mean()) for e in self.edges]).max()
                      if self.edges else 0.0, 1e-30)
            solution = root(lambda x: balance(x) / ref, x0 / scale, method="hybr", options={"xtol": 1e-12})
            if not solution.success and np.abs(balance(solution.x) / ref).max() > 1e-6:
                solution = root(lambda x: balance(x) / ref, x0 / scale, method="lm", options={"xtol": 1e-14})
            p[free] = solution.x * scale
        flows = np.array([self.edge_flow(e, p[e["a"]], p[e["b"]]) for e in self.edges])
        return p, flows
