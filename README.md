# MembraneNeuronSimulator

3D successor of `Sim_Build_v2/simulation.py`. Membranes are triangulated thin shells and the
equilibrium under fluid pressure and obstacle contact is found directly with Newton–Raphson,
instead of integrating damped dynamics until the system settles.

```
membrane_sim/
  shell.py        Shell: large-strain membrane triangle + Morley bending (node coords + edge rotations)
  fluid.py        FluidVolume: chamber pressure P(dV); dV from the moving shells only
  contact.py      Obstacle / ObstacleField: exact signed distance to closed triangle meshes, penalty contact
  solver.py       NewtonSolver: consistent tangent, sparse LU + Woodbury, line search, load stepping
  environment.py  Environment: builds the model, solve(), reset(), rendering
  mesh.py         rectangle/disk membrane meshes, box and extruded-polygon obstacles
  render.py       Plotly 3D output (scene, load-path animation)
examples/soft_neuron_3d.py   the notebooks' SoftNeuron rebuilt in 3D
tests/                       analytical benchmarks and tangent consistency checks
```

Run the tests with `python -m pytest tests` and the example with
`python examples/soft_neuron_3d.py --show`.

## Model

- **Membrane**: plane-stress triangle on the exact surface deformation gradient; incompressible
  neo-Hookean (default, rubber) or St. Venant–Kirchhoff. Optional pre-tension.
- **Bending**: rotation-free-in-the-nodes Morley triangle. Each edge has a mid-edge normal rotation
  unknown; clamped boundary edges hold the rest slope (`boundary_rotation="clamped"`, default) or are
  hinged (`"free"`). Set `bending=False` for a pure membrane (a flat start then relies on the
  solver's damping fallback or on `pretension`).
- **Fluid chambers**: `P = P0 - K dV`, or any differentiable `pressure_law(dV)`. The walls are not
  meshed. Because shell edges are pinned, dV comes from the shells alone, so a chamber is just a
  list of `(shell, side)` pairs. `shell.fluid_volume_contacts((behind, in_front))` works as in 2D:
  the chamber behind the shell (opposite its normal) pushes along +normal.
- **Obstacles**: closed triangle meshes (outward normals); `inverted=True` makes the outside solid
  (container walls). Contact is a nodal penalty `1/2 k A_node gap^2`, so `k` is pressure per unit
  penetration (penetration ≈ contact pressure / k).
- **Solve**: all pressures are ramped by a load factor 0 → 1 (`load_steps`). Steps are cut
  automatically when Newton fails. `env.history` stores every converged load step.

Units are whatever you use consistently. The 2D code's `depth`, `mass` and `dampening` no longer
exist: depth is real geometry now, and a static solve has no mass or damping.

## Verification (tests/)

| Check | Result |
|---|---|
| Residual = dEnergy, tangent = dResidual (finite differences, with pressure, volume stiffness, contact, bending) | 1e-5 rel |
| Clamped square plate, `w = 0.00126 q a^4 / D` | 32x32 mesh: +2.5 % (8x8: +33 %, 16x16: +9 %) |
| Clamped circular plate, `w = q a^4 / 64D` | 24 rings: +0.6 % |
| Pure-bending energy on a cylinder at 0/30/45° mesh orientation | orientation-independent, converges O(h) |
| Hencky inflated membrane, `w = 0.655 a (qa/Et)^(1/3)` | 0.654 |
| Imposed paraboloid volume, stiff chamber vs constant pressure, custom law vs linear law | exact to tolerance |
