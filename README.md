# Membrane Neuron Simulator

Desktop simulation software for fluid-driven membranes and shells. You build the device in CAD,
import it as STEP, click each solid to give it a role (membrane, shell, rigid body, fluid chamber),
and solve for static equilibrium with a Newton–Raphson solver.

```
python run_app.py                    # start the application
python run_app.py model.step         # ...and import a STEP file
python run_app.py project.mns        # ...or open a saved project
python -m pytest tests               # solver benchmarks + application pipeline tests
```

Requirements (Anaconda): numpy, scipy, torch, pyvista, pyvistaqt, PyQt5, matplotlib, plus
`pip install gmsh` for STEP import and meshing.

## Workflow

1. **Model in CAD** so that every region is its own solid body:
   - each membrane or shell is a thin solid;
   - each fluid chamber is a solid that fills the fluid space;
   - frames and obstacles are solids.

   Export the assembly or multi-body part as STEP, in millimetres. Part names (or Fusion 360
   body names) are carried over.
2. **File → Open STEP.** Every solid shows up in the model tree.
3. **Assign roles.** Click a part in the 3D view (Ctrl+click adds more), or select it in the tree,
   then pick its role. To reach parts inside the frame, untick the frame in the tree or use a
   section cut. *Edit → Auto-assign roles from names* guesses roles from names like `Membrane_1`
   or `Chamber_A`.
4. **Set properties:**
   - membranes/shells: material, Young's modulus, thickness (0 = measured from the CAD solid),
     element size, which edges are fixed;
   - chambers: pressure model and pressure (gauge: the surroundings are 0 kPa, and a membrane face
     that touches no chamber sees 0 kPa):
     - **Constant pressure** (green): the inputs.
     - **Closed: ideal gas** (blue, darker = more liquid): sealed air, optionally partly filled
       with incompressible liquid.
     - **Closed: incompressible** (purple): sealed and full of liquid. Stiffness is in kPa per %
       volume change; the default 1 000 kPa/% already gives results indistinguishable from water.
     - **Vent** (transparent): open to the surroundings, always 0 kPa.
5. **Mesh / Check model** (Ctrl+M / Ctrl+K). Membranes and shells are replaced by their
   mid-surfaces, and fixed nodes are shown in blue. The message log lists which chamber acts on
   which membrane.
6. **Solve** (F5) and look at the **Results** tab:
   - displacement or area-stretch contours, with a slider through the load steps;
   - a table of chamber pressures and volumes;
   - VTK and CSV export.
7. **Sweep** (F6) maps the response over one or two input chamber pressures (line plot or heat
   map, CSV export). Each point starts from the previous solution, so sweeps are several times
   faster than separate solves.

`examples/make_neuron_step.py` writes `examples/soft_neuron.step`, the soft neuron as a named
STEP assembly, for trying the workflow.

### How CAD solids become the simulation model

| Role | Simulated as |
|---|---|
| Membrane | Shell on the mid-surface of the solid, no bending stiffness |
| Shell | Same, with bending stiffness |
| Rigid body | Contact obstacle (closed surface mesh) |
| Fluid chamber | Pressure on every membrane/shell whose face it touches; closed chambers change pressure with volume |
| Ignore / Unassigned | Left out |

- The mid-surface is built from the largest CAD face of the thin solid. Each node is moved half the
  locally measured thickness into the solid. A membrane should therefore have one main face (a
  plate, disc or dome made as a single face).
- Membrane edges are clamped: either every boundary edge, or only the edges that touch rigid bodies.
- Chamber-membrane coupling is found geometrically: points just beyond each membrane face are
  tested against every chamber solid.

## Layout

```
app/                    desktop application (PyQt5 via qtpy, PyVista viewport, gmsh CAD kernel)
  cad.py                STEP import, per-body surface meshing, mid-surface extraction
  project.py            roles, property schemas, solver settings, .mns project files
  builder.py            project + CAD -> membrane_sim.Environment, chamber/membrane coupling detection
  main_window.py        main window; panels.py, viewport.py, sweep.py, workers.py
membrane_sim/           solver core (usable on its own, see examples/soft_neuron_3d.py)
  shell.py              large-strain membrane triangle + Morley bending (node coords + edge rotations)
  fluid.py              FluidVolume: chamber pressure P(dV); dV from the moving shells only
  contact.py            exact signed distance to closed triangle meshes, penalty contact
  solver.py             Newton-Raphson: consistent tangent, sparse LU + Woodbury, line search
  environment.py        Environment: build, solve (fresh or warm-started), reset
examples/               soft neuron script and STEP generator
cad_models/             user CAD files
tests/                  analytical benchmarks, consistency checks, application pipeline tests
```

## Solver model

- **Membrane**: plane-stress triangle on the exact surface deformation gradient. Incompressible
  neo-Hookean (the default, for rubber) or St. Venant–Kirchhoff. Optional pre-tension.
- **Bending**: Morley triangle with one mid-edge rotation unknown per edge. It is independent of
  mesh orientation and converges to Kirchhoff plate theory.
- **Fluid chambers** (all pressures gauge, 0 = atmospheric):
  - constant pressure (inputs);
  - closed ideal gas (isothermal) with an optional incompressible share. With a fraction `f` of
    liquid the gas volume is `Vg = (1 - f) V0`, all volume change goes into the gas, and
    `P = (Patm + P0) Vg / (Vg + dV) - Patm`. The gas can never be compressed to zero volume;
  - closed incompressible: `P = P0 - s * 100 dV / V0` with `s` in pressure per % volume change;
  - vent: always 0;
  - or any differentiable `pressure_law(dV, P0)` in scripts.

  Chamber walls are not meshed: shell edges are fixed, so `dV` follows from the shells alone.
- **Contact**: nodal penalty `1/2 k A_node gap^2` against rigid bodies, with the mid-surface kept
  half a thickness away. `k` is pressure per unit penetration. The automatic value gives about 5%
  of the thickness at the highest input pressure.
- **Solve**: pressures are ramped by a load factor from 0 to 1, and a step is cut automatically
  when Newton fails. A warm start ramps only the change from the previous solution.

Units in the application: mm, N, MPa (pressures entered in kPa).

## Verification (tests/)

| Check | Result |
|---|---|
| Residual = dEnergy, tangent = dResidual (finite differences, with pressure, volume stiffness, contact, bending) | 1e-5 rel |
| Clamped square plate, `w = 0.00126 q a^4 / D` | 32x32 mesh: +2.5 % (8x8: +33 %, 16x16: +9 %) |
| Clamped circular plate, `w = q a^4 / 64D` | 24 rings: +0.6 % |
| Pure-bending energy on a cylinder at 0/30/45° mesh orientation | orientation-independent, converges O(h) |
| Hencky inflated membrane, `w = 0.655 a (qa/Et)^(1/3)` | 0.654 |
| Contact signed distance vs brute force (non-convex prism, body with cavity) | exact |
| Warm-started solve vs fresh solve | identical |
| Gas chamber with incompressible share: energy/residual/tangent consistency, Boyle's law on the gas share, gas pocket never collapses | exact |
| STEP import: names, volumes, outward meshes, mid-surfaces, chamber couplings, project round trip | exact |

## Known limitations

- Membranes and shells do not contact each other, only rigid bodies. Only a closed chamber's gas
  keeps two membranes apart.
- Contact is checked at mesh nodes, so use an element size small enough to follow the obstacle's
  curvature.
- A membrane's mid-surface comes from its single largest CAD face.
