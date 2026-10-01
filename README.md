# Membrane Neuron Simulator

Desktop simulation software for fluid-driven membranes and shells. You build the device in CAD,
import it as STEP, click each solid to give it a role (membrane, shell, rigid body, fluid chamber),
and solve for static equilibrium with a Newton–Raphson solver.

The window has two **spaces** (tabs at the top), each with its own model:

- **Neuron: inputs → activation** — membranes between fluid chambers, sweeps and the neuron equation.
- **Activation function** — the valve that turns the activation pressure into a tube's open area:
  simulate it once and save it as an activation-function design (`*.mad`) to use as a part.

```
python run_app.py                    # start the application
python run_app.py model.step         # ...and import a STEP file
python run_app.py project.mns        # ...or open a saved neuron project
python run_app.py valve.mad          # ...or an activation-function design
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
       volume change (default 10 kPa/%; water is 22 000, and above ~1 000 the results barely change
       while the solve gets slower).
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
   faster than separate solves. With an **activation chamber** chosen, every point also records
   the mechanical weight of each input path into it, **W_j = dV_j / (p_j − p_a)**, where dV_j is the
   volume the path pushes into the activation chamber. A path lumps everything between input
   chamber j and the activation chamber: one membrane, or membrane – weight chamber – membrane
   for bulk modulus tuning. A membrane with nothing behind it is an input from the surroundings
   (0 kPa). The activation chamber's own fluid adds W_0 about its rest pressure p_0, so
   p_a = (Σ W_j p_j + W_0 p_0) / (Σ W_j + W_0); the table shows this rebuilt p_a as a check.
   After the sweep the weights are fitted as polynomials W(Δp), **one for Δp > 0 and one for
   Δp < 0** (a weight sampled on one side only uses that polynomial for both; points with Δp = 0
   are left out; the least squares is weighted by |Δp|, i.e. it fits the displaced volume W·Δp),
   of the lowest degrees for which the equation, solved for p_a from each point's inputs, is
   within the **equation tolerance** (default 1 kPa) of every simulated p_a. Every side of every
   weight has its own degree, all starting constant.
   *Degree search*: **Lowest total order** (default) tries every combination of degrees with total
   order 0, 1, 2, ... and stops at the first that meets the tolerance, so the result is the lowest
   possible total order; **Biggest own error first** instead keeps giving one order more to the
   weight that causes the biggest error on its own (faster, but errors of different weights can
   cancel, so it can end higher). Change the tolerance or search and press **Regenerate equation**
   to refit from the sweep's results without simulating again. The **Neuron equation** tab shows
   the equation rendered (click a weight to plot its fitted polynomial over the sampled points
   underneath) and as LaTeX (Copy LaTeX, two-sided weights as `cases`), with its error. The
   fit runs in the background; with many sides and a tight tolerance the exhaustive search can
   take tens of seconds. The export
   writes `<name>.csv` (all points), `<name>_weights.csv` (the fits) and `<name>_equation.tex`.

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

## Activation function space

The device is a valve: the pressure difference Δp across a membrane squeezes a soft tube, through a
part bonded to the membrane (e.g. a pusher, rigid or deformable), against a rigid body. Gas flows
through the tube from a supply to a sink, and the software computes how the tube's cross-section,
the mass flow and the pressures along the tube change with Δp - how the pressure divides depending
on how far the tube is clamped.

1. In CAD, make solids for the membrane, the pusher (touching the membrane's face and the tube), the
   tube, the rigid body behind it, and the gas: one body filling the tube, a supply body against its
   inlet end and a sink body against its outlet end. `examples/make_activation_step.py` writes such
   a device.
2. Open the STEP file in the Activation function tab and assign roles (Edit → Auto-assign guesses them
   from names like Membrane, Pusher, Tube, TubeFluid, InletFluid, OutletFluid).
3. **Flow connections** appear in the model tree wherever two fluid bodies touch (and where a dynamic
   fluid faces the outside). Click one to make it an **Opening** (no resistance), an **Orifice** (your
   equation) or **Closed** (Flow tab).
4. Set the Δp range and the gas (gas constant, temperature, atmospheric pressure, viscosity) in the
   **Study** tab and press **Simulate** (F5). At every Δp the structure and the flow are iterated until
   the pressures on the tube wall stop changing.
5. **Save design** (Ctrl+S) stores the roles, connections, settings and the results in one `.mad`
   file. Opening it shows the curves without simulating again; from Python,
   `ActivationDesign.load("valve.mad")` gives `.area(dp)`, `.mass_flow(dp)` and `.end_pressure(dp)`.

| Role | Simulated as |
|---|---|
| Membrane / Shell | As in the neuron space, clamped along its edges; Δp pushes it towards the tube; bonded to every free rigid body or solid its face touches |
| Channel (tube) | 3D solid (10-node tetrahedra, compressible neo-Hookean), fixed at its two end faces |
| Fluid, Constant pressure | A supply or sink held at a set pressure |
| Fluid, Dynamic pressure | Pressure follows from the flow. The fluid filling the tube is cut into **Segments** along it, each a flow resistance in series with its own pressure on the tube wall |
| Solid | Deformable 3D solid (e.g. a soft pusher); free, or fixed where it touches fixed rigid bodies; contact with the tube and other solids |
| Rigid body, Motion = Fixed | Fixed obstacle |
| Rigid body, Motion = Free | Moves and tilts as a rigid body (e.g. a rigid pusher); contact with the tube and solids |

Resistance equations give the pressure drop in Pa for a mass flow `mdot` (kg/s), in SI units, with the
variables `rho` (gas density at the mean pressure), `rho_up` (upstream density), `mu`, `A`, `P`
(perimeter), `h`, `w`, `L`, `Dh = 4A/P`, `p`, `p_up`, `p_down`. A segment uses the smallest section in
it; a connection uses its contact face. Defaults: laminar segments `32*mu*L*mdot/(rho*A*Dh**2)`, orifice
`(mdot/(0.61*A))**2/(2*rho_up)`. The gas density follows the ideal gas law. A closed tube keeps a small
area (about 2 % of A0) from the contact gap between its walls.

## Design studies with an agent

The **Design study** tab runs an AI design agent (Claude Code) on a study folder and follows it live:

1. **New study…**: choose an empty folder, write the brief (goal, free parameters and ranges, what to measure,
   budget) and add your starting models (`.mns`, `.mad`, `.mfn`); they are imported as designs.
2. **Start agent**: opens Windows Terminal in the study folder with Claude Code running. The first time,
   accept Claude Code's "trust this folder" question. The agent works only through the `mns` command
   line: its rules are in the study's `CLAUDE.md`, its manual in `MANUAL.md`, and `.claude/settings.json`
   blocks scripts and edits outside the study.
3. Watch the tab: the agent's current command and its reason, the design list, the event log, the
   deforming shape during solves, and every plot and picture. **Open in its space** opens a design in the
   neuron, activation or full-neuron tab.

At the end, `report/study.pdf` holds the overall findings, and `designs/<ID>_<name>/` holds each design:
`design.yaml` (parameters, CAD, roles), `model.step`, the simulator project, `results/`, `renders/` and
`report/design.pdf`.

The same command line works by hand (`<study>/.mns/bin/mns`, or `python -m mns_api` with this folder on
`PYTHONPATH`): `mns help` lists the commands, and `agent/MANUAL.md` documents them and the design file format.

## Layout

```
app/                    desktop application (PyQt5 via qtpy, PyVista viewport, gmsh CAD kernel)
  cad.py                STEP import, per-body surface meshing, mid-surface extraction
  project.py            roles, property schemas, solver settings, .mns project files
  builder.py            project + CAD -> membrane_sim.Environment, chamber/membrane coupling detection
  main_window.py        main window; panels.py, viewport.py, sweep.py, workers.py
  sweep_core.py         sweep, weights, equation fit and exports without Qt (shared with mns_api)
  study_window.py       Design study tab (follows a study folder; starts the agent)
mns_api/                headless API and the `mns` command line (design files -> CAD -> runs -> reports)
agent/                  the agent's CLAUDE.md, MANUAL.md, settings.json and brief template (copied into studies)
membrane_sim/           solver core (usable on its own, see examples/soft_neuron_3d.py)
  shell.py              large-strain membrane triangle + Morley bending (node coords + edge rotations)
  fluid.py              FluidVolume: chamber pressure P(dV); dV from the moving shells only
  contact.py            exact signed distance to closed triangle meshes, penalty contact
  solver.py             Newton-Raphson: consistent tangent, sparse LU + Woodbury, line search
  environment.py        Environment: build, solve (fresh or warm-started), reset
  characterise.py       input-path weights W_j = dV_j/(p_j - p_a), W(dp) polynomial fit, LaTeX neuron equation
examples/               soft neuron script and STEP generator
cad_models/             user CAD files
tests/                  analytical benchmarks, consistency checks, application pipeline tests
```

## Solver model

The full formulation (elements, pressure loads, contact, Newton globalisation, CAD pipeline and the
fitting of the neuron equation) is written up in [`docs/paper/membrane_neuron_simulator.pdf`](docs/paper/membrane_neuron_simulator.pdf)
(LaTeX source next to it). The summary below is the short version.

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
