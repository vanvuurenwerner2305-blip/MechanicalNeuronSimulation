# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Desktop simulation software for a Master's project on fluid-driven "mechanical neurons": rubber
membranes separating fluid chambers, deflecting against rigid obstacles. The user builds the device
in CAD (Fusion 360), exports STEP, assigns a role to every solid in the GUI, and solves for **static
equilibrium with Newton–Raphson**. It replaced the explicit-dynamics 2D code in `../Sim_Build_v2/`
(`simulation.py` + notebooks — legacy context only, not maintained).

GitHub remote `origin` = https://github.com/vanvuurenwerner2305-blip/MechanicalNeuronSimulation.git
(pushed on the user's request, last 2026-09-29 — ask before pushing again). The global git user.name is the placeholder
"Your Name"; commits use the configured identity. Files in `cad_models/` are the user's — do not
commit/modify them unless asked.

## Commands

```
python run_app.py [file.step | project.mns | design.mad]   # GUI (or double-click "Membrane Neuron Simulator.bat" / desktop shortcut)
python -m pytest tests                          # ~70 tests, ~10 min (solver benchmarks + app pipeline + activation)
python -m pytest tests/test_app.py -k ghost     # single test
python examples/make_neuron_step.py             # writes examples/soft_neuron.step (named test assembly)
python examples/make_activation_step.py         # writes examples/squeeze_valve.step (round-tube valve: TubeFluid, InletFluid, OutletFluid)
python examples/soft_neuron_3d.py               # solver core used from a script
```

Environment: Anaconda Python 3.11 at `C:\Users\werne\anaconda3` (torch 2.5, scipy, pyvista, pyvistaqt,
PyQt5, matplotlib, `gmsh` pip-installed). Windows.

## Architecture

`membrane_sim/` — solver core, usable without the GUI (units are whatever the caller uses; the app uses mm, N, MPa):
- `shell.py` — triangles: large-strain membrane (incompressible neo-Hookean or SVK) + Morley bending
  (one mid-edge rotation dof per edge; the naive "average face normal" hinge model was mesh-orientation
  dependent and was replaced). Energies per element; gradients/Hessians via `torch.func` (exact tangent).
- `fluid.py` — `FluidVolume`: pressure as a function of ΔV only. ΔV comes from the moving shells alone
  (cone volumes; valid because shell boundary nodes are pinned), so chamber walls are never meshed.
  Laws: linear `P0 - K dV`, native ideal gas (`gas_volume`, isothermal, infinite energy if gas → 0),
  or a torch `pressure_law(dV, P0)`. `load_state()` = load path (fresh ramp λ·P, or warm-start blend).
- `contact.py` — rigid obstacles = closed outward meshes; exact point-triangle distance with bbox broad
  phase, sign from angle-weighted pseudo-normals (exact for closed meshes). Nodal penalty
  `½ k A_node gap²`, per-shell offset (half thickness), reduced where the rest state is already closer.
  Free nodes next to a fixed node that start within 1.25x their offset of an obstacle (the clamp wall)
  are excluded from rigid contact (`NewtonSolver.__init__`): they sat on the penalty's on/off kink and
  made Newton cycle at ~1% residual (NeuronTest.mns never passed 0.5 kPa; now 20 kPa in 35 s).
  Contact queries use exact candidate (Verlet) lists: `CachedSignedDistance` (rigid, per shell in the
  solver, skin = max(reach, 1% model size)) and `ShellContact._pairs_within` (sheet pairs, skin = h);
  `safe_step` skips pairs whose bbox lower bound cannot limit the step. NeuronTest 29.8 s -> 19.5 s,
  identical iterations; profile now led by `shell.element_terms` (torch.func, ~40%).
  Speed notes: ~3.4k dofs, so CUDA (launch overhead) and a C++ rewrite are not worth it; next wins
  are hand-coded CST neo-Hookean gradient/Hessian and letting load increments grow past 1/load_steps.
- `solver.py` — `NewtonSolver.evaluate()` assembles energy, residual, sparse tangent; closed chambers add
  dense rank-1 terms `c g gᵀ` handled by Sherman–Morrison–Woodbury (`_Tangent`). `newton()`: Newton step
  with **second-order volume correction** (SOC) → LM-damped steps → Jacobi-preconditioned gradient step;
  energy-Armijo line search with quadratic interpolation; step capped at 10% of model size.
  **Energy is the only line-search merit** — mixing in residual decrease caused cycling (contact active set).
- `environment.py` — `Environment.solve()`: load stepping λ 0→1 with cutting (min 1e-3), one long
  "relaxation" Newton attempt when stalled (snap-through), `warm_start=True` ramps only the P0 change
  (used by sweeps; 3-10x faster). `history` holds every converged load step. `env.solver` keeps the
  last `NewtonSolver` for post-processing.
- `characterise.py` — back-inferring the neuron equation up to p_a. **W belongs to an input path, not a
  membrane** (user's definition, per the Article 2 chapter): W_j = dV_j/(p_j − p_a), dV_j = volume the
  path's shells push into the activation chamber. `input_paths` walks from each activation-chamber shell
  through closed intermediate chambers (`FluidVolume.is_closed`, e.g. a bulk-modulus weight chamber) to
  the constant-pressure input(s); no chamber on a side = "ambient" (0 kPa). Activation fluid adds
  W_0 = −dV_a/(p_a − p_0), p_0 = its law at rest volume. W_tan: unit load on the input with p_a fixed,
  other closed chambers keep their law (Woodbury `_Tangent` without the activation column). Only W is
  identified (user agreed A and K can not be separated). `fit_neuron_equation`: **the tolerance is on the
  equation's output p_a (absolute, default 1 kPa), not on W** (user's requirement: low-degree fits).
  **Weights are piecewise** (user, 2026-09-29): one polynomial for Δp > 0 and one for Δp < 0, each side a
  separate "piece" with its own degree in the search; fits[k] = {"kind": "piecewise", "sides": {"+", "-"}}
  (None = side not sampled, then the other side's polynomial is used for both, `weight_coefficients`).
  `polyfit_weight` weights the least squares by the sensitivity |∂p_a/∂W| = |Δp|/ΣW (Eq. 4.10,
  `activation_sensitivities`, ΣW over that sample's weights incl. W_0; user's request 2026-09-29, after an
  earlier attempt was rolled back). Without sensitivities it falls back to |Δp| (= error in ΔV). Unweighted, the
  huge W = ΔV/Δp just off Δp=0 (slack membrane) dominated and forced degree 4 with 1e6 coefficients. The
  W-vs-Δp plot (`_plot_fit`) colours each dot by |∂p_a/∂W| (`weight_sensitivity` in `app/sweep.py`).
  A sample whose ΣW cancels (≤ 1e-3 Σ|W|) gets NaN sensitivity and is left out of the fits: a gas weight chamber
  pre-pressurised inside a path is a hidden bias (NeuronTest2.mns, Weight1 at 20 kPa: at Input1 = 0 all pressures
  are 0 but p_a = 0.035, so W = -152.6 + 109.7 + 42.6 + 0.3 ≈ 0 and S → 2e10).
  **Bias term B** (user, 2026-09-29): p_a = (ΣW_j p_j + W_0 p_0 + B)/(ΣW_j + W_0). A path through a closed chamber that
  is not neutral at rest (`is_neutral`: pressure(0) ≠ 0) is biased: dV_j = b_j + W_j Δp_j. b_j is **measured, not
  fitted** (fitted, B wandered 0.6..18 mm³ with the degrees): `measure_bias` in app/sweep.py does one extra solve with
  every CONSTANT input (not vents) at the activation rest pressure, `bias_volumes` = dV - W_tan·Δp there (NeuronTest2
  with Weight1 at 20 kPa: b = 5.884). Rows carry row["bias"]; `sample_weight` gives (dV - b)/Δp.
  **Multivalued equations**: polynomial W can give several p_a roots; `solve_activation(all_roots=True)`; the fit
  scores each sample by its *worst* root ("ambiguous" count in the result) - picking the nearest root had accepted an
  equation 1.2 kPa off by another root. "Equation vs simulation" tab (`_plot_compare`): simulated p_a with the
  equation over it (1D curve + other roots as x; 2D surface + 25×25 wireframe).
  (Open issue found earlier: with a near-rigid activation fluid (W0 ≪ W) p_a depends
  only on weight *ratios*, so the p_a tolerance lets every W be off by the same factor — e.g. constant W
  fits 40-60% off the sampled W still met 1 kPa.) The 2D example
  sweep (Left, Right 0..20 kPa) is all constants per side at 1 kPa (W_Left 956 / 2652, W_Right 1752 / 4324
  mm³/kPa for Δp>0 / Δp<0, 0.41 kPa error). LaTeX: `neuron_equation_latex` returns pieces →
  `equation_align` (cases) / `equation_lines` (mathtext has no cases). Dialog fits in a `Worker` thread
  (`fit_worker`; exhaustive 715 combos ≈ 17 s at 0.3 kPa on 25 points).
  All pieces start at degree 0. `LOWEST_TOTAL` (default): exhaustive over degree combinations by total order,
  first total that meets the tolerance wins (least error among ties) — guaranteed minimal.
  `BIGGEST_ERROR` (user's earlier rule): +1 degree to the weight with the biggest *own* error (p_a solved
  with only that weight fitted, others at sampled W); can end higher because errors cancel (example
  neuron at 1 kPa: exhaustive total 1 vs greedy 3). Degrees capped at the first exact fit / points−1.
  Dialog: tolerance or method change does not refit; "Regenerate equation" refits from the stored rows
  (no re-simulation). Equation tab (redesigned 2026-09-29, user found it unreadable): verdict banner, then a
  scroll area with each line rendered at natural size (`math_pixmap` via mathtext) and one clickable `WeightCard` per
  weight (piecewise sides with a painted `Brace`), next to the selected weight's fit plot; LaTeX source in its own tab.
  The card's fit plot shows its fit
  over the sampled points (Δp≈0 points shown as dotted lines). p_a per point from `solve_activation` (implicit root of
  Σ W_k(x_k)(p_k − p_a) = 0 between min/max p_k). `polyfit_weight`: sensitivity-weighted least squares, Δp≈0 points left out.
  `neuron_equation_latex` → LaTeX lines (matplotlib-mathtext compatible). `app/sweep.py`: activation
  chamber combo, "p_a from weights" check column, "Neuron equation" tab (rendered + Copy LaTeX),
  export `<name>.csv`, `_weights.csv`, `_equation.tex`.

**Two spaces** (user, 2026-09-29): the window (`app/spaces.py` `AppWindow`) has two tabs, each a full embedded
QMainWindow with its own CAD model/project: **Neuron** (`MainWindow`, everything above) and **Activation function**
(`app/activation_window.py` `ActivationWindow(MainWindow)`, overriding role list / project class / results panel /
jobs via class attributes and hooks `_make_results_panel`, `_extra_tabs`, `_after_load`). The hidden tab's shortcuts
don't fire (hidden widgets). gmsh's single global model: `CadModel._ensure_active()` re-imports its own STEP when the
other space loaded one since (module global `_ACTIVE`).

Activation-function space (user's workflow: import CAD → simulate → save as a design that is used as a part,
no re-simulation): a membrane squeezes a soft **tube** against a rigid body through a part bonded to it, and gas
flows through the tube. The user's goal (2026-09-29): see **how the pressure divides along the tube depending on how
far it is clamped** - a lumped flow network (pressure divider), not a guessed p_out = f(A) formula (that first
version, with "Channel input/output side" roles, was replaced; `ActivationProject.load` migrates old files and
drops their results).
Roles (`ACTIVATION_ROLES`): Membrane/Shell, `CHANNEL` (tube, TET10 solid fixed at its planar end faces normal to
the axis), `FLUID` (**Constant pressure** = supply/sink, or **Dynamic pressure**; the dynamic fluid lining the tube is
cut into `segments` (default 10) along the tube, each a series resistor with its own wall pressure), `SOLID`
(deformable part, free or fixed where it touches fixed rigid bodies), Rigid body with **Motion: Fixed / Free**
(activation space only: `ACTIVATION_ROLE_FIELDS`, `Project.role_fields`, `Project.set_role` fills the space's
defaults). **No "Pusher" role** (user: rigid body with fix/free + a solid option instead); auto-assign makes
pusher/plunger/piston free rigid bodies, fluid-ish names Fluid (inlet/input/source/outlet/sink/ambient → constant).
**Flow connections** (`detect_connections`): wherever two fluid bodies touch (probe just outside each fluid face,
winding numbers) and where a dynamic fluid faces the outside (faces that probe outside are re-probed further out -
faceting gaps between the curved tube wall and the curved fluid faked an "outside" strip). Stored in
`project.connections[key]` = {"type": Opening | Orifice | Closed, "law"}; default opening between fluids, closed
to the outside. Shown under "Flow connections" in the model tree (items with negative indices -1-k,
`ModelTree.set_connections`) and in the **Flow** tab (`FlowPanel`); selecting one highlights the contact face.
**Flow network** (`membrane_sim/flow.py`, SI units): nodes (fixed or unknown gauge pressure), edges with the user's
law dp = f(mdot, rho, mu, A, P, h, w, L, Dh, p, p_up, p_down, rho_up) inverted per edge (brentq on |mdot|), openings
merge nodes (union-find), steady mass balance solved with `scipy.optimize.root`. Ideal gas ρ = p_abs/(R T); R, T,
p_atm, μ are Study settings. `rho` = mean pressure (friction), `rho_up` = upstream (orifice; default orifice law
`(mdot/(0.61*A))**2/(2*rho_up)`, default segment law laminar `32*mu*L*mdot/(rho*A*Dh**2)`). A connection's A, P, h, w
= its contact face (for the user's CAD that is the whole bore - a smaller orifice needs its area typed in the law).
Segment geometry = the smallest section in the segment (`ChannelSections.geometry`: A, perimeter, height along the
closing direction, width). Sections cut the **whole inside wall** (every tube face against any fluid, not end
faces): cutting only the dynamic fluid's faces lost the triangles at the junction with the next fluid (a fake 0.37
mm² dip). Axis = long direction of the tube fluid, oriented from the higher-pressure end (constant fluids and
outside openings count as ends).
`run_study`: per Δp, iterate structure ↔ flow: set wall pressures (segment k = mean of its end nodes; other fluids
against the tube get theirs) → warm-started solve (1-2 load steps, `min_load_increment` 1/64) → measure → network →
new wall pressures, under-relaxed when the change grows, until max change ≤ `coupling_tolerance` (kPa). Results:
A (min section), mdot, node pressures along the tube, p_end, travel, profiles; `ActivationDesign.load(path)`
.area/.mass_flow/.end_pressure(dp). Results tab: A & ṁ vs Δp, **tube-end pressures vs Δp** (user asked), and the
area + pressure profile along the tube at the selected point. The live plot follows the newest point
(`add_point`) - the slider kept an index of the previous longer run and raised IndexError on every point.
A membrane is bonded to every free rigid body (`RigidTie`) or Solid (`SurfaceTie`) its face touches (within 0.6 t);
Δp pushes it towards the tube; the closing direction comes from the membranes' area-weighted centroid (a vertex mean
tilted it 1%). Contact: every deformable solid (tube, Solids) vs every free rigid body (`MovingContact`) and vs
each other (`SolidContact`, both directions). `PartSettings.set_role` no longer carries `elements_per_side` across
kinds (a rigid body's 10 gave a Solid pusher 15.8k nodes and minutes per iteration). Core pieces:
- `membrane_sim/solid.py` `Solid`: TET4/TET10 compressible neo-Hookean, 4 Gauss points; shell-like interface
  (`faces` = boundary sub-triangles, 4 per TET10 face; `surface_nodes`; `allow_initial_overlap` → rest contact
  offset may be negative). **J via triple product**: `torch.linalg.det`'s Hessian is NaN at F = I (made the
  first Newton step fail).
- `membrane_sim/rigid.py`: `RigidBody` (6 dofs t, θ; Rodrigues with Taylor branch; `n_nodes = 0`, `coord_mask`),
  `RigidTie` (penalty 10 k), `MovingContact` (solid nodes vs the body in its frame; 2nd-order surrogate of the sd
  gives exact derivatives w.r.t. node and body).
- `membrane_sim/solid_contact.py`: `SolidContact` (nodes of A vs closest triangle of solid B, signed by the
  triangle normal, offset = min(0, rest sd) so touching at rest is neutral; C0 where the closest triangle changes)
  and `SurfaceTie` (membrane node follows a barycentric point of a solid's surface + non-rotating rest offset).
- `fluid.py` `SurfacePatch` / `FluidVolume.add_patch`: chamber wall on a solid, every boundary loop with a free
  node closed by a fan to its centroid (apex-independent volume; cap force = axial force of the pressure drop).
  `FluidVolume.volume_terms()` is what the solver now assembles (shell boundaries + patches). The tube's wall
  pressures are constant-pressure FluidVolumes (one per segment / per fluid) whose P0 the flow iteration updates.
- `shell_contact.py` `ShellContact(pairs=[(A, nodes, B, faces, h)])`: explicit node/face sets, used for the tube
  closing on itself (membrane-side wall vs far wall, h = 0.02 H). `safe_step` skips a node's own triangles when
  A is B — nodes on the seam have distance 0 and otherwise allowed no step at all.
- `solver.py`: bodies = Shell | Solid | RigidBody; `couplings` (bind/terms → global dofs, e, g, H) and
  `surface_contacts` lists (also on `Environment`).
- `lumen.py` `ChannelSections`: rest-configuration planes normal to the axis (a vertex on a plane counts as
  above), segments oriented by the wall normal, A = ½ Σ (p × q)·a (no ordering needed), clipped at 0.
Contact stiffness for the device: k = 4 p_ref / (0.02 H), H = inside height of the tube along the closing direction.
User's CAD (`cad_models/ActivationTest.step`, rebuilt 2026-09-29 22:01): lens-shaped Tube z -5..5 with TubeCavity
(dynamic fluid), End1/End2 (fixed rigid tube stubs z -7..-5 / 5..7) with fluids Input and Body10 inside them
(Body10 auto-assigns as a rigid body - the user sets it to a constant fluid). Pressure check with Input 10 kPa,
Body10 0 kPa, orifice TubeCavity→Body10 with A = whole bore: solver mdot = 6.5558e-5 kg/s = hand orifice formula;
the orifice takes ~9.1 kPa (jet ≈73 m/s; laminar segment law is really Re ≈ 3500). The user's own run used a
0.0001·A orifice, 500x smaller than even the closed tube (floor ≈0.012-0.036 mm²: contact gap + lens corners), so
the tube never controlled the flow - told them to size the orifice between open- and closed-tube resistance.
Earlier numbers (old model, rigid pusher, p_in 10 kPa in the input half): A = 0.33 at 10 kPa, closed from ≈20 kPa,
12-25 s per point; Solid pusher (E = 5 MPa): A = 0.367 at 10 kPa, 70-150 s per point. The user saved designs into
the session scratchpad (temporary) because QSettings' last_dir came from a GUI test - told them to save elsewhere.

`app/` — GUI (PyQt5 via qtpy + PyVista) on top of the core:
- `cad.py` — gmsh/OpenCASCADE STEP import (names from product labels, fallback to STEP solid names for
  Fusion multi-body parts), per-body surface meshing, topological outward orientation (handles cavities),
  mid-surface of thin solids = largest CAD face moved half the ray-cast thickness inward.
- `project.py` — roles (Membrane = no bending, Shell = bending, Rigid body, Fluid chamber, Ignore),
  property schemas (`Field` descriptors drive the forms), chamber models, colours, `.mns` JSON save/load
  (with legacy migration), auto-assign roles from names.
- `builder.py` — project + CAD → `Environment`. Chamber↔membrane coupling detected by probing just beyond
  each mid-surface face with the chamber mesh's winding number. Mesh size = shortest side / elements per
  side (default 10). Auto contact stiffness = P_ref / (5% of thinnest thickness), set at **solve time**
  (sweeps: from the sweep's max pressure).
- `main_window.py`, `panels.py`, `viewport.py`, `sweep.py`, `workers.py` — UI. Long jobs run in `Worker`
  QThreads; picking is our own ray cast (click again = next part behind); transparency slider.

Domain conventions: all pressures **gauge** (surroundings = 0 kPa; a membrane face touching no chamber
sees 0). Chamber models: Constant pressure (inputs, green), Closed ideal gas with incompressible % and
ghost volume (blue, darker = more liquid), Closed incompressible = linear law with stiffness in kPa per
% ΔV (purple, default 10 at user's request; ≥1000 behaves like water's 22000 but solves slower),
Vent (0 kPa, transparent). Membrane default E = 0.5 MPa.
Thickness is measured from CAD when a role is assigned.

`docs/paper/membrane_neuron_simulator.tex` (+ compiled PDF) — technical reference of the whole formulation,
linked from the thesis instead of describing the simulator there (user, 2026-09-29). **Keep it in sync when the
math changes.** Build: `pdflatex` twice in `docs/paper` (MiKTeX installed; aux files are git-ignored).

## Gotchas (learned the hard way)

- **Sweep dialog threads (fixed 2026-09-29):** the user's "crashes" were Windows app hangs (Event Viewer: AppHangB1,
  pythonw) - closing the dialog blocked the UI in `wait(60000)` on an equation fit that could not be cancelled and,
  at an unreachable tolerance, tried every degree combination (hours). Now: `fit_neuron_equation(check=, 
  max_evaluations=2000)` ("stopped" in the result), closing cancels and closes on `finished` (never waits on the
  UI thread; Esc goes through `reject`, which skips `closeEvent`), a new sweep cancels a running fit, and Export no
  longer fits on the UI thread. Crash forensics: `Get-WinEvent` Application log IDs 1000/1001/1002 - fast-fail
  aborts (0xc0000409) and hangs never reach faulthandler/crash.log.

- Only **PyQt5** works in this Anaconda env (PySide6/PyQt6 DLL conflicts); `app/__init__.py` sets `QT_API=pyqt5`.
  Qt6-only imports (e.g. `QAction` from QtGui) need a fallback.
- PyQt5 aborts on unhandled exceptions in slots; `app/__main__.py` installs an excepthook + faulthandler.
  Logs: `~/.membrane_neuron_simulator/errors.log` and `crash.log` (the app runs under pythonw, no console).
- Never delete a widget inside its own signal (crashed on dropdown change): forms use `deleteLater` and
  rebuilds are deferred with `QTimer.singleShot(0, ...)`.
- gmsh holds one global model: only the last loaded `CadModel` can be meshed; `gmsh.initialize` must run
  on the main thread.
- GUI test scripts open real windows on the user's desktop (they may click them); VTK does not work with
  `QT_QPA_PLATFORM=offscreen` on Windows. Capture the 3D view with `viewport.screenshot()`, not `grab()`.
- Bash heredocs containing Python with quotes sometimes break the harness — write edit scripts to the
  scratchpad and run them. `Path.write_text` without `encoding="utf-8"` writes cp1252 on Windows (broke `mm²`). `cmd | grep` buffers output of long runs; write to a log file instead.
  TaskStop may leave the Python child running — check with `Get-CimInstance Win32_Process`.
  The user's Jupyter Lab processes are theirs; don't kill them.

## Earlier work (2026-09-28/29, committed in 2b5d242)

1. **Membrane–membrane contact** (`membrane_sim/shell_contact.py`, wired into `NewtonSolver.evaluate`):
   node-to-triangle penalty between *different* shells (no self-contact), contact distance
   h = t_A/2 + t_B/2 (reduced to the rest distance where sheets start closer), C2-smoothed penalty
   (ramp over 0.25 h), **all triangles within range** contribute (closest-only made Newton cycle when a
   node slid across an edge), exact per-pair gradient/Hessian via torch.func. Anti-tunnelling:
   `ShellContact.safe_step()` bounds each line-search step so no node-triangle pair closes >90% of its
   distance (IPC-style conservative CCD). An earlier "recorded side" penetration check gave false
   positives on curved sheets and was removed.
   Also added in `newton()`: stagnation acceptance (residual stalled for 3 its and within 1000x tol),
   because contact energies are only C1 at triangle edges.
   Tests (`tests/test_membrane_sim.py -k sheet`): FD consistency ✓, inflating sheet lifts upper sheet
   without crossing ✓ (21 s), **`test_sheets_do_not_cross_under_a_hard_push` FAILS** — stalls at load
   ≈0.29 (large deformation, k=50). Was about to trace whether `safe_step` or the line search limits
   the steps there. Contact not yet tried on the user's models (`cad_models/Example.mns`, `NeuronTest.mns`).
2. **Defaults changed** (done, in `app/project.py`): incompressible chamber stiffness default
   `INCOMPRESSIBLE_STIFFNESS = 10` kPa/%; membrane Young's modulus default 0.5 MPa (Shell stays 1.0).
   `tests/test_app.py` incompressible test now sets stiffness=1000 explicitly.
3. **Fluid volume for "Closed: incompressible" chambers** (done 2026-09-28; `FluidVolume(liquid_volume=...)`,
   field `fluid_volume`, auto-filled with the body volume in `_detect_thickness`; 0 = body volume).
   The liquid's rest volume is the fluid volume: P = P0 − K (V − V_fluid), K = stiffness·100/V_fluid, so less
   liquid than the chamber gives suction (user's physics: sealed, nothing else in it), more liquid inflates it.
   (An earlier "empty slack, P stays P0" version was rejected by the user.)
   History step 0 ("0 (start)" in the results slider) records the full chamber pressures on the undeformed
   geometry (e.g. the initial suction), not the zero pressures of the load ramp (user asked for an iteration 0).

4. **Input-path weights + neuron equation in sweeps** (done 2026-09-28; `membrane_sim/characterise.py`,
   `app/sweep.py`; tests `-k "path or ambient or tangent_weight or solve_activation or equation_fit or usable or latex"` and
   test_app `-k weights` pass). Equation tolerance added 2026-09-29. Known: an unpretensioned membrane has W ∝ Δp^(−2/3) (singular at Δp=0),
   so tight tolerances drive W(Δp) to high degrees; at the default 1 kPa the example neuron gives
   W1 constant, W2 linear, W0 constant (0.64 kPa error) with the exhaustive search. Δp=0 points are left out (user's choice).

## Status and open issues

- **Deformable (Solid) pusher does not converge on the generated round-tube valve** (`SolidContact`: residual
  stalls at load ≈0.2 of 30 kPa, also with a finer pusher mesh; test marked xfail). Works on the user's lens tube.
  Likely the closest-triangle-only pairing (C0); the fix to try: all triangles within range with a smoothed
  penalty, as `ShellContact` does.
- Flow model limits: steady, isothermal, lumped (uniform pressure per segment); no choking/compressible orifice law
  unless the user writes one; the closed tube keeps ≈2% of A0 (contact gap h = 0.02 H - could be a setting).

- The new robust `newton()` (commit 7540c3e) solves `cad_models/Example.mns` up to 19 kPa, but
  **regressed on the soft-neuron benchmark** (old: converged 110 its / 32 s; new: stalls at λ≈0.995 after
  6 min). Needs investigation (likely SOC/line-search interplay with contact) — or `git revert 7540c3e`.
- `Example.mns` cannot reach 20 kPa for a physical reason: the side window Body13 (5 mm, E=0.1 MPa)
  balloons past its limit point at ~9 kPa (Body6 pressure then *drops*), and plates Body2/Body3 pass
  through each other beyond ~16 kPa.
- Missing: self-contact of a sheet; a warning when stretches exceed material limits; a mixed
  (pressure-unknown) formulation would make stiff chambers exact at any stiffness.
