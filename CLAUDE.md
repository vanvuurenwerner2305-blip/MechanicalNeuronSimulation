# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Desktop simulation software for a Master's project on fluid-driven "mechanical neurons": rubber
membranes separating fluid chambers, deflecting against rigid obstacles. The user builds the device
in CAD (Fusion 360), exports STEP, assigns a role to every solid in the GUI, and solves for **static
equilibrium with Newton–Raphson**. It replaced the explicit-dynamics 2D code in `../Sim_Build_v2/`
(`simulation.py` + notebooks — legacy context only, not maintained).

GitHub remote `origin` = https://github.com/vanvuurenwerner2305-blip/MechanicalNeuronSimulation.git
(empty remote, **nothing pushed yet** — ask before pushing). The global git user.name is the placeholder
"Your Name"; commits use the configured identity. Files in `cad_models/` are the user's — do not
commit/modify them unless asked.

## Commands

```
python run_app.py [file.step | project.mns]      # GUI (or double-click "Membrane Neuron Simulator.bat" / desktop shortcut)
python -m pytest tests                          # ~35 tests, 2-7 min (solver benchmarks + app pipeline)
python -m pytest tests/test_app.py -k ghost     # single test
python examples/make_neuron_step.py             # writes examples/soft_neuron.step (named test assembly)
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
  `polyfit_weight` weights the least squares by |Δp| (= error in ΔV): unweighted, the huge W = ΔV/Δp just
  off Δp=0 (slack membrane) dominated and forced degree 4 with 1e6 coefficients. (A sensitivity weighting
  |Δp|/ΣW from Eq. 4.10, a W>0 check and sensitivity-coloured dots were tried 2026-09-29 and **rolled back
  at the user's request**. Open issue found then: with a near-rigid activation fluid (W0 ≪ W) p_a depends
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
  (no re-simulation). Clicking a weight line in the rendered equation (matplotlib pick) plots its fit
  over the sampled points below it (Δp≈0 points shown as dotted lines). p_a per point from `solve_activation` (implicit root of
  Σ W_k(x_k)(p_k − p_a) = 0 between min/max p_k). `polyfit_weight`: least squares, Δp≈0 points left out.
  `neuron_equation_latex` → LaTeX lines (matplotlib-mathtext compatible). `app/sweep.py`: activation
  chamber combo, "p_a from weights" check column, "Neuron equation" tab (rendered + Copy LaTeX),
  export `<name>.csv`, `_weights.csv`, `_equation.tex`.

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

## Gotchas (learned the hard way)

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
  scratchpad and run them. `cmd | grep` buffers output of long runs; write to a log file instead.
  TaskStop may leave the Python child running — check with `Get-CimInstance Win32_Process`.
  The user's Jupyter Lab processes are theirs; don't kill them.

## Work in progress (UNCOMMITTED, as of end of 2026-09-28 session)

Uncommitted edits in `membrane_sim/solver.py`, `membrane_sim/shell_contact.py` (new), `app/project.py`,
`README.md`, `tests/`. Commit once the tests below pass. (`cad_models/*` changes are the user's.)

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

- The new robust `newton()` (commit 7540c3e) solves `cad_models/Example.mns` up to 19 kPa, but
  **regressed on the soft-neuron benchmark** (old: converged 110 its / 32 s; new: stalls at λ≈0.995 after
  6 min). Needs investigation (likely SOC/line-search interplay with contact) — or `git revert 7540c3e`.
- `Example.mns` cannot reach 20 kPa for a physical reason: the side window Body13 (5 mm, E=0.1 MPa)
  balloons past its limit point at ~9 kPa (Body6 pressure then *drops*), and plates Body2/Body3 pass
  through each other beyond ~16 kPa.
- Missing: self-contact of a sheet; a warning when stretches exceed material limits; a mixed
  (pressure-unknown) formulation would make stiff chambers exact at any stiffness.
