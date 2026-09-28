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
- `solver.py` — `NewtonSolver.evaluate()` assembles energy, residual, sparse tangent; closed chambers add
  dense rank-1 terms `c g gᵀ` handled by Sherman–Morrison–Woodbury (`_Tangent`). `newton()`: Newton step
  with **second-order volume correction** (SOC) → LM-damped steps → Jacobi-preconditioned gradient step;
  energy-Armijo line search with quadratic interpolation; step capped at 10% of model size.
  **Energy is the only line-search merit** — mixing in residual decrease caused cycling (contact active set).
- `environment.py` — `Environment.solve()`: load stepping λ 0→1 with cutting (min 1e-3), one long
  "relaxation" Newton attempt when stalled (snap-through), `warm_start=True` ramps only the P0 change
  (used by sweeps; 3-10x faster). `history` holds every converged load step.

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
% ΔV (purple, default 1000 — results identical to water's 22000 but far faster), Vent (0 kPa, transparent).
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

## Status and open issues (as of 2026-09-28)

- The new robust `newton()` (commit 7540c3e) solves `cad_models/Example.mns` up to 19 kPa, but
  **regressed on the soft-neuron benchmark** (old: converged 110 its / 32 s; new: stalls at λ≈0.995 after
  6 min). Needs investigation (likely SOC/line-search interplay with contact) — or `git revert 7540c3e`.
- `Example.mns` cannot reach 20 kPa for a physical reason: the side window Body13 (5 mm, E=0.1 MPa)
  balloons past its limit point at ~9 kPa (Body6 pressure then *drops*), and plates Body2/Body3 pass
  through each other beyond ~16 kPa.
- Missing: membrane–membrane contact; a warning when stretches exceed material limits; a mixed
  (pressure-unknown) formulation would make stiff chambers exact at any stiffness.
