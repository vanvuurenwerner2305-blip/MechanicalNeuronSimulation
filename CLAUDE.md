# CLAUDE.md

Guidance for Claude Code (or any coding agent) working in this repository. The user-facing documentation is
`README.md` and `docs/manual/user_manual.pdf` (LaTeX source next to it); read the README first.

## What this is

The Membrane Neuron Simulator computes the **static equilibrium** of fluid-driven soft devices: rubber membranes and
shells between fluid chambers, pushing on rigid stops, on each other and on soft tubes. It was built to study a
*fluidic mechanical neuron*: input pressures act through membranes on a sealed **pre-activation** chamber (pressure
p_a), and a valve (a membrane squeezing a soft tube carrying gas) turns p_a into an output, the **activation**.

The user draws the device in CAD, imports it as STEP (one named solid per body), gives every solid a **role** in the
GUI, and solves. There is no time integration: every state is found by Newton–Raphson with load stepping.

## Setup and commands

Windows, Python 3.10–3.12 (tested 3.11). Packages in `requirements.txt` (`install.bat` makes a venv in
`%LOCALAPPDATA%\MembraneNeuronSimulator\venv`; the long-path limit is why it is outside the repo). Run from the repo root:

```
python run_app.py [file.step | project.mns | design.mad | neuron.mfn]   # GUI (or "Membrane Neuron Simulator.bat")
python -m pytest tests                       # full suite, slow (~15 min: real solves)
python -m pytest tests/test_membrane_sim.py  # solver core only
python -m pytest tests/test_app.py -k name   # single test
python -m mns_api help                       # headless command line
python examples/make_neuron_step.py          # writes examples/soft_neuron.step (test neuron assembly)
python examples/make_activation_step.py      # writes a round-tube test valve STEP
```

Tests build their own CAD with the `examples/make_*_step.py` generators; `tests/conftest.py` holds a
session-scoped simulated valve fixture (`design`) shared by the activation/full-neuron tests.

## Architecture: three layers

```
membrane_sim/   solver core: no Qt, no CAD. Units are whatever the caller uses (the app uses mm, N, MPa; kPa in the UI)
app/            desktop application (PyQt5 via qtpy, PyVista, gmsh). Each space = a Qt-free model + a window
mns_api/        headless API + `mns` CLI over the same app models (used by scripts and the design agent)
agent/          files copied into an agent study folder (AGENT_RULES.md becomes the study's CLAUDE.md)
```

Dependencies point downwards only: `mns_api` -> `app` (Qt-free modules) -> `membrane_sim`. Keep the computational
parts of `app/` importable without a display (`builder`, `project`, `cad`, `activation`, `full_neuron`, `analysis`,
`network`, `sweep_core`, `meshview`); the `*_window.py`, `panels`, `viewport`, `spaces`, `workers` modules are the GUI.

### membrane_sim (solver core)

- `shell.py`: 3-node triangles. Large-strain membrane (incompressible neo-Hookean or St Venant–Kirchhoff, optional
  pre-tension) plus Morley bending (one mid-edge rotation dof per edge). Membrane terms have hand-coded gradients and
  Hessians (`membrane_terms`); bending uses `torch.func`. Tests compare the hand-coded terms with torch to 1e-10.
- `solid.py`: TET4/TET10 compressible neo-Hookean solids (the valve's tube and pusher).
- `fluid.py`: `FluidVolume`, chamber pressure as a function of the volume change only. The volume change comes from
  the moving shells (cone volumes about a point; valid because shell boundaries are fixed), so chamber walls are
  never meshed. Laws: constant pressure (input), linear/incompressible, isothermal ideal gas, or a custom law.
- `contact.py` (rigid obstacles as closed triangle meshes, exact signed distance, penalty), `shell_contact.py`
  (sheet against sheet), `solid_contact.py` (bonds and contact with solids), `rigid.py` (free 6-dof rigid bodies
  and their couplings). Rigid bodies that overlap a soft body at rest are "inserted" (pushed in during a set-up stage).
- `flow.py` + `lumen.py`: steady gas flow through a chain of resistances, and the deforming tube cross-section.
- `empirical.py`: a membrane replaced by its pre-simulated response curve (used to link a valve into a neuron).
- `solver.py`: `NewtonSolver`, total energy + residual + sparse tangent; closed chambers add rank-1 terms handled
  with Sherman–Morrison–Woodbury. Energy-Armijo line search (energy is the only merit), LM damping, an Adam fallback,
  a gradient step as last resort.
- `environment.py`: `Environment` collects shells, solids, obstacles and fluids; `solve()` ramps the load
  (with step cutting) and supports warm starts (sweeps). `history` keeps every converged load step.

### app (spaces)

`spaces.py` is the top-level window: one tab per space, each with its own CAD model and project.

| Space (tab) | Model (Qt-free) | Window | File |
|---|---|---|---|
| Inputs → pre-activation (the neuron) | `project.py`, `builder.py`, `sweep_core.py` | `main_window.py` (+ `panels`, `viewport`, `sweep`) | `.mns` |
| Pre-activation → activation (the valve) | `activation.py` | `activation_window.py` | `.mad` |
| Full neuron (both linked, characterisation grids) | `full_neuron.py` | `full_neuron_window.py` | `.mfn` |
| Analysis (plots a stored characterisation) | `analysis.py` | `analysis_window.py`, `surface_view.py` | reads `.mfn` |
| Network (characterised neurons as blocks) | `network.py` | `network_window.py` | `.mnn` |
| Agentic research study | `mns_api` | `study_window.py` | study folder |

- `cad.py`: STEP import through gmsh/OpenCASCADE, per-body surface meshes, mid-surface extraction of thin bodies.
- `project.py`: roles (`Membrane`, `Shell`, `Rigid body`, `Fluid chamber`, `Ignore`, and in the valve space
  `Channel (tube)`, `Fluid`, `Solid`), their property schemas and defaults, solver settings, JSON save/load with
  migration of old files. Auto-assign guesses roles from body names; it is a convenience, often wrong.
- `builder.py`: CAD + `Project` -> `membrane_sim.Environment` (which chamber loads which membrane, clamped edges,
  obstacles). `activation.py` does the same for the valve and iterates structure <-> gas flow at every Δp.
- `full_neuron.py`: embeds a `.mns`, references a `.mad`, replaces one neuron membrane by the valve's empirical
  membrane, and characterises the neuron over a grid of swept parameters (stored in the `.mfn`). It tracks whether
  its components are out of date.
- `workers.py`: background `QThread` jobs. Job functions report through `worker.check() / report() / log() /
  item.emit()`; `mns_api/headless.py` provides the same interface without Qt.

All project/design files are JSON. Results stored in `.mad`/`.mfn` let a design be used as a part without
simulating it again.

### mns_api (headless API, CLI, agent studies)

`cli.py` is the `mns` command: every command prints one JSON object (`{"ok": true, ...}` or an error with a hint).
A **study** folder holds `designs/<ID>_<name>/design.yaml`; `cadspec.py` builds the STEP from the design's
primitives and parameters, `model.py` turns it into a `.mns/.mad/.mfn`, `ops.py` runs check/solve/sweep/characterise,
`report.py` writes LaTeX reports, `events.py`/`jobs.py` feed the GUI's live study view and background jobs.
`workspace.py` creates a study folder for the design agent from `agent/` (rules, manual, permissions, brief template).

## Conventions

- Units: mm, N, MPa in the models (`project.py`); pressures are entered and shown in kPa.
- Naming: the neuron's sealed chamber and its pressure are the **pre-activation** (p_a) in user-facing text; some
  code identifiers still say `activation` for it (`activation_index`, `solve_activation`). The valve output is the
  **activation**.
- British spelling in user-facing text and docs.
- Comments and docstrings are sparse and explain *why*; module docstrings describe the model. Match that style.
- When behaviour, defaults or UI labels change, update `README.md` and `docs/manual/user_manual.tex` (rebuild the
  PDF: `pdflatex` twice in `docs/manual`) in the same change.
- `cad_models/` and `docs/paper/` are git-ignored (local test models and an unpublished paper).
