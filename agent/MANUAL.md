# Membrane Neuron Simulator — manual for the design agent

You run design studies with the `mns` command line. This manual covers what the simulator models, how a study
folder is organised, every command, the design file (CAD + roles), proven workflows, what to do when something
fails, and how to write the reports. Read it once fully; later, look up single sections.

Contents: 1 Physics · 2 Study folder · 3 Commands · 4 Design files · 5 Workflows · 6 Speed and budget ·
7 When things fail · 8 Reports · 9 Saving tokens

---

## 1. What is simulated

**Mechanical neurons** made of soft rubber membranes between fluid chambers. Everything is quasi-static:
the solver finds the equilibrium shape (Newton–Raphson, load ramped from 0 to full in steps).

Units everywhere: **mm, MPa (moduli), kPa (pressures), mm³ (volumes)**. Pressures are **gauge**: the
surroundings are 0 kPa; a membrane face that touches no chamber sees 0 kPa.

The device is split into three *spaces*, each with its own design kind:

| Space | ID | What it is | Main result |
|---|---|---|---|
| neuron ("inputs → pre-activation") | N### | input chambers push on membranes into a closed **pre-activation chamber** | the pre-activation pressure p_a against the input pressures |
| activation ("pre-activation → activation") | A### | a membrane, driven by the pressure difference Δp across it, squeezes a soft **tube** through a pusher; gas flows through the tube | the **activation function**: tube area, mass flow and named output pressures against Δp |
| full | F### | an N design whose membrane is replaced by an A design (its pre-simulated Δp → swept-volume curve) | outputs of the whole neuron against its inputs (**characterisation**) |

### Neuron space roles
- **Membrane** — thin rubber sheet, no bending stiffness, simulated on its mid-surface (the CAD body is a thin
  plate; the thickness is measured from it). Edges are fixed (clamped): all boundary edges, or only edges
  touching rigid bodies. Materials: incompressible neo-Hookean (default, E = 0.5 MPa) or St. Venant–Kirchhoff.
  Optional pre-tension (N/mm).
- **Shell** — like a membrane but with bending stiffness (E default 1.0 MPa); edges clamped or pinned.
- **Rigid body** — undeformable obstacle (housing, stops). Membranes can not pass through it (contact).
- **Fluid chamber** — a fluid region; its pressure acts on every membrane/shell face it touches. Models:
  - `input` (Constant pressure): held at `pressure` (kPa). These are the neuron's inputs.
  - `liquid` (Closed: incompressible): sealed liquid, P = P0 − K·ΔV; `stiffness` in kPa per % volume change
    (default 10; water would be 22000, ≥1000 behaves as water but solves slower). `fluid_volume` (mm³, default
    the body volume): less liquid than the cavity gives suction.
  - `gas` (Closed: ideal gas, isothermal): sealed air at `pressure` when sealed; `ghost_volume` adds unmodelled
    volume (tubing); `incompressible` % of the volume is liquid (stiffer).
  - `vent`: open to the surroundings, 0 kPa.
- **Activation membrane** — a neuron membrane replaced by an activation design (`design: A###`); the chamber
  on its driving side (`driving`, default Automatic = the closed chamber it touches) pushes it towards the tube.
- **Ignore** — left out.

### Activation space roles
- **Membrane / Shell** — loaded by Δp (positive Δp pushes it towards the tube), clamped at its edges.
- **Channel (tube)** — the soft tube (3D solid, quadratic tetrahedra), fixed at its two planar end faces.
- **Fluid** — `constant` (a supply or sink at `pressure`) or `dynamic` (pressure from the flow). The dynamic
  fluid inside the tube is cut into `segments` along it (series flow resistances, each with its own wall
  pressure). Its `outputs` name segments whose gas pressure is an output: `[{name: activation, segment: 10}]`.
- **Solid** — deformable 3D part (e.g. a soft pusher), free (held by the bonded membrane) or fixed.
- **Rigid body** — `motion: fixed` (obstacle, e.g. the block the tube lies on) or `motion: free` (6 degrees of
  freedom, e.g. a rigid pusher). A membrane is **bonded** to every free rigid body or solid its face touches.
- **Flow connections** — wherever two fluids touch (and where a dynamic fluid faces the outside):
  `opening` (no resistance, default between fluids), `orifice` (law, default `(mdot/(0.61*A))**2/(2*rho_up)`
  with A = contact face area), or `closed` (default to the outside).
At every Δp the structure and the flow are iterated until the wall pressures agree. The result (tube area,
mass flow, output pressures, swept volume vs Δp) is stored in the design and is what a neuron uses.

### Full neuron
A neuron design with one membrane/shell replaced by an activation design (`link: {part: <membrane>}`). Only
chamber values can change here (pressures, stiffness, volumes; not models). `characterise` solves a grid of
chamber parameters and records quantities (keys `P:<chamber>`, `dV:<chamber>`, `dp` (pre-activation Δp),
`out:<output>`, `area`, `mdot`). If Δp leaves the activation design's simulated range the result is
**extrapolated** — re-run the activation design's `dpsweep` over a wider range.

---

## 2. The study folder

```
brief.md                 the user's goal — your assignment (read-only)
CLAUDE.md, MANUAL.md     your rules and this manual (read-only)
background/              the research article behind the study and how its terms map to mns (read-only)
notes.md                 your lab notebook (create and keep it current; see CLAUDE.md)
feature_requests.md      what the software could not do (append)
inputs/                  copies of the user's starting files
designs/<ID>_<name>/     one folder per design
    design.yaml          the design file (you write it; see section 4)
    state.json           status, run summaries, metrics (written by mns; read with mns design show)
    model.step           CAD built from design.yaml (opens in Fusion/any CAD)
    model.mns|.mad|.mfn  the simulator project (the user opens it in the GUI)
    preview.npz          meshes for the GUI's live view
    results/             csv, json, plots of every run
    renders/             pictures of the model and of deformed results
    report/              design.tex (generated), narrative.tex (yours), auto_*.tex (generated), design.pdf
report/                  study.tex (generated), narrative.tex (yours), auto_*.tex, figures/, study.pdf
.mns/                    events (the GUI follows them live), jobs, logs — never edit
```

Statuses: created → built → checked → solved / swept / studied / characterised → reported.
The user watches the GUI's Study tab: it shows each design as you build it, live progress and the deforming
shape during solves, and your `--why` reasons. Always give `--why` on commands that start work.

---

## 3. Commands

Every command prints **one JSON line** (`"ok": true/false`; on failure `error` and usually a `hint`). Paths in the
output are relative to the study folder. Options can go anywhere after the command. Values starting with `-`
need `=`: `--dp=-10:10:5`.

Ranges: `from:to:points` (e.g. `0:20:5` → 0, 5, 10, 15, 20), a list `0,5,12`, or one value.

| Command | Does |
|---|---|
| `mns status` | designs, running jobs, last events |
| `mns help [command]` · `mns help properties [role] [--space neuron\|activation]` | usage · every role's properties, defaults, units, choices |
| `mns templates` | starting design files |
| `mns design list` | all designs with status |
| `mns design show <ID> [--spec]` | parameters, runs, warnings, files (`--spec` adds design.yaml) |
| `mns design new <name> --space neuron\|activation\|full [--template T] --why "..."` | new design folder with a design.yaml to edit |
| `mns design derive <ID> <name> --set p=v [--set Body.prop=v] [--set solver.x=v] --why "..."` | copy with changes, then build (`--no-build` to skip) |
| `mns design import <file.mns\|.mad\|.mfn\|.step> [--name n]` | an existing project as design(s) |
| `mns cad build <ID>` | CAD + project from design.yaml; geometry check; renders/model.png |
| `mns cad inspect <ID>` | per body volume, bounding box, thin-plate thickness; overlaps; touching pairs |
| `mns cad render <ID> [--views iso,front,top,right,section-x:0.5] [--exploded 0.3] [--only A,B] [--name f]` | a picture of the bodies |
| `mns check <ID>` | mesh + assemble (no solve): which chamber loads which membrane and from which side, inputs, links, flow connections, bonds, warnings |
| `mns solve <ID> [--set Part.field=v]` | one static solve (neuron, full); `--set` only for this run |
| `mns sweep <ID> --input Ch=from:to:n [--input Ch2=...]` | neuron sweep over 1–2 inputs: chamber pressures and linked designs' outputs at every point |
| `mns dpsweep <ID> [--dp 0:30:16] [--set study.field=v]` | activation-function study over Δp (stores the results in model.mad) |
| `mns characterise <ID> --axis Part.field=from:to:n [--axis ...] [--record key,key]` | full-neuron grid (stored in model.mfn) |
| `mns compare [IDs] [--metrics a,b] [--x param --y metric]` | parameters and metrics of designs side by side; `--x/--y` plots one against the other into report/figures |
| `mns report design <ID> [--build]` · `mns report study [--build]` | generate the LaTeX parts; `--build` compiles the PDF |
| `mns note "text" [--design ID]` | message to the live view and log (milestones, decisions) |
| `mns jobs` · `mns wait <job> [--timeout 540]` · `mns cancel <job>` | background jobs |

Global options: `--why "reason"` (shown live to the user), `--background` (solve/sweep/dpsweep/characterise:
returns a job id at once), `--max-minutes N` (stop a run after N minutes), `--no-render` (skip pictures).

**Long runs.** Anything that may take more than ~90 s: start it with `--background`, then `mns wait <job>`
(blocks up to 540 s and returns the result, or the progress if still running — then call `mns wait` again).
Never poll with short loops. You may run two background jobs at once; more slows all of them.

---

## 4. Design files (design.yaml)

YAML. Keys written by mns at creation (`id`, `name`, `space`, `why`, `parent`, `created`) — leave them.

```yaml
why: "Thinner membranes for a stronger response to Input1"   # the question this design answers
parameters:            # numbers or expressions of parameters defined above them (mm, MPa, kPa)
  R: 8
  t: 0.5
  gap: "2*t + 0.25"    # quote expressions; allowed: + - * / ** % ( ), sqrt sin cos tan exp log min max abs pi
base_step: base.step   # optional: a STEP file in the design folder whose bodies `import` uses
bodies:
  - name: Membrane1    # unique; becomes the part name (use it in --set, --input, link, outputs)
    shape: {cylinder: {base: [0, 0, "-t/2"], axis: [0, 0, t], radius: R}}
    role: membrane     # see the role aliases: membrane shell rigid chamber ignore "activation membrane"
                       #   (activation space: membrane shell tube fluid solid rigid ignore)
    props: {youngs_modulus: 0.5, thickness: t, elements_per_side: 12}
solver: {load_steps: 10, max_iterations: 40}      # optional
# activation space only:
connections: {"OutletFluid <-> TubeFluid": {type: orifice, law: "(mdot/(0.61*0.05e-6))**2/(2*rho_up)"}}
study: {dp_min: 0, dp_max: 30, points: 16}
```

A body without `shape` (when `base_step` is set) is taken unchanged from base.step — imported designs look like
that. Replace its shape with `{import: Name}` plus modifiers to move/scale it, or with a new shape.

### Shapes (one kind per shape; every number may be an expression)

| Kind | Syntax |
|---|---|
| box | `{box: {min: [x,y,z], size: [dx,dy,dz]}}` or `{box: {center: [...], size: [...]}}` |
| cylinder | `{cylinder: {base: [x,y,z], axis: [ax,ay,az], radius: r}}` (axis vector = height + direction) |
| cone | `{cone: {base: [...], axis: [...], radius1: r1, radius2: r2}}` |
| sphere | `{sphere: {center: [...], radius: r}}` |
| torus | `{torus: {center: [...], radius: R, tube_radius: r, axis: [0,0,1]}}` |
| extrude | `{extrude: {sketch: SKETCH, distance: d, symmetric: false}}` (along the sketch plane normal) |
| revolve | `{revolve: {sketch: SKETCH, axis_point: [...], axis: [...], angle: 360}}` (degrees) |
| pipe | `{pipe: {path: [[x,y,z], ...], radius: r, inner_radius: 0, smooth: false}}` (hollow when inner_radius > 0) |
| union | `{union: [SHAPE, SHAPE, ...]}` |
| cut | `{cut: {from: SHAPE, remove: [SHAPE, ...]}}` |
| intersect | `{intersect: [SHAPE, SHAPE, ...]}` |
| ref | `{ref: BodyName}` — a copy of another body's shape |
| cavity | `{cavity: {inside: SHAPE, minus: [BodyName, ...]}}` — the region of SHAPE not taken by those bodies (fluids!) |
| import | `{import: BodyName}` — a body of base_step |

Modifiers on any shape (applied in this order): `scale: s` or `[sx,sy,sz]` (about `scale_origin`, default
origin), `rotate: {axis: [...], angle: deg, origin: [...]}` (or a list), `mirror: {normal: [...], origin: [...]}`,
`translate: [dx,dy,dz]`, `fillet: r` (rounds every edge; keep r below the thinnest wall).

SKETCH = `{plane: xy|yz|zx|xz | {origin: [...], normal: [...], u: [...]}, offset: d, <outline>, holes: [<outline>, ...]}`
with one outline: `circle: {center: [u,v], radius: r}` · `rectangle: {min: [u,v], size: [du,dv]}` (or `center`;
`corner_radius: r` rounds it) · `polygon: [[u,v], ...]` · `path: [[u,v], [u,v], {arc: {through: [u,v], to: [u,v]}},
{arc: {center: [u,v], to: [u,v]}}, {spline: [[u,v], ..., [u,v]]}]` (closed automatically). Plane axes:
xy → (u,v) = (x,y), normal +z; yz → (y,z), +x; zx → (z,x), +y; xz → (x,z), −y. `offset` moves along the normal.

### Modelling rules (the build and check tell you when you break them)
1. **Bodies touch, they never overlap.** A membrane and the chambers either side share faces; a chamber fills
   exactly the space between walls. Build fluids with `cavity` (region minus the solid bodies) or with exact
   coordinates. `cad build` reports every OVERLAP — fix all of them.
2. **Membranes/shells are thin plates** (thickness well below width). Their largest face defines the
   mid-surface. Edges are clamped where the boundary is (all boundary edges by default).
3. A **chamber must touch the membrane faces it loads**, over the whole face (coverage ~100% in `check`).
4. Every input path needs a chamber on each side of its membrane, or the far side sees 0 kPa (ambient).
5. Leave room: a membrane that bulges into a rigid wall is stopped by contact; that may be intended (a stop) but
   it changes W strongly.
6. `elements_per_side` (default 10, along the shortest in-plane side) sets the mesh; 8–14 is the useful range.
   Finer is slower; coarser is less accurate. Keep it the same across designs you compare.
7. Activation: the tube needs two planar end faces normal to its axis; the dynamic fluid fills the tube exactly;
   constant fluids touch the tube's ends; the membrane touches the pusher (bond).
8. Names: letters, digits, `_`; no spaces.

Properties: run `mns help properties <role> --space <space>` for the exact list. Choices accept short aliases
(`model: input|liquid|gas|vent`, `model: constant|dynamic`, `motion: fixed|free`, `material: neo-hookean|svk`).

Templates (`mns templates`): `neuron_basic` (two inputs, liquid pre-activation chamber, two disc membranes in a
housing), `activation_valve` (round tube, block, free rigid pusher under a membrane, inlet/outlet), `full`.

---

## 5. Workflows

**A. Start of a study**
1. `mns status`; read brief.md; `mns design show` each starting design (they were imported from the user's files).
2. Run the starting designs as they are (baseline): `mns check`, then the run that answers the brief (`sweep`,
   `dpsweep`, `characterise`). Record the baseline numbers in notes.md.
3. If the study varies geometry, rebuild the baseline as a **parametric** design (section 4): `mns cad inspect`
   the imported design for dimensions, write a new design.yaml with parameters, `mns cad build`, and confirm it
   matches the original (body volumes within ~1%, same touching pairs, same check couplings, and run results
   close to the baseline). Note the match in notes.md. Then derive variants from the parametric version.
   Property-only studies (pressures, stiffness, E, thickness value) don't need a rebuild: derive with
   `--set Body.prop=v`.

**B. One design step**
`mns design derive <parent> <name> --set ... --why "..."` (builds) → read the build output (overlaps,
warnings) → if the geometry changed, look at renders/model.png once → `mns check` (for new geometry) →
run → read the summary → update notes.md → `mns report design <ID>` and write its narrative.

**C. Neuron**: `mns sweep N### --input A=0:20:5 --input B=0:20:5` — about 5–15 s per point. Read the
closed chambers' pressure ranges and look at results/sweep_response.png; compare designs on how the
pre-activation pressure responds to each input (sweep.csv has every point).

**D. Activation**: `mns dpsweep A### --dp 0:30:16` — 10–40 s per point. Key metrics: `A0_mm2`,
`half_area_dp_kPa`, `closing_dp_kPa` (area below 5% of A0), output ranges, swept volume. A design whose tube
never closes in the range has `closing_dp_kPa: null` — widen the range or strengthen the squeeze.

**E. Full neuron**: needs a built N design and an A design with results. `mns design new <name> --space full`,
write `neuron`, `activation`, `link: {part: <membrane>}` in design.yaml, `mns cad build F###`, `mns check F###`,
`mns solve F###`, then `mns characterise F### --axis Weight1.pressure=0:20:5 --axis Weight2.pressure=0:20:5`.

**F. Exploring a space**: vary one parameter at a time first (3–5 designs spanning the range), find which
parameters matter, then combine. Prefer few informative designs over many similar ones. After each batch,
`mns compare --x <param> --y <metric>` to see the trend (the plot goes to report/figures for the study report).

---

## 6. Speed and budget

| Step | Typical time |
|---|---|
| status, design list/show, note, jobs, wait | under 1 s |
| commands that load the model (start-up) | 4–7 s |
| cad build | 5–10 s |
| check | 5–15 s |
| neuron solve | 5–30 s |
| neuron sweep | 5–15 s per point (the first ~4 kPa from rest are the slowest) |
| activation dpsweep | 10–40 s per point (solid pushers are much slower) |
| full characterise | 1–10 s per point |
| report --build | 10–30 s |

Start sweeps coarse (4–5 points per axis); refine only where something happens. Respect the time budget in
brief.md. Use `--max-minutes` on runs that might hang.

---

## 7. When things fail

| Symptom | Try (in order) | Then |
|---|---|---|
| OVERLAP in build | fix the coordinates or build the fluid with `cavity` | — |
| chamber "touches no membrane" | check its faces meet the membrane face exactly; `cad render --views section-x` | — |
| solve not converged (`load_reached` < 1) | `--set solver.load_steps=20`; lower pressures; `elements_per_side` 8 | record as a finding (e.g. a limit point / snap-through) |
| membrane hits a wall / huge stretch (`max_area_stretch` > 2) | bigger chamber, thicker or stiffer membrane | treat as a design limit |
| "EXTRAPOLATING" | dpsweep the activation design over a wider Δp | — |
| activation point not converged | fewer/larger dp steps don't help; try `--set study.max_coupling_iterations=60`, a coarser tube mesh | note it |
| unexpected error inside the simulator | read the last lines of .mns/cli.log | write it in feature_requests.md, move on |

Known limits: a deformable Solid pusher may not converge on round tubes (use a free rigid pusher); no
self-contact of one sheet; flow is steady, isothermal and lumped.

Two attempts per problem, then record it and move on. A failed design is a result: report it.

---

## 8. Reports

`mns report design <ID>` writes `report/auto_design.tex` (parameters, bodies, model picture) and
`report/auto_results.tex` (tables, figures) — never edit those — and once creates
`report/narrative.tex`, a skeleton with sections Aim / Design / Results / Findings and the `\input` lines.
Replace each `%` comment with prose (Edit tool), keep the `\input` lines, then `mns report design <ID> --build`.
The output lists `unwritten_sections` and any `latex_errors` (fix them in narrative.tex).

Writing rules:
- Refer to the generated tables and figures; don't retype numbers you did not get from a command output.
- Say what was varied (vs the parent), what happened, why (physics), and what it means for the brief.
- LaTeX: escape `_ % & #` in text (`\_`), math in `$...$`, units as `kPa`, `mm$^3$`.

The study report: `mns report study` creates report/narrative.tex (Introduction, Method, Designs, Findings,
Conclusions) with `auto_overview.tex` (all designs and metrics) and `auto_designs.tex` (one section per
design) generated. Put comparison figures from `mns compare --x --y` in Findings. Build with `--build`.

---

## 9. Saving tokens

- Command outputs are summaries; the full data is in results/*.csv/json. Read those only for a specific number.
- Never read model.step, model.mns/.mad/.mfn, preview.npz, sweep_rows.json or cli.log whole (use `tail`/`grep`).
- Look at pictures when they decide something (new geometry, a surprising result); not after every run.
- `mns design show <ID>` instead of reading state.json; `mns compare` instead of opening many designs.
- Keep notes.md short and current: it is your memory if the conversation is compacted.
