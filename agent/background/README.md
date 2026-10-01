# Background: the research behind this study

Read `article.tex` once at the start of the study (its LaTeX source is shorter and more exact than the PDF;
`article.pdf` is the same article for people). It is the theory this simulator was built to test: a
mechanical neuron that computes with fluid pressure, so you know **why** the designs look the way they do and what
a "good" neuron is. You don't need to reproduce its derivations; use it to frame your questions and findings.

## What to take from it

- **The neuron** (Sec. 3): input pressures push on membranes into a sealed chamber; the chamber pressure p_a is
  the neuron's internal state; a pressure-driven valve turns p_a into an output. The inputs are not consumed
  (no fluid enters the sealed chamber), which is why the membranes and sealed chambers must stay sealed.
- **Weights** (Eqs. 3.11-3.12): each input's weight is its membrane's compliance W_j = A_j/K_j, and p_a is the
  weighted average of the inputs. Bigger/softer membrane = bigger weight.
- **Changing weights after fabrication** (Sec. 3.2.3), the point of the research (neuroplasticity):
  *deformation limiting* (a stop that restricts how far a membrane can bulge) and *bulk-modulus tuning* (two
  membranes with a weight chamber between them; the stiffer the fluid in it, the more force passes through).
- **Output** (Sec. 3.2.4): a valve driven by p_a, inverted or non-inverted, modelled as a sigmoid (Eq. 4.2).
- **Expressiveness** (Sec. 4.3): with constant stiffness the neuron is an artificial neuron with non-negative
  weights that, with the bias and chamber weights, sum to 1; every reachable surface passes through the
  equal-pressure diagonal (all pressures equal gives p_a equal to them). With deformation-dependent stiffness it
  becomes a nonlinear, state-dependent element: this is where geometry can buy expressiveness.

## Same ideas, different names in the simulator

| Article | Simulator (MANUAL.md) |
|---|---|
| activation chamber, activation pressure p_a | **pre-activation chamber**, pre-activation pressure p_a (neuron space, N###) |
| W_j = A_j/K_j (area over volumetric stiffness) | W_j = ΔV_j/(p_j − p_a), fitted as a polynomial in Δp (`mns fit`) |
| W_0 (chamber compliance to atmosphere) | W_0 of the chamber's own liquid/gas |
| bias membrane W_b p_b | an extra input held at a constant pressure; **B** in the neuron equation is a measured bias (not the bulk modulus) |
| **B = bulk modulus** of the weight fluid | the weight chamber's `liquid` `stiffness` (kPa per % volume) or a `gas` chamber |
| weight chamber (Fig. 4) | a closed intermediate chamber in an input path |
| deformation limiting (Fig. 3) | a rigid body (stop) next to the membrane |
| activation-to-output kernel, valve (Fig. 5) | activation space (A###): a membrane squeezes a soft tube; the full neuron (F###) joins the two |

## What the simulations have shown since (not yet in the article)

- A closed chamber in an input path that is not neutral at rest (gas above 0 kPa, over- or under-filled liquid)
  adds a bias B: p_a = (Σ W_j p_j + W_0 p_0 + B)/ΣW. B is measured, not fitted.
- A compressible weight chamber does not simply push W towards 0 as the article's limit (Eq. 3.19) suggests: the
  inner membrane becomes a hidden weight to the weight chamber's rest pressure, and a single-W fit can even go
  negative when p_a lies between 0 and p_j.
- W stays positive for passive paths (energy argument) except with bias, coupling between paths, snap-through, or
  near Δp = 0 where it is noise.

Treat these as known; a design study that confirms, quantifies or contradicts the article is a useful result, so
say so in the study report when your findings bear on it.
