"""Starting points for `mns design new --template`: small, working designs to copy from."""
import yaml

from .util import ApiError

TEMPLATES = {
    "neuron_basic": ("neuron", "Two inputs pushing on a liquid-filled pre-activation chamber through two clamped disc "
                               "membranes, in a rigid housing (all cylinders on the z axis).", """
parameters:
  R: 8            # membrane and chamber radius (mm)
  t: 0.5          # membrane thickness (mm)
  h_pre: 4        # pre-activation chamber height (mm)
  h_in: 3         # input chamber height (mm)
  wall: 2         # housing wall (mm)
  E: 0.5          # membrane Young's modulus (MPa)
bodies:
  - name: Membrane1
    shape: {cylinder: {base: [0, 0, "-h_pre/2 - t"], axis: [0, 0, t], radius: R}}
    role: membrane
    props: {youngs_modulus: E, thickness: t, elements_per_side: 12}
  - name: Membrane2
    shape: {cylinder: {base: [0, 0, "h_pre/2"], axis: [0, 0, t], radius: R}}
    role: membrane
    props: {youngs_modulus: E, thickness: t, elements_per_side: 12}
  - name: PreActivation
    shape: {cylinder: {base: [0, 0, "-h_pre/2"], axis: [0, 0, h_pre], radius: R}}
    role: chamber
    props: {model: liquid, stiffness: 10}
  - name: Input1
    shape: {cylinder: {base: [0, 0, "-h_pre/2 - t - h_in"], axis: [0, 0, h_in], radius: R}}
    role: chamber
    props: {model: input, pressure: 10}
  - name: Input2
    shape: {cylinder: {base: [0, 0, "h_pre/2 + t"], axis: [0, 0, h_in], radius: R}}
    role: chamber
    props: {model: input, pressure: 0}
  - name: Housing
    shape:
      cut:
        from: {cylinder: {base: [0, 0, "-h_pre/2 - t - h_in - wall"], axis: [0, 0, "h_pre + 2*t + 2*h_in + 2*wall"],
                          radius: "R + wall"}}
        remove: [{cylinder: {base: [0, 0, "-h_pre/2 - t - h_in"], axis: [0, 0, "h_pre + 2*t + 2*h_in"], radius: R}}]
    role: rigid
solver: {load_steps: 10}
"""),
    "activation_valve": ("activation", "A round silicone tube on a rigid block, squeezed by a pusher bonded under a "
                                       "membrane; gas from an inlet (+z) through the tube to an outlet (-z).", """
parameters:
  R_out: 1.5      # tube outer radius (mm)
  R_in: 0.8       # tube inner radius (mm)
  L: 10           # tube length, along z (mm)
  pusher_w: 2     # pusher width (x) (mm)
  pusher_l: 4     # pusher length (z) (mm)
  mem_w: 6        # membrane width (x) (mm)
  t: 0.5          # membrane thickness (mm)
  top: 3          # pusher top = membrane underside (y) (mm)
  E_tube: 0.5     # tube Young's modulus (MPa)
  p_in: 10        # inlet pressure (kPa)
bodies:
  - name: Tube
    shape: {cut: {from: {cylinder: {base: [0, 0, "-L/2"], axis: [0, 0, L], radius: R_out}},
                  remove: [{cylinder: {base: [0, 0, "-L/2"], axis: [0, 0, L], radius: R_in}}]}}
    role: tube
    props: {youngs_modulus: E_tube, elements_per_side: 4}
  - name: Block
    shape: {box: {min: [-2, "-R_out - 1", "-L/2"], size: [4, 1, L]}}
    role: rigid
  - name: Membrane
    shape: {box: {min: ["-mem_w/2", top, "-L/2"], size: [mem_w, t, L]}}
    role: membrane
    props: {thickness: t, elements_per_side: 6}
  - name: Pusher
    shape: {cut: {from: {box: {min: ["-pusher_w/2", 0, "-pusher_l/2"], size: [pusher_w, top, pusher_l]}},
                  remove: [{cylinder: {base: [0, 0, "-L/2"], axis: [0, 0, L], radius: R_out}}]}}
    role: rigid
    props: {motion: free}
  - name: TubeFluid
    shape: {cylinder: {base: [0, 0, "-L/2"], axis: [0, 0, L], radius: R_in}}
    role: fluid
    props: {model: dynamic, segments: 4, outputs: [{name: activation, segment: 4}]}
  - name: InletFluid
    shape: {box: {min: [-2, -2, "L/2"], size: [4, 4, 1]}}
    role: fluid
    props: {model: constant, pressure: p_in}
  - name: OutletFluid
    shape: {box: {min: [-2, -2, "-L/2 - 1"], size: [4, 4, 1]}}
    role: fluid
    props: {model: constant, pressure: 0}
connections:
  "OutletFluid <-> TubeFluid": {type: orifice}
study: {dp_min: 0, dp_max: 30, points: 7}
"""),
    "full": ("full", "A neuron design with an activation design in place of one of its membranes.", """
neuron: N001          # the neuron design (N###)
activation: A001      # the activation design (A###), simulated with mns dpsweep
link: {part: Membrane2, driving: Automatic}   # the neuron membrane the activation design's membrane replaces
set: {}               # chamber values for this full neuron, e.g. {Input1.pressure: 10}
"""),
}
BLANK = {"neuron": "parameters: {}\nbodies: []\nsolver: {load_steps: 10}\n",
         "activation": "parameters: {}\nbodies: []\nconnections: {}\nstudy: {dp_min: 0, dp_max: 20, points: 11}\n",
         "full": TEMPLATES["full"][2]}


def list_templates():
    return {name: {"space": space, "description": text} for name, (space, text, _) in TEMPLATES.items()}


def template_spec(name, space):
    if not name:
        return yaml.safe_load(BLANK[space]) or {}
    if name not in TEMPLATES:
        raise ApiError(f"No template {name!r}", f"Templates: {', '.join(TEMPLATES)}")
    t_space, _, text = TEMPLATES[name]
    if space and space != t_space and space != "neuron":
        raise ApiError(f"Template {name} is a {t_space} design, not {space}.")
    return yaml.safe_load(text)


def template_space(name):
    return TEMPLATES[name][0] if name in TEMPLATES else None
