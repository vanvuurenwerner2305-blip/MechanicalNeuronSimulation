"""
Build a small activation-function device (a squeezed-tube valve) as a STEP file in millimetres:
a round silicone tube lying on a rigid block, a pusher resting on the tube and bonded under a
membrane, the gas filling the tube, a gas supply against one open end and a sink at the other.

Run:  python examples/make_activation_step.py [output.step]
"""
import re
import sys
from pathlib import Path

import gmsh

R_OUT, R_IN, LENGTH = 1.5, 0.8, 10.0   # tube along z, from -LENGTH/2 to LENGTH/2
PUSHER_HALF = (1.0, 2.0)               # half width (x) and half length (z) of the pusher
MEMBRANE = (3.0, 0.5)                  # half width (x) and thickness; spans the tube's length
TOP = 3.0                              # pusher top / membrane underside (y)


def build(path: Path):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    occ = gmsh.model.occ
    L2 = LENGTH / 2
    parts = []

    outer = occ.addCylinder(0, 0, -L2, 0, 0, LENGTH, R_OUT)
    inner = occ.addCylinder(0, 0, -L2, 0, 0, LENGTH, R_IN)
    tube = occ.cut([(3, outer)], [(3, inner)], removeObject=False, removeTool=False)[0][0][1]
    parts.append(("Tube", tube))
    parts.append(("Rigid Body", occ.addBox(-2.0, -R_OUT - 1.0, -L2, 4.0, 1.0, LENGTH)))
    parts.append(("Membrane", occ.addBox(-MEMBRANE[0], TOP, -L2, 2 * MEMBRANE[0], MEMBRANE[1], LENGTH)))
    hx, hz = PUSHER_HALF
    block = occ.addBox(-hx, 0.0, -hz, 2 * hx, TOP, 2 * hz)
    pusher = occ.cut([(3, block)], [(3, outer)], removeTool=False)[0][0][1]
    parts.append(("Pusher", pusher))
    parts.append(("TubeFluid", occ.addCylinder(0, 0, -L2, 0, 0, LENGTH, R_IN)))
    # gas supply against the tube's open end at +z, and the 0 kPa sink at -z
    parts.append(("InletFluid", occ.addBox(-2.0, -2.0, L2, 4.0, 4.0, 1.0)))
    parts.append(("OutletFluid", occ.addBox(-2.0, -2.0, -L2 - 1.0, 4.0, 4.0, 1.0)))
    occ.synchronize()

    order = {tag: i for i, (_, tag) in enumerate(parts)}
    for dim, tag in gmsh.model.getEntities(3):
        if tag not in order:
            occ.remove([(dim, tag)], recursive=True)
    occ.synchronize()
    gmsh.write(str(path))
    written = [tag for _, tag in gmsh.model.getEntities(3)]
    gmsh.finalize()

    tag_to_name = {tag: name for name, tag in parts}
    text = path.read_text()
    text = re.sub(r"Open CASCADE STEP translator [\d.]+ 1\.(\d+)",
                  lambda m: tag_to_name[written[int(m.group(1)) - 1]], text)
    text = re.sub(r"'Open CASCADE STEP translator [\d.]+ 1'", "'SqueezeValve'", text)
    path.write_text(text)
    return [name for name, _ in parts]


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("squeeze_valve.step")
    print("Wrote", out, "with parts:", ", ".join(build(out)))
