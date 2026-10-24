"""
Build the 3D soft neuron as a STEP assembly (millimetres), the way it would come out of CAD:
every part is its own solid - frame, two membranes, three fluid chambers, two obstacles.

Run:  python examples/make_neuron_step.py [output.step]
"""
import re
import sys
from pathlib import Path

import gmsh

# Geometry (mm)
CAVITY = (25.0, 30.0, 20.0)  # half extents x, y, z of the cavity inside the frame
WALL = 10.0                  # frame wall thickness
MEMBRANE_X = 15.0            # membrane mid-planes at x = -+15
THICKNESS = 1.0              # membrane thickness
W = [(-10.0, 10.0), (5.0, 10.0)]  # obstacle control points (the old weight tensor, scaled to mm)
X_END = 14.0


def build(path: Path):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    occ = gmsh.model.occ
    cx, cy, cz = CAVITY
    t2 = THICKNESS / 2

    parts = []  # (name, tag) in the order they are created

    outer = occ.addBox(-cx - WALL, -cy - WALL, -cz - WALL, 2 * (cx + WALL), 2 * (cy + WALL), 2 * (cz + WALL))
    cavity = occ.addBox(-cx, -cy, -cz, 2 * cx, 2 * cy, 2 * cz)
    frame = occ.cut([(3, outer)], [(3, cavity)])[0][0][1]
    parts.append(("Frame", frame))

    for name, x in (("Membrane_Left", -MEMBRANE_X), ("Membrane_Right", MEMBRANE_X)):
        parts.append((name, occ.addBox(x - t2, -cy, -cz, THICKNESS, 2 * cy, 2 * cz)))

    def obstacle(sign):
        pts = [(-X_END, cy), *W, (X_END, cy)]
        tags = [occ.addPoint(x, sign * y, -cz) for x, y in pts]
        lines = [occ.addLine(tags[i], tags[(i + 1) % len(tags)]) for i in range(len(tags))]
        surface = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        return [e for e in occ.extrude([(2, surface)], 0, 0, 2 * cz) if e[0] == 3][0][1]

    top, bottom = obstacle(+1), obstacle(-1)
    parts += [("Obstacle_Top", top), ("Obstacle_Bottom", bottom)]

    left = occ.addBox(-cx, -cy, -cz, cx - MEMBRANE_X - t2, 2 * cy, 2 * cz)
    right = occ.addBox(MEMBRANE_X + t2, -cy, -cz, cx - MEMBRANE_X - t2, 2 * cy, 2 * cz)
    middle_box = occ.addBox(-MEMBRANE_X + t2, -cy, -cz, 2 * (MEMBRANE_X - t2), 2 * cy, 2 * cz)
    middle = occ.cut([(3, middle_box)], [(3, top), (3, bottom)], removeTool=False)[0][0][1]
    parts += [("Chamber_Left", left), ("Chamber_Middle", middle), ("Chamber_Right", right)]
    occ.synchronize()

    # Write only the parts, in order
    order = {tag: i for i, (_, tag) in enumerate(parts)}
    for dim, tag in gmsh.model.getEntities(3):
        if tag not in order:
            occ.remove([(dim, tag)], recursive=True)
    occ.synchronize()
    gmsh.write(str(path))
    written = [tag for _, tag in gmsh.model.getEntities(3)]
    gmsh.finalize()

    # gmsh writes generic product names; give every solid its part name like a CAD export would
    tag_to_name = {tag: name for name, tag in parts}
    text = path.read_text()
    text = re.sub(r"Open CASCADE STEP translator [\d.]+ \d+\.(\d+)",
                  lambda m: tag_to_name[written[int(m.group(1)) - 1]], text)
    text = re.sub(r"'Open CASCADE STEP translator [\d.]+ \d+'", "'SoftNeuron'", text)
    path.write_text(text)
    return [name for name, _ in parts]


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("soft_neuron.step")
    print("Wrote", out, "with parts:", ", ".join(build(out)))
