"""Draw the app icon (app/icon.png, app/icon.ico): a neuron on a rounded tile. Run: python examples/make_icon.py app"""
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

S = 1024  # drawn large, scaled down (anti-aliasing)
out = Path(sys.argv[1])

img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
# tile: deep blue gradient, rounded
tile = Image.new("RGBA", (S, S))
td = ImageDraw.Draw(tile)
for y in range(S):
    t = y / S
    td.line([(0, y), (S, y)], fill=(int(18 + 20 * t), int(40 + 30 * t), int(92 + 40 * t), 255))
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([24, 24, S - 24, S - 24], radius=200, fill=255)
img.paste(tile, (0, 0), mask)

glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
g = ImageDraw.Draw(glow)
art = Image.new("RGBA", (S, S), (0, 0, 0, 0))
a = ImageDraw.Draw(art)

CELL = (255, 196, 64, 255)      # warm amber neuron
MYELIN = (120, 220, 255, 255)   # cyan sheath
soma = (430, 420)


def branch(draw, x, y, angle, length, width, depth, color):
    x2, y2 = x + length * math.cos(angle), y + length * math.sin(angle)
    draw.line([(x, y), (x2, y2)], fill=color, width=int(width))
    draw.ellipse([x2 - width / 2, y2 - width / 2, x2 + width / 2, y2 + width / 2], fill=color)
    if depth > 0:
        for da in (-0.5, 0.45):
            branch(draw, x2, y2, angle + da, length * 0.68, max(width * 0.62, 10), depth - 1, color)


for ang in (math.radians(v) for v in (200, 250, 300, 150, 110)):
    for draw, col, extra in ((g, (255, 200, 80, 255), 18), (a, CELL, 0)):
        branch(draw, *soma, ang, 150, 44 + extra, 2, col)

# axon towards the lower right, with myelin segments and terminal branches
end = (800, 790)
for draw, col, extra in ((g, (255, 200, 80, 255), 18), (a, CELL, 0)):
    draw.line([soma, end], fill=col, width=30 + extra)
dx, dy = end[0] - soma[0], end[1] - soma[1]
L = math.hypot(dx, dy)
ux, uy = dx / L, dy / L
for k in range(3):
    t0, t1 = 0.30 + 0.21 * k, 0.30 + 0.21 * k + 0.12
    p0 = (soma[0] + dx * t0, soma[1] + dy * t0)
    p1 = (soma[0] + dx * t1, soma[1] + dy * t1)
    a.line([p0, p1], fill=MYELIN, width=62)
    for p in (p0, p1):
        a.ellipse([p[0] - 31, p[1] - 31, p[0] + 31, p[1] + 31], fill=MYELIN)
base = math.atan2(dy, dx)
for da in (-0.7, 0.0, 0.7):
    x2, y2 = end[0] + 90 * math.cos(base + da), end[1] + 90 * math.sin(base + da)
    a.line([end, (x2, y2)], fill=CELL, width=22)
    a.ellipse([x2 - 26, y2 - 26, x2 + 26, y2 + 26], fill=(255, 240, 200, 255))

# soma with nucleus
r = 120
for draw, col, rr in ((g, (255, 200, 80, 255), r + 30), (a, CELL, r)):
    draw.ellipse([soma[0] - rr, soma[1] - rr, soma[0] + rr, soma[1] + rr], fill=col)
a.ellipse([soma[0] - 50, soma[1] - 50, soma[0] + 50, soma[1] + 50], fill=(200, 90, 30, 255))
a.ellipse([soma[0] - 30, soma[1] - 38, soma[0] - 4, soma[1] - 12], fill=(255, 220, 170, 200))

glow = glow.filter(ImageFilter.GaussianBlur(28))
glow.putalpha(Image.eval(glow.getchannel("A"), lambda v: v * 0.45))
img = Image.alpha_composite(img, glow)
img = Image.alpha_composite(img, art)
clip = Image.new("RGBA", (S, S), (0, 0, 0, 0))
clip.paste(img, (0, 0), mask)

clip.resize((256, 256), Image.LANCZOS).save(out / "icon.png")
clip.save(out / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("written", out)
