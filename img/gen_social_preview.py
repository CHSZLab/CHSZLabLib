#!/usr/bin/env python3
"""Generate GitHub social preview image for CHSZLabLib (1280x640)."""

from PIL import Image, ImageDraw, ImageFont
import math

W, H = 1280, 640

# Colors
BG_DARK = (15, 23, 42)        # slate-900
BG_MID = (30, 41, 59)         # slate-800
ACCENT = (59, 130, 246)       # blue-500
ACCENT_LIGHT = (96, 165, 250) # blue-400
WHITE = (255, 255, 255)
GRAY = (148, 163, 184)        # slate-400
LIGHT_GRAY = (203, 213, 225)  # slate-300
GREEN = (34, 197, 94)         # green-500
PURPLE = (168, 85, 247)       # purple-500
ORANGE = (249, 115, 22)       # orange-500
CYAN = (6, 182, 212)          # cyan-500

# Fonts
font_bold_64 = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-Bold.otf", 64)
font_bold_48 = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-Bold.otf", 48)
font_semi_28 = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-SemiBold.otf", 28)
font_med_24 = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-Medium.otf", 24)
font_reg_22 = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-Regular.otf", 22)
font_mono_20 = ImageFont.truetype("/home/c_schulz/.local/share/fonts/FiraCodeNerdFont-Bold.ttf", 20)
font_mono_18 = ImageFont.truetype("/home/c_schulz/.local/share/fonts/FiraCodeNerdFont-Bold.ttf", 18)

img = Image.new("RGB", (W, H), BG_DARK)
draw = ImageDraw.Draw(img)

# Subtle gradient overlay (top to bottom)
for y in range(H):
    alpha = y / H
    r = int(BG_DARK[0] * (1 - alpha * 0.3) + BG_MID[0] * (alpha * 0.3))
    g = int(BG_DARK[1] * (1 - alpha * 0.3) + BG_MID[1] * (alpha * 0.3))
    b = int(BG_DARK[2] * (1 - alpha * 0.3) + BG_MID[2] * (alpha * 0.3))
    draw.line([(0, y), (W, y)], fill=(r, g, b))

# Draw decorative graph nodes and edges in background
nodes = [
    (95, 120, 8, ACCENT),
    (180, 80, 6, PURPLE),
    (140, 200, 7, CYAN),
    (250, 150, 5, GREEN),
    (60, 250, 6, ACCENT_LIGHT),
    (200, 280, 5, ORANGE),
    # Right side
    (1100, 100, 7, ACCENT),
    (1200, 160, 6, PURPLE),
    (1150, 250, 8, CYAN),
    (1050, 200, 5, GREEN),
    (1220, 80, 5, ACCENT_LIGHT),
    (1080, 310, 6, ORANGE),
    # Bottom corners
    (100, 500, 6, ACCENT),
    (200, 550, 5, PURPLE),
    (1150, 520, 7, CYAN),
    (1050, 570, 5, GREEN),
    (160, 580, 4, ACCENT_LIGHT),
    (1200, 560, 5, ORANGE),
]

edges = [
    (0, 1), (0, 2), (1, 3), (2, 4), (2, 5), (3, 5), (4, 5),
    (6, 7), (6, 8), (7, 9), (8, 9), (6, 10), (8, 11), (9, 11),
    (12, 13), (13, 16), (12, 16),
    (14, 15), (14, 17), (15, 17),
]

for i, j in edges:
    x1, y1 = nodes[i][0], nodes[i][1]
    x2, y2 = nodes[j][0], nodes[j][1]
    c = tuple(int(v * 0.25) for v in nodes[i][3])
    draw.line([(x1, y1), (x2, y2)], fill=c, width=2)

for x, y, r, color in nodes:
    faded = tuple(int(v * 0.4) for v in color)
    draw.ellipse([x - r, y - r, x + r, y + r], fill=faded)
    # inner bright dot
    r2 = max(2, r - 3)
    bright = tuple(min(255, int(v * 0.6)) for v in color)
    draw.ellipse([x - r2, y - r2, x + r2, y + r2], fill=bright)

# Accent line at top
draw.rectangle([0, 0, W, 4], fill=ACCENT)

# Title — "CHSZ" in red, "LabLib" in white
RED = (220, 38, 38)
part1 = "CHSZ"
part2 = "LabLib"
bbox1 = draw.textbbox((0, 0), part1, font=font_bold_64)
bbox2 = draw.textbbox((0, 0), part2, font=font_bold_64)
tw = (bbox1[2] - bbox1[0]) + (bbox2[2] - bbox2[0])
tx = (W - tw) // 2
draw.text((tx, 100), part1, fill=RED, font=font_bold_64)
tx += bbox1[2] - bbox1[0]
draw.text((tx, 100), part2, fill=WHITE, font=font_bold_64)

# Subtitle
sub = "State-of-the-art graph algorithms from C++ to Python"
bbox = draw.textbbox((0, 0), sub, font=font_semi_28)
sw = bbox[2] - bbox[0]
draw.text(((W - sw) // 2, 185), sub, fill=LIGHT_GRAY, font=font_semi_28)

# Divider
div_y = 235
div_w = 200
draw.line([(W // 2 - div_w, div_y), (W // 2 + div_w, div_y)], fill=ACCENT, width=2)

# Module boxes
modules = [
    ("Decomposition", "Partition, Cuts, Cluster", ACCENT),
    ("Independence", "MIS, MWIS, HyperMIS", GREEN),
    ("Orientation", "Edge Orientation", PURPLE),
    ("Dynamic", "Dynamic Problems", ORANGE),
]

box_w = 250
box_h = 90
gap = 30
total_w = len(modules) * box_w + (len(modules) - 1) * gap
start_x = (W - total_w) // 2
box_y = 265

for i, (name, desc, color) in enumerate(modules):
    x = start_x + i * (box_w + gap)
    # Box background
    box_color = tuple(int(v * 0.15) + BG_DARK[j] for j, v in enumerate(color))
    draw.rounded_rectangle([x, box_y, x + box_w, box_y + box_h], radius=10, fill=box_color)
    # Border
    border_color = tuple(int(v * 0.5) for v in color)
    draw.rounded_rectangle([x, box_y, x + box_w, box_y + box_h], radius=10, outline=border_color, width=2)
    # Top accent
    draw.line([(x + 10, box_y + 1), (x + box_w - 10, box_y + 1)], fill=color, width=2)
    # Name
    bbox = draw.textbbox((0, 0), name, font=font_med_24)
    nw = bbox[2] - bbox[0]
    draw.text((x + (box_w - nw) // 2, box_y + 15), name, fill=WHITE, font=font_med_24)
    # Description
    bbox = draw.textbbox((0, 0), desc, font=font_reg_22)
    dw = bbox[2] - bbox[0]
    draw.text((x + (box_w - dw) // 2, box_y + 50), desc, fill=GRAY, font=font_reg_22)

# Code snippet
code_y = 390
code_bg = (22, 30, 48)
code_h = 130
code_w = 700
code_x = (W - code_w) // 2
draw.rounded_rectangle([code_x, code_y, code_x + code_w, code_y + code_h], radius=12, fill=code_bg)
draw.rounded_rectangle([code_x, code_y, code_x + code_w, code_y + code_h], radius=12, outline=(51, 65, 85), width=1)

# Terminal dots
for ci, col in enumerate([(239, 68, 68), (250, 204, 21), (34, 197, 94)]):
    draw.ellipse([code_x + 16 + ci * 20, code_y + 12, code_x + 28 + ci * 20, code_y + 24], fill=col)

lines = [
    ("from", (198, 120, 221)),  # keyword
    (" chszlablib ", LIGHT_GRAY),
    ("import", (198, 120, 221)),
    (" Graph, Decomposition", (230, 192, 123)),
]
lx = code_x + 20
ly = code_y + 40
for text, color in lines:
    draw.text((lx, ly), text, fill=color, font=font_mono_18)
    bbox = draw.textbbox((0, 0), text, font=font_mono_18)
    lx += bbox[2] - bbox[0]

# Second line
lx = code_x + 20
ly = code_y + 68
line2 = [
    ("g = Graph(", LIGHT_GRAY),
    ("n=100", (230, 192, 123)),
    (").add_edges(", LIGHT_GRAY),
    ("edges", (152, 195, 121)),
    (").finalize()", LIGHT_GRAY),
]
for text, color in line2:
    draw.text((lx, ly), text, fill=color, font=font_mono_18)
    bbox = draw.textbbox((0, 0), text, font=font_mono_18)
    lx += bbox[2] - bbox[0]

# Third line
lx = code_x + 20
ly = code_y + 96
line3 = [
    ("result = Decomposition.partition(g, k=", LIGHT_GRAY),
    ("4", (230, 192, 123)),
    (")", LIGHT_GRAY),
]
for text, color in line3:
    draw.text((lx, ly), text, fill=color, font=font_mono_18)
    bbox = draw.textbbox((0, 0), text, font=font_mono_18)
    lx += bbox[2] - bbox[0]

# Footer
footer = "Algorithm Engineering Group, Heidelberg University"
bbox = draw.textbbox((0, 0), footer, font=font_reg_22)
fw = bbox[2] - bbox[0]

# Badges row — colors from README shields.io badges
badges = ["Python 3.9+", "C++17", "pybind11"]
badge_font = font_mono_20
badge_gap = 20
badge_colors = [
    (0x37, 0x76, 0xab),  # #3776ab — Python badge
    (0x00, 0x59, 0x9C),  # #00599C — C++ badge
    (0x06, 0x4F, 0x8C),  # #064F8C — build/cmake badge
]

# Measure total width
total_badge_w = 0
badge_dims = []
for b in badges:
    bbox = draw.textbbox((0, 0), b, font=badge_font)
    bw = bbox[2] - bbox[0] + 24
    bh = 30
    badge_dims.append((bw, bh))
    total_badge_w += bw
total_badge_w += (len(badges) - 1) * badge_gap

bx = (W - total_badge_w) // 2
by = 548

for i, (b, (bw, bh)) in enumerate(zip(badges, badge_dims)):
    col = badge_colors[i]
    bg = tuple(int(v * 0.3) for v in col)
    draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=5, fill=bg)
    draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=5, outline=col, width=1)
    bbox = draw.textbbox((0, 0), b, font=badge_font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((bx + (bw - tw) // 2, by + (bh - th) // 2 - 2), b, fill=WHITE, font=badge_font)
    bx += bw + badge_gap

# Footer text below badges
draw.text(((W - fw) // 2, 588), footer, fill=GRAY, font=font_reg_22)

out = "/home/c_schulz/projects/coding/CHSZLabLib/social-preview.png"
img.save(out, "PNG")
print(f"Saved: {out}")
print(f"Size: {img.size}")
