#!/usr/bin/env python3
"""Generate a clean SVG diagram for the FAnDa algorithm."""

from __future__ import annotations

from html import escape
from pathlib import Path


OUTPUT = Path("results/figures/fanda_algorithm_diagram.svg")

WIDTH = 1100
HEIGHT = 1180

BLUE = "#1b4db1"
BLUE_DARK = "#143b89"
BLUE_LIGHT = "#edf3ff"
TEXT = "#111111"
TEXT_MUTED = "#4a4a4a"
WHITE = "#ffffff"
FONT_SANS = "Arial, Helvetica, sans-serif"
FONT_SERIF = "Times New Roman, Times, serif"


def text(x, y, content, *, size=24, fill=TEXT, anchor="start", weight="normal", family=FONT_SANS):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'font-family="{family}" text-anchor="{anchor}" font-weight="{weight}">{escape(content)}</text>'
    )


def rect(x, y, w, h, *, fill=WHITE, stroke=BLUE, stroke_width=2.0, rx=18):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.1f}" rx="{rx:.1f}" />'
    )


def line(x1, y1, x2, y2, *, stroke=TEXT, stroke_width=3.5, marker_end=False):
    marker = ' marker-end="url(#arrow)"' if marker_end else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.1f}"{marker} />'
    )


def circle(cx, cy, r):
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{BLUE}" '
        f'stroke="{BLUE}" stroke-width="2.0" />'
    )


def box(svg, *, x, y, w, h, number, title, lines, formula=None):
    header_h = 58
    svg.append(rect(x, y, w, h))
    svg.append(line(x, y + header_h, x + w, y + header_h, stroke=BLUE, stroke_width=2.0))
    svg.append(circle(x + 38, y + header_h / 2, 20))
    svg.append(text(x + 38, y + header_h / 2 + 8, str(number), size=28, fill=WHITE, anchor="middle", weight="bold"))
    svg.append(text(x + 78, y + 38, title, size=28, fill=BLUE_DARK, weight="bold"))

    body_y = y + header_h + 38
    for idx, item in enumerate(lines):
        family = FONT_SERIF if any(sym in item for sym in ["x_<", "z_", "l*", "y_t", "t + 1"]) else FONT_SANS
        fill = TEXT_MUTED if idx == 0 else TEXT
        svg.append(text(x + 26, body_y + idx * 38, item, size=24, fill=fill, family=family))

    if formula:
        fx = x + 26
        fy = y + h - 72
        fw = w - 52
        fh = 54
        svg.append(rect(fx, fy, fw, fh, fill=BLUE_LIGHT, stroke=BLUE, stroke_width=1.8, rx=14))
        svg.append(text(x + w / 2, fy + 36, formula, size=28, anchor="middle", family=FONT_SERIF))


def main():
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        '  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="strokeWidth">',
        '    <path d="M 0 0 L 10 5 L 0 10 z" fill="#111111" />',
        "  </marker>",
        "</defs>",
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="{WHITE}" />',
        text(WIDTH / 2, 72, "Frozen-Anchor Contrastive Decoding (FAnDa)", size=46, anchor="middle", weight="bold"),
    ]

    x = 150
    w = 800
    gap = 26

    boxes = [
        {
            "h": 112,
            "title": "Input prefix",
            "lines": ["At decoding step t", "current prefix: x_<t"],
            "formula": None,
        },
        {
            "h": 136,
            "title": "Initial forward pass",
            "lines": ["At t = 0", "compute final and shallow logits"],
            "formula": "z_F^(0) and { z_l^(0) }",
        },
        {
            "h": 168,
            "title": "Choose frozen anchor",
            "lines": ["Select one shallow layer once", "freeze the same anchor for all later steps"],
            "formula": "l* = argmax_l JSD( softmax(z_F^(0)/tau), softmax(z_l^(0)/tau) )",
        },
        {
            "h": 150,
            "title": "Contrast at each step",
            "lines": ["Rerun on the updated prefix", "use raw-logit contrast"],
            "formula": "s^(t) = z_F^(t) - z_(l*)^(t)",
        },
        {
            "h": 150,
            "title": "Pick token and continue",
            "lines": ["Choose the next token", "append token, set t <- t + 1, keep l* fixed"],
            "formula": "y_t = argmax s^(t)",
        },
    ]

    y = 120
    centers = []
    for idx, spec in enumerate(boxes, start=1):
        box(svg, x=x, y=y, w=w, h=spec["h"], number=idx, title=spec["title"], lines=spec["lines"], formula=spec["formula"])
        centers.append((x + w / 2, y, spec["h"]))
        y += spec["h"] + gap

    for (_, y0, h0), (_, y1, _) in zip(centers, centers[1:]):
        cx = x + w / 2
        svg.append(line(cx, y0 + h0, cx, y1, marker_end=True))

    loop_right = x + w + 60
    last_top = centers[-1][1]
    last_bottom = centers[-1][1] + centers[-1][2]
    first_mid = centers[0][1] + centers[0][2] / 2
    svg.append(line(x + w / 2, last_bottom, x + w / 2, last_bottom + 24, marker_end=True))
    svg.append(line(x + w / 2, last_bottom + 24, loop_right, last_bottom + 24))
    svg.append(line(loop_right, last_bottom + 24, loop_right, first_mid))
    svg.append(line(loop_right, first_mid, x + w + 8, first_mid, marker_end=True))
    svg.append(text(loop_right - 8, first_mid - 16, "repeat until EOS", size=22, fill=TEXT_MUTED, anchor="end"))

    svg.append("</svg>")
    OUTPUT.write_text("\n".join(svg), encoding="utf-8")


if __name__ == "__main__":
    main()
