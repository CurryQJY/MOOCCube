import matplotlib

matplotlib.use("Agg")

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch


# Publication-style method overview for the CKG-RL framework.
# The diagram is hand-arranged on a fixed grid so the paper export remains
# deterministic and editable across SVG/PDF/TIFF outputs.

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 9,
        "legend.frameon": False,
    }
)


W, H = 2180, 820
TEXT_SCALE = 0.82

COL = {
    "ink": "#1b1d21",
    "muted": "#4d5663",
    "line": "#202833",
    "flow": "#223145",
    "aux": "#5f6f80",
    "panel": "#303944",
    "paper": "#ffffff",
    "left": "#dfe8d4",
    "middle": "#d8e4ee",
    "right": "#e1dfeb",
    "card": "#f3f1ea",
    "card2": "#edf0ea",
    "cream": "#f2ead5",
    "blue": "#184b73",
    "blue2": "#5b7890",
    "blue_soft": "#dbe5ec",
    "green": "#345f46",
    "green_soft": "#dce8d4",
    "orange": "#9c542c",
    "orange_soft": "#ebd2bf",
    "gold": "#7b611e",
    "gold_soft": "#eadfbf",
    "violet": "#514761",
    "violet_soft": "#dfdbe8",
    "rose": "#7c4958",
    "rose_soft": "#ead4da",
    "red": "#c8242e",
    "red_dark": "#9b2f2f",
    "red_soft": "#f3d8d6",
    "warn": "#9b2f2f",
    "warn_soft": "#efd8d4",
    "ice_dark": "#2b6d8c",
    "gray": "#e7eaed",
    "accent_a": "#657a5a",
    "accent_b": "#4f6b82",
    "accent_c": "#6c657c",
}


def txt(ax, x, y, s, size=10, weight="normal", color=None, ha="center", va="center", z=20):
    size = max(size, 11.0)
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        fontweight=weight,
        color=color or COL["ink"],
        ha=ha,
        va=va,
        linespacing=1.08,
        zorder=z,
    )


def box(ax, x, y, w, h, fc, ec=None, lw=1.8, r=8, ls="-", z=2):
    lw = max(lw, 1.05)
    r = min(max(r, 0), 1.2)
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec or COL["line"],
        facecolor=fc,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def panel_box(ax, x, y, w, h, fc, accent=None):
    patch = patches.Rectangle(
        (x, y),
        w,
        h,
        linewidth=2.25,
        edgecolor=COL["panel"],
        facecolor=fc,
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def orth_arrow(ax, pts, color=None, lw=2.2, ms=15, ls="-", z=12, alpha=1.0):
    color = color or COL["line"]
    lw = max(lw, 0.8)
    ms = max(ms, 9)
    if len(pts) < 2:
        raise ValueError("orth_arrow requires at least two points")
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if abs(x0 - x1) > 1e-9 and abs(y0 - y1) > 1e-9:
            raise ValueError(f"Diagonal connector segment: {(x0, y0)} -> {(x1, y1)}")
    if len(pts) > 2:
        xs, ys = zip(*pts[:-1])
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=lw,
            linestyle=ls,
            solid_capstyle="butt",
            dash_capstyle="butt",
            alpha=alpha,
            zorder=z,
        )
    arr = FancyArrowPatch(
        pts[-2],
        pts[-1],
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=ls,
        color=color,
        shrinkA=0,
        shrinkB=0,
        alpha=alpha,
        zorder=z,
    )
    ax.add_patch(arr)
    return arr


def red_boundary_arrow(ax, x, y, scale=1.0, z=30):
    pts = [
        (x - 18 * scale, y - 14 * scale),
        (x + 4 * scale, y - 14 * scale),
        (x + 4 * scale, y - 23 * scale),
        (x + 30 * scale, y),
        (x + 4 * scale, y + 23 * scale),
        (x + 4 * scale, y + 14 * scale),
        (x - 18 * scale, y + 14 * scale),
    ]
    ax.add_patch(
        patches.Polygon(
            pts,
            closed=True,
            facecolor=COL["paper"],
            edgecolor=COL["red"],
            linewidth=2.45,
            zorder=z,
        )
    )


def draw_top_training_route(ax):
    y = 54
    x0, x1 = 140, 1320
    ls = (0, (6, 5))
    ax.plot([x0, x1], [y, y], color=COL["red"], linewidth=2.0, linestyle=ls, zorder=27)
    for x in [x0, 740, x1]:
        ax.plot([x, x], [y, 96], color=COL["red"], linewidth=2.0, linestyle=ls, zorder=27)
    for x in [330, 720, 1080]:
        ax.add_patch(
            FancyArrowPatch(
                (x + 56, y - 11),
                (x + 8, y - 11),
                arrowstyle="->",
                mutation_scale=17,
                linewidth=1.75,
                color=COL["red"],
                zorder=28,
            )
        )
        ax.add_patch(
            FancyArrowPatch(
                (x + 56, y + 2),
                (x + 8, y + 2),
                arrowstyle="->",
                mutation_scale=17,
                linewidth=1.75,
                color=COL["red"],
                zorder=28,
            )
        )
    txt(
        ax,
        1020,
        78,
        "training-only feedback",
        size=8.2,
        weight="bold",
        color=COL["red"],
        ha="left",
        z=29,
    )


def vector(ax, x, y, w=70, h=17, colors=None, z=8):
    colors = colors or ["#ffffff", "#dfe6ee", "#8ea7bd", "#2f5f89"]
    n = len(colors)
    for i, c in enumerate(colors):
        ax.add_patch(
            patches.Rectangle(
                (x + i * w / n, y),
                w / n,
                h,
                facecolor=c,
                edgecolor=COL["line"],
                linewidth=1.15,
                zorder=z,
            )
        )
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="none", edgecolor=COL["line"], linewidth=1.05, zorder=z + 1))


def state_capsule(ax, x, y, w=42, h=110, label="$s_t$", colors=None, accent=None, band_labels=None, z=9):
    colors = colors or ["#f4f6f8", "#cfd8e3", "#6f879e", "#284f73"]
    accent = accent or COL["blue"]
    cap = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=2.2",
        linewidth=1.75,
        edgecolor=accent,
        facecolor=COL["paper"],
        zorder=z,
    )
    ax.add_patch(cap)
    pad = 7
    slot_h = (h - 2 * pad) / len(colors)
    for i, c in enumerate(colors):
        ax.add_patch(
            patches.Rectangle(
                (x + pad, y + pad + i * slot_h),
                w - 2 * pad,
                slot_h - 1.0,
                facecolor=c,
                edgecolor="none",
                zorder=z + 1,
            )
        )
        if band_labels and i < len(band_labels):
            ax.text(
                x + w / 2,
                y + pad + (i + 0.5) * slot_h - 0.4,
                band_labels[i],
                fontsize=11.5,
                fontweight="bold",
                color=COL["ink"] if i != 1 else COL["paper"],
                ha="center",
                va="center",
                zorder=z + 3,
            )
    ax.add_patch(
        patches.FancyBboxPatch(
            (x + pad, y + pad),
            w - 2 * pad,
            h - 2 * pad,
            boxstyle="round,pad=0.01,rounding_size=1.0",
            linewidth=1.0,
            edgecolor=COL["line"],
            facecolor="none",
            zorder=z + 2,
        )
    )
    txt(ax, x + w / 2, y + h + 20, label, size=13.0, weight="bold", color=accent)
    return cap


def hollow_arrow(ax, start, end, color=None, lw=1.35, tail=0.55, head_w=8, head_l=10, z=15):
    color = color or COL["flow"]
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=f"Simple,tail_width={tail},head_width={head_w},head_length={head_l}",
        linewidth=lw,
        edgecolor=color,
        facecolor=COL["paper"],
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(arr)
    return arr


def icon_pt(cx, cy, size, px, py):
    scale = size / 24.0
    return cx + (px - 12) * scale, cy + (py - 12) * scale


def icon_line(ax, cx, cy, size, pts, color=None, lw=1.7, z=18):
    color = color or COL["line"]
    xs = []
    ys = []
    for px, py in pts:
        ix, iy = icon_pt(cx, cy, size, px, py)
        xs.append(ix)
        ys.append(iy)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=z)


def icon_circle(ax, cx, cy, size, px, py, r, color=None, lw=1.7, fc="none", z=18):
    color = color or COL["line"]
    ix, iy = icon_pt(cx, cy, size, px, py)
    scale = size / 24.0
    ax.add_patch(patches.Circle((ix, iy), r * scale, facecolor=fc, edgecolor=color, lw=lw, zorder=z))


def snowflake_icon(ax, x, y, r=8, color=None, lw=1.15, z=18):
    color = color or COL["ice_dark"]
    # Path adapted from the Lucide snowflake icon's 24px line-art geometry.
    paths = [
        [(10, 20), (8.75, 17.5), (6, 18)],
        [(10, 4), (8.75, 6.5), (6, 6)],
        [(14, 20), (15.25, 17.5), (18, 18)],
        [(14, 4), (15.25, 6.5), (18, 6)],
        [(17, 21), (14, 15), (10, 15)],
        [(17, 3), (14, 9), (15.5, 12)],
        [(2, 12), (8.5, 12), (10, 9)],
        [(20, 10), (18.5, 12), (20, 14)],
        [(22, 12), (15.5, 12), (14, 15)],
        [(4, 10), (5.5, 12), (4, 14)],
        [(7, 21), (10, 15), (8.5, 12)],
        [(7, 3), (10, 9), (14, 9)],
    ]
    scale = r / 10.0
    for pts in paths:
        xs = [x + (px - 12) * scale for px, _ in pts]
        ys = [y + (py - 12) * scale for _, py in pts]
        ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=z)


def book_open_icon(ax, cx, cy, size=54, color=None, lw=1.55, z=18):
    color = color or COL["line"]
    scale = size / 24.0

    def p(px, py):
        return icon_pt(cx, cy, size, px, py)

    icon_line(ax, cx, cy, size, [(12, 7), (12, 21)], color=color, lw=lw, z=z)
    left = [
        (p(12, 7), MplPath.MOVETO),
        (p(11.5, 4.8), MplPath.CURVE4),
        (p(10.0, 3.0), MplPath.CURVE4),
        (p(8, 3), MplPath.CURVE4),
        (p(3, 3), MplPath.LINETO),
        (p(2.45, 3), MplPath.CURVE4),
        (p(2, 3.45), MplPath.CURVE4),
        (p(2, 4), MplPath.CURVE4),
        (p(2, 17), MplPath.LINETO),
        (p(2, 17.55), MplPath.CURVE4),
        (p(2.45, 18), MplPath.CURVE4),
        (p(3, 18), MplPath.CURVE4),
        (p(9, 18), MplPath.LINETO),
        (p(10.8, 18), MplPath.CURVE4),
        (p(12, 19.4), MplPath.CURVE4),
        (p(12, 21), MplPath.CURVE4),
    ]
    right = [
        (p(12, 7), MplPath.MOVETO),
        (p(12.5, 4.8), MplPath.CURVE4),
        (p(14.0, 3.0), MplPath.CURVE4),
        (p(16, 3), MplPath.CURVE4),
        (p(21, 3), MplPath.LINETO),
        (p(21.55, 3), MplPath.CURVE4),
        (p(22, 3.45), MplPath.CURVE4),
        (p(22, 4), MplPath.CURVE4),
        (p(22, 17), MplPath.LINETO),
        (p(22, 17.55), MplPath.CURVE4),
        (p(21.55, 18), MplPath.CURVE4),
        (p(21, 18), MplPath.CURVE4),
        (p(15, 18), MplPath.LINETO),
        (p(13.2, 18), MplPath.CURVE4),
        (p(12, 19.4), MplPath.CURVE4),
        (p(12, 21), MplPath.CURVE4),
    ]
    for verts_codes in [left, right]:
        verts, codes = zip(*verts_codes)
        ax.add_patch(
            patches.PathPatch(
                MplPath(verts, codes),
                facecolor="none",
                edgecolor=color,
                lw=lw,
                capstyle="round",
                joinstyle="round",
                zorder=z,
            )
        )
    icon_line(ax, cx, cy, size, [(5, 7), (8, 7.2)], color=color, lw=0.85 * lw, z=z)
    icon_line(ax, cx, cy, size, [(16, 7.2), (19, 7)], color=color, lw=0.85 * lw, z=z)
    icon_line(ax, cx, cy, size, [(5, 11), (8.5, 11.4)], color=color, lw=0.85 * lw, z=z)
    icon_line(ax, cx, cy, size, [(15.5, 11.4), (19, 11)], color=color, lw=0.85 * lw, z=z)
    ax.add_patch(patches.Circle(p(12, 7), 0.4 * scale, facecolor=color, edgecolor="none", zorder=z + 1))


def user_round_icon(ax, cx, cy, size=34, color=None, lw=1.55, z=18):
    color = color or COL["line"]
    icon_circle(ax, cx, cy, size, 12, 8, 5, color=color, lw=lw, z=z)
    scale = size / 24.0
    pts = []
    for t in [i / 24 for i in range(25)]:
        ang = -math.pi * t
        px = 12 + 8 * math.cos(ang)
        py = 21 + 8 * math.sin(ang)
        pts.append((cx + (px - 12) * scale, cy + (py - 12) * scale))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw, solid_capstyle="round", zorder=z)


def mlp_wedge(ax, x, y, w=62, h=42, ec=None, label="MLP", label_size=14.0):
    poly = patches.Polygon(
        [[x, y], [x + w, y + 8], [x + w, y + h - 8], [x, y + h]],
        closed=True,
        facecolor=COL["paper"],
        edgecolor=ec or COL["blue"],
        linewidth=2.4,
        zorder=8,
    )
    ax.add_patch(poly)
    txt(ax, x + w / 2 + 4, y + h / 2, label, size=label_size, weight="bold")


def course_icon(ax, x, y, w=98, h=90, framed=True):
    if framed:
        box(ax, x, y, w, h, COL["card"], lw=1.4, r=7)
    snowflake_icon(ax, x + w - 22, y + 18, r=7.2, color=COL["ice_dark"], lw=1.05, z=11)
    book_open_icon(ax, x + w / 2, y + 43, size=52, color=COL["line"], lw=1.65, z=10)
    txt(ax, x + w / 2, y + 73, "course i", size=12.2, weight="bold")
    snowflake_icon(ax, x + 28, y + 85, r=5.2, color=COL["ice_dark"], lw=0.9, z=12)
    txt(ax, x + 60, y + 89, "cold", size=11.5, weight="bold", color=COL["ice_dark"])


def user_icon(ax, x, y):
    user_round_icon(ax, x, y + 22, size=34, color=COL["line"], lw=1.6, z=10)


def mini_edu_graph(ax, x, y, s=1.0):
    pts = [
        (x + 0 * s, y + 28 * s),
        (x + 38 * s, y + 4 * s),
        (x + 76 * s, y + 28 * s),
        (x + 38 * s, y + 58 * s),
    ]
    for i, j in [(0, 1), (1, 2), (0, 3), (1, 3), (2, 3)]:
        ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], lw=1.2, color=COL["line"], zorder=12)
    for k, (px, py) in enumerate(pts):
        fill = COL["orange"] if k in {0, 2} else COL["blue2"]
        ax.add_patch(patches.Circle((px, py), 6.2 * s, facecolor=fill, edgecolor=COL["line"], lw=1.0, zorder=13))


def gate_network_icon(ax, cx, cy, size=44, color=None, accent=None, z=18):
    color = color or COL["violet"]
    accent = accent or COL["blue2"]
    scale = size / 44.0
    pts = [
        (cx - 19 * scale, cy - 12 * scale),
        (cx - 4 * scale, cy - 12 * scale),
        (cx + 12 * scale, cy - 5 * scale),
        (cx + 12 * scale, cy + 5 * scale),
        (cx - 4 * scale, cy + 12 * scale),
        (cx - 19 * scale, cy + 12 * scale),
    ]
    ax.add_patch(
        patches.Polygon(pts, closed=True, facecolor=COL["paper"], edgecolor=color, linewidth=1.7, zorder=z)
    )
    for dy in [-10, 0, 10]:
        ax.plot([cx - 30 * scale, cx - 19 * scale], [cy + dy * scale, cy + dy * scale], color=color, lw=1.35, zorder=z)
        ax.add_patch(patches.Circle((cx - 33 * scale, cy + dy * scale), 2.8 * scale, facecolor=accent, edgecolor=color, lw=0.9, zorder=z + 1))
    ax.plot([cx + 12 * scale, cx + 26 * scale], [cy, cy], color=color, lw=1.45, zorder=z)
    ax.add_patch(patches.Circle((cx + 29 * scale, cy), 3.0 * scale, facecolor=COL["orange_soft"], edgecolor=color, lw=0.9, zorder=z + 1))
    ax.text(cx - 2 * scale, cy + 1 * scale, "g", fontsize=11, fontweight="bold", color=color, ha="center", va="center", zorder=z + 2)


def brain_circuit_icon(ax, cx, cy, size=42, color=None, accent=None, z=18):
    color = color or COL["violet"]
    accent = accent or COL["blue2"]
    scale = size / 42.0
    # Compact brain-circuit motif inspired by Lucide's 24px line icon style.
    left = [
        (cx - 4 * scale, cy - 15 * scale),
        (cx - 17 * scale, cy - 14 * scale),
        (cx - 20 * scale, cy - 3 * scale),
        (cx - 15 * scale, cy + 5 * scale),
        (cx - 17 * scale, cy + 14 * scale),
        (cx - 5 * scale, cy + 16 * scale),
    ]
    right = [
        (cx - 4 * scale, cy - 15 * scale),
        (cx + 11 * scale, cy - 15 * scale),
        (cx + 18 * scale, cy - 6 * scale),
        (cx + 15 * scale, cy + 2 * scale),
        (cx + 20 * scale, cy + 11 * scale),
        (cx + 8 * scale, cy + 17 * scale),
        (cx - 5 * scale, cy + 16 * scale),
    ]
    ax.plot([p[0] for p in left], [p[1] for p in left], color=color, lw=1.55, solid_capstyle="round", solid_joinstyle="round", zorder=z)
    ax.plot([p[0] for p in right], [p[1] for p in right], color=color, lw=1.55, solid_capstyle="round", solid_joinstyle="round", zorder=z)
    nodes = [
        (cx - 8 * scale, cy - 7 * scale),
        (cx + 7 * scale, cy - 8 * scale),
        (cx - 1 * scale, cy + 3 * scale),
        (cx + 10 * scale, cy + 9 * scale),
    ]
    for i, j in [(0, 2), (1, 2), (2, 3)]:
        ax.plot([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]], color=color, lw=1.1, zorder=z)
    for px, py in nodes:
        ax.add_patch(patches.Circle((px, py), 3.0 * scale, facecolor=accent, edgecolor=color, lw=0.9, zorder=z + 1))


def update_loop_icon(ax, cx, cy, size=38, color=None, z=18):
    color = color or COL["line"]
    scale = size / 38.0
    ax.add_patch(patches.Arc((cx, cy), 28 * scale, 24 * scale, theta1=35, theta2=205, color=color, lw=1.55, zorder=z))
    ax.add_patch(patches.Arc((cx, cy), 28 * scale, 24 * scale, theta1=215, theta2=25, color=color, lw=1.55, zorder=z))
    ax.add_patch(patches.Polygon([(cx - 12 * scale, cy - 9 * scale), (cx - 18 * scale, cy - 8 * scale), (cx - 14 * scale, cy - 3 * scale)], facecolor=color, edgecolor=color, zorder=z))
    ax.add_patch(patches.Polygon([(cx + 12 * scale, cy + 9 * scale), (cx + 18 * scale, cy + 8 * scale), (cx + 14 * scale, cy + 3 * scale)], facecolor=color, edgecolor=color, zorder=z))
    ax.text(cx, cy + 1 * scale, r"$\nabla$", fontsize=11, fontweight="bold", color=color, ha="center", va="center", zorder=z + 1)


def trophy_icon(ax, cx, cy, size=34, color=None, accent=None, z=18):
    color = color or COL["gold"]
    accent = accent or COL["gold_soft"]
    scale = size / 34.0
    ax.add_patch(
        patches.FancyBboxPatch(
            (cx - 9 * scale, cy - 12 * scale),
            18 * scale,
            17 * scale,
            boxstyle=f"round,pad=0,rounding_size={3 * scale}",
            facecolor=accent,
            edgecolor=color,
            linewidth=1.45,
            zorder=z,
        )
    )
    ax.add_patch(patches.Arc((cx - 11 * scale, cy - 5 * scale), 13 * scale, 15 * scale, theta1=95, theta2=265, color=color, lw=1.3, zorder=z))
    ax.add_patch(patches.Arc((cx + 11 * scale, cy - 5 * scale), 13 * scale, 15 * scale, theta1=-85, theta2=85, color=color, lw=1.3, zorder=z))
    ax.plot([cx, cx], [cy + 5 * scale, cy + 14 * scale], color=color, lw=1.45, zorder=z)
    ax.plot([cx - 9 * scale, cx + 9 * scale], [cy + 14 * scale, cy + 14 * scale], color=color, lw=1.45, zorder=z)
    ax.plot([cx - 13 * scale, cx + 13 * scale], [cy + 19 * scale, cy + 19 * scale], color=color, lw=1.55, zorder=z)


def loss_curve_icon(ax, cx, cy, size=34, color=None, z=18):
    color = color or COL["red"]
    scale = size / 34.0
    ax.plot([cx - 15 * scale, cx - 15 * scale, cx + 15 * scale], [cy - 12 * scale, cy + 13 * scale, cy + 13 * scale], color=color, lw=1.25, zorder=z)
    pts = [
        (cx - 12 * scale, cy - 5 * scale),
        (cx - 4 * scale, cy + 2 * scale),
        (cx + 4 * scale, cy + 1 * scale),
        (cx + 12 * scale, cy + 8 * scale),
    ]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=1.55, solid_capstyle="round", zorder=z)
    for px, py in pts:
        ax.add_patch(patches.Circle((px, py), 2.1 * scale, facecolor=COL["red_soft"], edgecolor=color, lw=0.8, zorder=z + 1))


def reward_term_icon(ax, cx, cy, kind, color, accent, size=30, z=18):
    scale = size / 30.0
    if kind == "coverage":
        verts = [
            (cx, cy - 13 * scale),
            (cx + 12 * scale, cy - 7 * scale),
            (cx + 9 * scale, cy + 10 * scale),
            (cx, cy + 14 * scale),
            (cx - 9 * scale, cy + 10 * scale),
            (cx - 12 * scale, cy - 7 * scale),
        ]
        ax.add_patch(patches.Polygon(verts, closed=True, facecolor=accent, edgecolor=color, lw=1.35, zorder=z))
        ax.plot([cx - 6 * scale, cx - 1 * scale, cx + 8 * scale], [cy + 1 * scale, cy + 6 * scale, cy - 6 * scale], color=color, lw=1.6, solid_capstyle="round", zorder=z + 1)
    elif kind == "prereq":
        nodes = [(cx - 10 * scale, cy), (cx, cy - 9 * scale), (cx + 10 * scale, cy)]
        ax.plot([nodes[0][0], nodes[1][0], nodes[2][0]], [nodes[0][1], nodes[1][1], nodes[2][1]], color=color, lw=1.4, zorder=z)
        for px, py in nodes:
            ax.add_patch(patches.Circle((px, py), 4.0 * scale, facecolor=accent, edgecolor=color, lw=1.0, zorder=z + 1))
        ax.plot([cx - 6 * scale, cx - 2 * scale, cx + 7 * scale], [cy + 10 * scale, cy + 14 * scale, cy + 5 * scale], color=color, lw=1.45, solid_capstyle="round", zorder=z + 1)
    elif kind == "difficulty":
        ax.add_patch(patches.Arc((cx, cy + 4 * scale), 27 * scale, 25 * scale, theta1=200, theta2=-20, color=color, lw=1.45, zorder=z))
        for ang in [210, 250, 290, 330]:
            rad = math.radians(ang)
            ax.plot([cx + 10 * scale * math.cos(rad), cx + 13 * scale * math.cos(rad)], [cy + 4 * scale + 10 * scale * math.sin(rad), cy + 4 * scale + 13 * scale * math.sin(rad)], color=color, lw=1.0, zorder=z)
        ax.plot([cx, cx + 9 * scale], [cy + 4 * scale, cy - 6 * scale], color=color, lw=1.55, zorder=z)
        ax.add_patch(patches.Circle((cx, cy + 4 * scale), 2.7 * scale, facecolor=accent, edgecolor=color, lw=0.9, zorder=z + 1))
    else:
        ax.add_patch(patches.Rectangle((cx - 12 * scale, cy - 11 * scale), 17 * scale, 17 * scale, facecolor=accent, edgecolor=color, lw=1.1, zorder=z))
        ax.add_patch(patches.Rectangle((cx - 4 * scale, cy - 3 * scale), 17 * scale, 17 * scale, facecolor=COL["paper"], edgecolor=color, lw=1.1, zorder=z + 1))
        ax.plot([cx - 10 * scale, cx + 11 * scale], [cy + 13 * scale, cy - 10 * scale], color=COL["red"], lw=1.55, zorder=z + 2)


def reward_calc_tile(ax, x, y, w, h, title, formula, kind, color, accent, inputs=None, z=13):
    box(ax, x, y, w, h, "#fbfaf6", ec=COL["panel"], lw=1.25, r=1, z=z)
    reward_term_icon(ax, x + 15, y + h / 2 + 1, kind, COL["aux"], COL["gray"], size=14, z=z + 5)
    ax.text(
        x + 34,
        y + 17,
        title,
        fontsize=12.0,
        fontweight="bold",
        color=COL["ink"],
        ha="left",
        va="center",
        zorder=z + 6,
    )
    ax.text(
        x + 34,
        y + 39,
        formula,
        fontsize=11.2,
        fontweight="bold",
        color=COL["ink"],
        ha="left",
        va="center",
        zorder=z + 6,
    )


def concept_prereq_family_icon(ax, cx, cy, size=42, color=None, accent=None, z=18):
    color = color or COL["green"]
    accent = accent or COL["green_soft"]
    scale = size / 42.0
    # Concept cluster.
    nodes = [
        (cx - 18 * scale, cy - 10 * scale),
        (cx - 4 * scale, cy - 16 * scale),
        (cx + 8 * scale, cy - 7 * scale),
        (cx - 6 * scale, cy + 2 * scale),
    ]
    for i, j in [(0, 1), (1, 2), (0, 3), (2, 3)]:
        ax.plot([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]], color=color, lw=1.15, zorder=z)
    for px, py in nodes:
        ax.add_patch(patches.Circle((px, py), 3.1 * scale, facecolor=accent, edgecolor=color, lw=0.9, zorder=z + 1))
    # Prerequisite arrow.
    ax.plot([cx - 16 * scale, cx + 13 * scale], [cy + 12 * scale, cy + 12 * scale], color=color, lw=1.45, zorder=z)
    ax.add_patch(
        patches.Polygon(
            [
                (cx + 13 * scale, cy + 12 * scale),
                (cx + 7 * scale, cy + 8 * scale),
                (cx + 7 * scale, cy + 16 * scale),
            ],
            facecolor=color,
            edgecolor=color,
            zorder=z + 1,
        )
    )
    # Family stack.
    for dx, dy in [(7, -18), (12, -13), (17, -8)]:
        ax.add_patch(
            patches.Rectangle(
                (cx + dx * scale, cy + dy * scale),
                11 * scale,
                9 * scale,
                facecolor=COL["paper"],
                edgecolor=color,
                lw=1.0,
                zorder=z + 1,
            )
        )


def cosine_icon(ax, cx, cy, size=34, color=None, z=18):
    color = color or COL["line"]
    scale = size / 34.0
    ax.add_patch(patches.Arc((cx, cy), 26 * scale, 26 * scale, theta1=0, theta2=360, color=COL["panel"], lw=1.0, zorder=z))
    ax.plot([cx, cx + 12 * scale], [cy, cy - 8 * scale], color=color, lw=1.4, zorder=z)
    ax.plot([cx, cx + 14 * scale], [cy, cy + 2 * scale], color=COL["blue"], lw=1.4, zorder=z)
    ax.add_patch(patches.Arc((cx, cy), 12 * scale, 12 * scale, theta1=-8, theta2=34, color=COL["orange"], lw=1.2, zorder=z))
    ax.text(cx - 4 * scale, cy + 10 * scale, r"$\theta$", fontsize=8.5, fontweight="bold", color=COL["muted"], zorder=z + 1)


def recommendation_row(ax, x, y, w, h, rank, course, score, cold=False, z=6):
    fc = "#edf2f5" if cold else "#faf9f4"
    ec = COL["blue"] if cold else COL["panel"]
    ax.add_patch(
        patches.Rectangle(
            (x, y),
            w,
            h,
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.05,
            zorder=z,
        )
    )
    ax.plot([x + 20, x + 20], [y, y + h], color=ec, lw=0.95, zorder=z + 1)
    ax.text(
        x + 10.0,
        y + h / 2 + 0.4,
        str(rank),
        fontsize=11.5,
        fontweight="bold",
        color=COL["blue"] if cold else COL["muted"],
        ha="center",
        va="center",
        zorder=z + 3,
    )
    if cold:
        snowflake_icon(ax, x + 17, y + 7, r=2.3, color=COL["ice_dark"], lw=0.6, z=z + 4)
    ax.text(
        x + 28,
        y + h / 2 + 0.2,
        course,
        fontsize=11.4,
        fontweight="bold",
        color=COL["blue"] if cold else COL["ink"],
        ha="left",
        va="center",
        zorder=z + 3,
    )


def draw_panel_headers(ax):
    txt(ax, 54, 120, "a", size=17.0, weight="bold", ha="left")
    txt(ax, 84, 120, "Cold-course evidence\nencoder", size=13.2, weight="bold", ha="left")
    txt(ax, 446, 120, "b", size=17.0, weight="bold", ha="left")
    txt(ax, 478, 120, "Course-knowledge guided simulation", size=15.0, weight="bold", ha="left")
    txt(ax, 1142, 120, "c", size=17.0, weight="bold", ha="left")
    txt(ax, 1174, 120, "Strict Item-Cold Ranking", size=12.4, weight="bold", ha="left")


def draw_left(ax):
    panel_box(ax, 28, 96, 370, 570, COL["left"], accent=COL["accent_a"])

    arrow_lw = 1.55
    arrow_tail = 0.42
    arrow_head_w = 8.4
    arrow_head_l = 9.4

    txt(ax, 213, 152, "Dropout-style course encoder", size=11.8, weight="bold", color=COL["blue"])
    course_icon(ax, 171, 158, 84, 80, framed=False)

    # Course evidence branches through one trunk before entering both towers.
    course_cx = 213
    branch_y = 282
    tower_y = 300
    tower_h = 84
    left_tower_cx = 136
    right_tower_cx = 288
    ax.plot(
        [course_cx, course_cx],
        [254, branch_y],
        color=COL["flow"],
        lw=arrow_lw,
        solid_capstyle="butt",
        solid_joinstyle="round",
        zorder=11,
    )
    ax.plot(
        [left_tower_cx, right_tower_cx],
        [branch_y, branch_y],
        color=COL["flow"],
        lw=arrow_lw,
        solid_capstyle="butt",
        solid_joinstyle="round",
        zorder=11,
    )
    hollow_arrow(
        ax,
        (left_tower_cx, branch_y),
        (left_tower_cx, tower_y),
        lw=arrow_lw,
        tail=arrow_tail,
        head_w=arrow_head_w,
        head_l=arrow_head_l,
        z=12,
    )
    hollow_arrow(
        ax,
        (right_tower_cx, branch_y),
        (right_tower_cx, tower_y),
        lw=arrow_lw,
        tail=arrow_tail,
        head_w=arrow_head_w,
        head_l=arrow_head_l,
        z=12,
    )

    box(ax, 70, tower_y, 132, tower_h, "#f8f8f2", ec=COL["blue"], lw=1.8, r=0.5)
    txt(ax, 136, tower_y + 20, "side-feature\ntower", size=11.8, weight="bold", color=COL["blue"])
    vector(ax, 97, tower_y + 50, w=78, h=18)
    txt(ax, 136, tower_y + 77, "$x_i \\rightarrow c_i$", size=11.5, weight="bold", color=COL["blue"])

    box(ax, 222, tower_y, 132, tower_h, "#f8f8f2", ec=COL["panel"], lw=1.8, r=0.5)
    txt(ax, 288, tower_y + 20, "ID-factor\ntower", size=11.8, weight="bold", color=COL["muted"])
    vector(ax, 249, tower_y + 50, w=78, h=18, colors=["#ffffff", "#e5e8ec", "#aab4bf", "#66717e"])
    txt(ax, 288, tower_y + 77, "$v_i\\odot m_i$", size=11.5, weight="bold", color=COL["muted"])
    ax.plot([263, 313], [tower_y + 48, tower_y + 68], color=COL["red"], lw=1.8, zorder=16)
    ax.plot([313, 263], [tower_y + 48, tower_y + 68], color=COL["red"], lw=1.8, zorder=16)
    box(ax, 326, 272, 56, 22, COL["red_dark"], ec=COL["red_dark"], lw=1.0, r=0.5, z=14)
    ax.text(
        354,
        283,
        "dropout",
        fontsize=11.0,
        fontweight="bold",
        color=COL["paper"],
        ha="center",
        va="center",
        linespacing=0.92,
        zorder=18,
    )

    fusion_y = 414
    merge_y = 396
    left_fusion_x = 176
    right_fusion_x = 248
    ax.plot(
        [left_tower_cx, left_tower_cx, left_fusion_x],
        [tower_y + tower_h, merge_y, merge_y],
        color=COL["flow"],
        lw=arrow_lw,
        solid_capstyle="butt",
        solid_joinstyle="round",
        zorder=11,
    )
    ax.plot(
        [right_tower_cx, right_tower_cx, right_fusion_x],
        [tower_y + tower_h, merge_y, merge_y],
        color=COL["flow"],
        lw=arrow_lw,
        solid_capstyle="butt",
        solid_joinstyle="round",
        zorder=11,
    )
    hollow_arrow(
        ax,
        (left_fusion_x, merge_y),
        (left_fusion_x, fusion_y),
        lw=arrow_lw,
        tail=arrow_tail,
        head_w=arrow_head_w,
        head_l=arrow_head_l,
        z=12,
    )
    hollow_arrow(
        ax,
        (right_fusion_x, merge_y),
        (right_fusion_x, fusion_y),
        lw=arrow_lw,
        tail=arrow_tail,
        head_w=arrow_head_w,
        head_l=arrow_head_l,
        z=12,
    )

    box(ax, 88, fusion_y, 248, 62, "#f8f8f2", ec=COL["violet"], lw=1.85, r=0.5)
    gate_network_icon(ax, 119, fusion_y + 31, size=30, color=COL["violet"], accent=COL["blue2"], z=18)
    txt(ax, 234, fusion_y + 21, "fusion MLP + gate", size=12.0, weight="bold", color=COL["violet"])
    txt(ax, 234, fusion_y + 47, "$q_i=Gate(c_i,v_i\\odot m_i)$", size=11.5, color=COL["muted"])

    q_vec_y = 504
    hollow_arrow(
        ax,
        (212, fusion_y + 62),
        (212, q_vec_y),
        lw=arrow_lw,
        tail=arrow_tail,
        head_w=arrow_head_w,
        head_l=arrow_head_l,
        z=12,
    )
    vector(ax, 174, q_vec_y, w=76, h=18)
    txt(ax, 212, 540, "$q_i$", size=13.5, weight="bold", color=COL["violet"])

    ax.plot([64, 362], [558, 558], color="#d8dee7", lw=0.95, zorder=9)
    txt(ax, 82, 575, "user-history encoder", size=12.5, weight="bold", ha="left", color=COL["orange"])
    user_icon(ax, 96, 584)
    txt(ax, 146, 596, "history\nonly", size=11.5, weight="bold", color=COL["muted"])
    mlp_wedge(ax, 198, 579, w=58, h=38, ec=COL["orange"], label="MLP", label_size=11.5)
    vector(ax, 290, 590, w=66, h=17, colors=["#ffffff", "#ecd8c9", "#c77a4b", "#8e4628"])
    txt(ax, 323, 637, "$z_u$", size=11.5, weight="bold", color=COL["orange"])
    orth_arrow(ax, [(154, 603), (198, 603)], color=COL["line"], lw=1.15, ms=10)
    orth_arrow(ax, [(256, 603), (290, 603)], color=COL["line"], lw=1.15, ms=10)


def draw_middle(ax):
    panel_box(ax, 420, 96, 675, 570, COL["middle"], accent=COL["accent_b"])

    box(ax, 448, 144, 634, 500, "#c9dbea", ec=COL["blue"], lw=2.15, r=2)
    txt(ax, 765, 178, "T-step learner-course simulator", size=15.0, weight="bold")

    state_capsule(
        ax,
        493,
        270,
        w=46,
        h=106,
        label="$s_t$",
        colors=["#f3f6f9", "#426f95", "#f0eadb"],
        band_labels=["$h_t$", "$q_i$", "$l_t$"],
        z=12,
    )
    box(ax, 476, 244, 80, 19, COL["cream"], ec=COL["gold"], lw=1.25, r=1, z=13)
    txt(ax, 516, 253, "$l_t=T-t$", size=11.5, weight="bold", color=COL["gold"], z=18)

    box(ax, 570, 222, 214, 106, "#f8f8f2", ec=COL["aux"], lw=1.5, r=0.5, ls=(0, (6, 4)))
    txt(ax, 677, 248, "Exploration set\nconstruction", size=11.8, weight="bold")
    box(ax, 588, 276, 52, 38, COL["card2"], ec=COL["green"], lw=1.3, r=0.5)
    txt(ax, 614, 295, "top-M", size=11.5, weight="bold", color=COL["green"])
    box(ax, 654, 276, 62, 38, COL["card2"], ec=COL["gold"], lw=1.3, r=0.5)
    txt(ax, 685, 295, "course\nprior", size=11.0, weight="bold", color=COL["gold"])
    box(ax, 730, 276, 42, 38, COL["red_soft"], ec=COL["red_dark"], lw=1.3, r=0.5)
    txt(ax, 751, 295, "$a_{end}$", size=11.5, weight="bold", color=COL["red"])

    box(ax, 570, 352, 214, 66, "#f8f8f2", ec=COL["line"], lw=1.55, r=0.5)
    brain_circuit_icon(ax, 598, 379, size=30, color=COL["violet"], accent=COL["blue2"], z=18)
    txt(ax, 690, 372, "actor-critic agent", size=12.0, weight="bold")
    txt(ax, 690, 398, "$\\pi_\\theta(a\\mid s_t,S_t)$", size=11.3, color=COL["muted"])

    box(ax, 808, 222, 172, 196, "#f8f8f2", ec=COL["aux"], lw=1.5, r=0.5, ls=(0, (6, 4)))
    txt(ax, 894, 248, "State transition", size=12.0, weight="bold")
    vector(ax, 826, 284, w=48, h=15)
    update_loop_icon(ax, 894, 292, size=28, color=COL["line"], z=18)
    vector(ax, 928, 284, w=38, h=15, colors=["#f3f6f9", "#d1dbe5", "#6f879e", "#284f73"])
    txt(ax, 850, 321, "$h_t$", size=11.5, weight="bold", color=COL["blue"])
    txt(ax, 947, 321, "$h_{t+1}$", size=11.5, weight="bold", color=COL["blue"])
    txt(ax, 894, 354, "$h_{t+1}=\\rho(h_t,a_t,q_i)$", size=12.0, color=COL["muted"])
    txt(ax, 894, 383, "$l_{t+1}=l_t-1$", size=12.0, color=COL["gold"])

    state_capsule(
        ax,
        1000,
        270,
        w=46,
        h=106,
        label="$s_{t+1}$",
        colors=["#f3f6f9", "#b65a5a", "#f0eadb"],
        accent=COL["red"],
        band_labels=["$h_{t+1}$", "$q_i$", "$l_{t+1}$"],
        z=12,
    )

    box(ax, 486, 426, 558, 176, "#f8f8f2", ec=COL["aux"], lw=1.5, r=0.5, ls=(0, (6, 4)))
    txt(ax, 765, 446, "Reward computation", size=12.8, weight="bold")
    reward_calc_tile(
        ax,
        506,
        458,
        158,
        56,
        "target align.",
        "$h_{t+1}\\approx q_i$",
        "coverage",
        COL["rose"],
        COL["rose_soft"],
        "$h_{t+1},q_i$",
    )
    reward_calc_tile(
        ax,
        687,
        458,
        158,
        56,
        "progress",
        "$d_t-d_{t+1}$",
        "difficulty",
        COL["green"],
        COL["green_soft"],
        "$h_t,h_{t+1}$",
    )
    reward_calc_tile(
        ax,
        868,
        458,
        158,
        56,
        "concept",
        "$s_u^TA_i^{con}$",
        "prereq",
        COL["green"],
        COL["green_soft"],
        "$s_u,A_i^{con}$",
    )
    reward_calc_tile(
        ax,
        506,
        524,
        158,
        56,
        "prereq",
        "$A_i^{pre}$ gap",
        "prereq",
        COL["orange"],
        COL["orange_soft"],
        "$s_u,A_i^{pre}$",
    )
    reward_calc_tile(
        ax,
        687,
        524,
        158,
        56,
        "difficulty",
        "$d_i-r_u$ gap",
        "difficulty",
        COL["gold"],
        COL["gold_soft"],
        "$d_i,r_u$",
    )
    reward_calc_tile(
        ax,
        868,
        524,
        158,
        56,
        "repeat",
        "$\\mathbb{1}[a_t\\in H_u]$",
        "repeat",
        COL["muted"],
        "#eef2f6",
        "$a_t,H_u$",
    )
    ax.text(
        765,
        597,
        "$r_t=\\sum_k\\lambda_k r_t^k$",
        fontsize=11.2,
        fontweight="bold",
        color=COL["red"],
        ha="center",
        va="center",
        zorder=18,
    )

    orth_arrow(ax, [(539, 323), (570, 323)], color=COL["blue"], lw=2.0, ms=14)
    orth_arrow(ax, [(671, 324), (671, 350)], color=COL["blue"], lw=2.0, ms=14)
    orth_arrow(ax, [(772, 379), (808, 379)], color=COL["blue"], lw=2.0, ms=14)
    txt(ax, 792, 362, "$a_t$", size=11.0, color=COL["muted"])
    orth_arrow(ax, [(980, 323), (1000, 323)], color=COL["blue"], lw=2.0, ms=14)
    orth_arrow(ax, [(671, 432), (671, 408)], color=COL["orange"], lw=1.5, ms=11, ls=(0, (5, 4)), alpha=0.95)

    ax.plot([1023, 1023, 516, 516], [270, 218, 218, 270], color=COL["line"], lw=1.35, zorder=14)
    ax.add_patch(
        FancyArrowPatch(
            (516, 238),
            (516, 270),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.35,
            color=COL["line"],
            shrinkA=0,
            shrinkB=0,
            zorder=15,
        )
    )


def draw_right(ax):
    panel_box(ax, 1118, 96, 354, 570, COL["right"], accent=COL["accent_c"])

    source_x = 1148
    source_w = 80
    source_right = source_x + source_w
    score_x = 1244
    score_w = 96
    score_right = score_x + score_w
    score_cx = score_x + score_w / 2
    topk_x = 1364
    topk_w = 90
    topk_cx = topk_x + topk_w / 2
    input_elbow_x = score_x - 14

    box(ax, 1140, 150, 318, 60, COL["cream"], ec=COL["panel"], lw=1.45, r=0.5)
    snowflake_icon(ax, 1180, 180, r=6.0, color=COL["ice_dark"], lw=1.0)
    txt(ax, 1299, 176, "strict item-cold inference", size=12.0, weight="bold")
    txt(ax, 1299, 198, "cold-only full catalog", size=11.5, color=COL["muted"])

    box(ax, 1138, 238, 324, 378, "#f8f8f2", ec=COL["line"], lw=1.7, r=0.5, ls=(0, (7, 5)))
    txt(ax, 1158, 268, "dot-product scoring", size=12.0, weight="bold", ha="left")

    user_icon(ax, 1156, 315)
    txt(ax, 1174, 302, "user embedding", size=11.8, weight="bold", color=COL["orange"], ha="left")
    vector(ax, source_x, 324, w=source_w, h=17, colors=["#ffffff", "#ecd8c9", "#c77a4b", "#8e4628"])
    txt(ax, source_x + source_w / 2, 359, "$z_u$", size=11.8, weight="bold", color=COL["orange"])

    snowflake_icon(ax, 1151, 487, r=5.4, color=COL["ice_dark"], lw=0.9)
    book_open_icon(ax, 1168, 487, size=22, color=COL["line"], lw=1.15, z=18)
    txt(ax, 1188, 476, "cold bank", size=11.6, weight="bold", color=COL["blue"], ha="left")
    vector(ax, source_x, 500, w=source_w, h=17)
    txt(ax, source_x + source_w / 2, 535, "$Z_{cold}$", size=11.8, weight="bold", color=COL["blue"])

    box(ax, score_x, 364, score_w, 112, COL["paper"], ec=COL["line"], lw=1.75, r=0.5)
    txt(ax, score_cx, 394, "dot-prod.", size=11.6, weight="bold", color=COL["line"])
    txt(ax, score_cx, 425, "$s(u,i)$", size=12.0, weight="bold", color=COL["ink"])
    txt(ax, score_cx, 455, "$=z_u^{\\mathsf{T}}z_i^{cold}$", size=10.8, weight="bold", color=COL["ink"])

    orth_arrow(
        ax,
        [(source_right, 333), (input_elbow_x, 333), (input_elbow_x, 407), (score_x, 407)],
        color=COL["orange"],
        lw=2.0,
        ms=14,
    )
    orth_arrow(
        ax,
        [(source_right, 509), (input_elbow_x, 509), (input_elbow_x, 449), (score_x, 449)],
        color=COL["blue"],
        lw=2.0,
        ms=14,
    )
    orth_arrow(ax, [(score_right, 421), (topk_x, 421)], color=COL["line"], lw=2.0, ms=14)

    box(ax, topk_x, 316, topk_w, 206, "#faf9f4", ec=COL["panel"], lw=1.55, r=0.5, z=3)
    txt(ax, topk_cx, 340, "Top-K\ncourses", size=11.3, weight="bold", color=COL["ink"])
    recommendation_row(ax, topk_x + 5, 374, topk_w - 10, 31, 1, "C-17", "", cold=True)
    recommendation_row(ax, topk_x + 5, 410, topk_w - 10, 31, 2, "C-04", "", cold=True)
    recommendation_row(ax, topk_x + 5, 446, topk_w - 10, 31, 3, "C-31", "", cold=False)
    recommendation_row(ax, topk_x + 5, 482, topk_w - 10, 31, "K", "C-K", "", cold=False)


def draw_cross_panel_routes(ax):
    red_boundary_arrow(ax, 400, 344, scale=0.78, z=31)
    txt(ax, 434, 325, "$h_0=q_i$", size=8.0, weight="bold", color=COL["muted"], z=30)

    red_boundary_arrow(ax, 1099, 478, scale=0.78, z=31)


def draw_legend(ax):
    box(ax, 414, 684, 706, 50, COL["paper"], ec=COL["panel"], lw=1.4, r=8, z=25)
    y = 703
    ax.plot([438, 482], [y, y], color=COL["line"], lw=2.5, zorder=26)
    txt(ax, 490, y, "black solid = forward computation", size=8.2, weight="bold", ha="left", color=COL["muted"], z=26)
    ax.plot([696, 740], [y, y], color=COL["green"], lw=2.5, zorder=26)
    txt(ax, 748, y, "green = cold-side guidance / retrieval", size=8.2, weight="bold", ha="left", color=COL["muted"], z=26)
    y2 = 722
    ax.plot([624, 668], [y2, y2], color=COL["red"], lw=2.2, linestyle=(0, (6, 4)), zorder=26)
    txt(ax, 676, y2, "red dashed = training optimization only", size=8.2, weight="bold", ha="left", color=COL["muted"], z=26)


def overview_text(ax, x, y, s, size=16, weight="normal", color=None, ha="center", va="center", z=20):
    ax.text(
        x,
        y,
        s,
        fontsize=size * TEXT_SCALE,
        fontweight=weight,
        color=color or COL["ink"],
        ha=ha,
        va=va,
        linespacing=1.02,
        zorder=z,
    )


def overview_box(
    ax,
    x,
    y,
    w,
    h,
    label,
    fc="#fbfbf7",
    ec=None,
    color=None,
    size=17,
    weight="bold",
    ls="-",
    lw=2.15,
    z=5,
):
    patch = patches.Rectangle(
        (x, y),
        w,
        h,
        linewidth=lw,
        edgecolor=ec or COL["line"],
        facecolor=fc,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(patch)
    if label:
        overview_text(ax, x + w / 2, y + h / 2, label, size=size, weight=weight, color=color, z=z + 2)
    return patch


def overview_arrow(ax, pts, color=None, lw=2.75, ms=19, ls="-", z=20):
    color = color or COL["flow"]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if abs(x0 - x1) > 1e-6 and abs(y0 - y1) > 1e-6:
            raise ValueError(f"overview_arrow expects orthogonal segments: {(x0, y0)} -> {(x1, y1)}")
    if len(pts) > 2:
        xs, ys = zip(*pts[:-1])
        ax.plot(xs, ys, color=color, lw=lw, linestyle=ls, solid_capstyle="butt", zorder=z)
    ax.add_patch(
        FancyArrowPatch(
            pts[-2],
            pts[-1],
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            linestyle=ls,
            shrinkA=0,
            shrinkB=0,
            zorder=z + 1,
        )
    )


def overview_vector(ax, x, y, w=106, h=22, colors=None, z=10):
    colors = colors or ["#ffffff", "#dbe7d4", "#8fb395", "#386b4a"]
    step = w / len(colors)
    for i, c in enumerate(colors):
        ax.add_patch(
            patches.Rectangle((x + i * step, y), step, h, facecolor=c, edgecolor=COL["line"], linewidth=1.25, zorder=z)
        )
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="none", edgecolor=COL["line"], linewidth=1.35, zorder=z + 1))


def overview_panel(ax, x, y, w, h, fc, label, title_lines):
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=fc, edgecolor=COL["panel"], linewidth=3.2, zorder=1))
    ax.add_patch(patches.Rectangle((x, y), w, 112, facecolor=fc, edgecolor=COL["panel"], linewidth=0, zorder=2))
    ax.plot([x, x + w], [y + 112, y + 112], color=COL["panel"], lw=2.4, zorder=3)
    overview_text(ax, x + 22, y + 46, f"({label})", size=28, weight="bold", ha="left", z=5)
    for j, line in enumerate(title_lines):
        overview_text(ax, x + 86, y + 28 + j * 32, line, size=25.5, weight="bold", ha="left", z=5)


def course_marker(ax, x, y, w=18, h=15, color=None, z=12):
    color = color or COL["green"]
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="#f4f7f3", edgecolor=COL["line"], linewidth=1.25, zorder=z))
    ax.add_patch(patches.Rectangle((x + 3, y + 3), w - 6, h - 6, facecolor=color, edgecolor="none", alpha=0.55, zorder=z + 1))


def draw_encoder_overview(ax):
    x, y, w, h = 40, 50, 520, 720
    overview_panel(ax, x, y, w, h, "#e4ecd9", "a", ["Cold-course", "Representation", "Encoder"])

    cx = x + w / 2
    book_open_icon(ax, cx - 16, y + 150, size=43, color=COL["line"], lw=1.9, z=10)
    snowflake_icon(ax, cx + 36, y + 128, r=7.2, color=COL["ice_dark"], lw=1.15, z=12)
    overview_text(ax, cx, y + 184, "cold course $i$", size=22, weight="bold")
    overview_text(ax, cx, y + 212, "no ID interaction signal", size=16, color=COL["muted"])

    side = (x + 50, y + 270, 190, 112)
    ident = (x + 280, y + 270, 190, 112)
    overview_box(ax, *side, "", fc="#fbfbf7", ec=COL["green"], lw=2.35, z=8)
    overview_box(ax, *ident, "", fc="#f4f4f0", ec=COL["muted"], lw=2.35, z=8)
    overview_text(ax, side[0] + side[2] / 2, side[1] + 34, "Side-feature\ntower", size=16.2, weight="bold", color=COL["green"])
    overview_text(ax, ident[0] + ident[2] / 2, ident[1] + 34, "Masked ID-factor\ntower", size=14.8, weight="bold", color=COL["muted"])
    overview_vector(ax, side[0] + 43, side[1] + 78, w=104, h=18)
    overview_vector(
        ax,
        ident[0] + 43,
        ident[1] + 78,
        w=104,
        h=18,
        colors=["#ffffff", "#e6e8eb", "#b5bdc7", "#6b7480"],
    )
    overview_box(
        ax,
        ident[0] + 128,
        ident[1] - 30,
        72,
        28,
        "mask",
        fc=COL["red_soft"],
        ec=COL["red_dark"],
        color=COL["red_dark"],
        size=15.5,
        lw=1.9,
    )
    ax.plot([ident[0] + 45, ident[0] + 146], [ident[1] + 88, ident[1] + 88], color=COL["red_dark"], lw=3.0, zorder=20)

    fusion = (x + 108, y + 434, 304, 90)
    overview_box(ax, *fusion, "", fc="#fbfbf7", ec=COL["violet"], lw=2.35, z=8)
    overview_text(ax, fusion[0] + fusion[2] / 2, fusion[1] + 30, "Fusion gate", size=21, weight="bold", color=COL["violet"])
    overview_text(
        ax,
        fusion[0] + fusion[2] / 2,
        fusion[1] + 64,
        "$q_i=Gate(c_i,v_i\\odot m_i)$",
        size=16.5,
        color=COL["muted"],
    )

    overview_vector(ax, cx - 58, y + 566, w=116, h=22)
    overview_text(ax, cx, y + 618, "$q_i$ / $z_i^{cold}$", size=22, weight="bold", color=COL["green"])

    trunk_y = y + 246
    overview_arrow(ax, [(cx, y + 224), (cx, trunk_y), (side[0] + side[2] / 2, trunk_y), (side[0] + side[2] / 2, side[1])], color=COL["flow"])
    overview_arrow(ax, [(cx, trunk_y), (ident[0] + ident[2] / 2, trunk_y), (ident[0] + ident[2] / 2, ident[1])], color=COL["flow"])
    merge_y = y + 404
    overview_arrow(ax, [(side[0] + side[2] / 2, side[1] + side[3]), (side[0] + side[2] / 2, merge_y), (fusion[0] + 92, merge_y), (fusion[0] + 92, fusion[1])], color=COL["flow"])
    overview_arrow(ax, [(ident[0] + ident[2] / 2, ident[1] + ident[3]), (ident[0] + ident[2] / 2, merge_y), (fusion[0] + 212, merge_y), (fusion[0] + 212, fusion[1])], color=COL["flow"])
    overview_arrow(ax, [(cx, fusion[1] + fusion[3]), (cx, y + 566)], color=COL["flow"])

    ax.plot([x + 34, x + w - 34], [y + 642, y + 642], color="#c9d6c0", lw=1.8, zorder=3)
    overview_text(ax, x + 52, y + 670, "User-history encoder", size=18.5, weight="bold", color=COL["orange"], ha="left")
    user_round_icon(ax, x + 72, y + 714, size=32, color=COL["line"], lw=1.7, z=10)
    overview_box(ax, x + 126, y + 688, 104, 52, "history", fc="#fbfbf7", ec=COL["orange"], color=COL["orange"], size=17)
    mlp_wedge(ax, x + 262, y + 684, w=74, h=60, ec=COL["orange"], label="MLP", label_size=17)
    overview_vector(ax, x + 378, y + 706, w=86, h=20, colors=["#ffffff", "#efd8c8", "#c87b4c", "#8f4628"])
    overview_text(ax, x + 422, y + 748, "$z_u$", size=20, weight="bold", color=COL["orange"])
    overview_arrow(ax, [(x + 230, y + 714), (x + 262, y + 714)], color=COL["flow"], lw=2.25, ms=16)
    overview_arrow(ax, [(x + 336, y + 714), (x + 378, y + 714)], color=COL["flow"], lw=2.25, ms=16)


def draw_simulation_overview(ax):
    x, y, w, h = 605, 50, 930, 720
    overview_panel(ax, x, y, w, h, "#dce9f2", "b", ["Knowledge-guided", "Learner-Course", "Simulation"])
    sim = (x + 46, y + 136, w - 92, 574)
    overview_box(ax, *sim, "", fc="#d6e6f1", ec=COL["blue"], lw=2.8, z=2)
    overview_text(ax, x + w / 2, y + 174, "Training-time MDP rollout", size=24, weight="bold")

    state = (x + 70, y + 236, 210, 82)
    policy = (x + 326, y + 222, 216, 110)
    action = (x + 596, y + 244, 132, 70)
    trans = (x + 772, y + 228, 150, 102)
    overview_box(ax, *state, "$s_t=[h_t,q_i,l_t]$", fc="#fbfbf7", ec=COL["blue"], color=COL["blue"], size=17.2)
    overview_box(ax, *policy, "Actor-critic\npolicy\n$\\pi_\\theta(a_t\\mid s_t)$", fc="#fbfbf7", ec=COL["line"], size=16.4)
    overview_box(ax, *action, "selected\ncourse $a_t$", fc="#fff8ee", ec=COL["orange"], color=COL["orange"], size=17)
    overview_box(ax, *trans, "$s_{t+1}$\n$=\\rho(s_t,a_t,q_i)$", fc="#fbfbf7", ec=COL["blue"], color=COL["blue"], size=13.8)
    flow_y = state[1] + state[3] / 2
    overview_arrow(ax, [(state[0] + state[2], flow_y), (policy[0], flow_y)], color=COL["blue"])
    overview_arrow(ax, [(policy[0] + policy[2], flow_y), (action[0], flow_y)], color=COL["blue"])
    overview_arrow(ax, [(action[0] + action[2], flow_y), (trans[0], flow_y)], color=COL["blue"])

    reward = (x + 96, y + 430, w - 192, 252)
    overview_box(ax, *reward, "", fc="#fbfbf7", ec=COL["orange"], ls=(0, (9, 5)), lw=2.6, z=4)
    overview_text(ax, x + w / 2, reward[1] + 32, "Knowledge-guided reward signals", size=21, weight="bold", color=COL["orange"])
    labels = [
        "target\nalignment",
        "learning\nprogress",
        "concept\ncoverage",
        "prerequisite\nconsistency",
        "difficulty\ngap",
        "repeat\npenalty",
    ]
    chip_w, chip_h = 212, 62
    sx, sy = reward[0] + 36, reward[1] + 72
    for r in range(2):
        for c in range(3):
            title = labels[r * 3 + c]
            bx, by = sx + c * 240, sy + r * 76
            overview_box(ax, bx, by, chip_w, chip_h, "", fc="#fffaf0", ec=COL["orange"], lw=1.85, z=8)
            overview_text(ax, bx + chip_w / 2, by + chip_h / 2, title, size=15.2, weight="bold", color=COL["ink"])
    overview_text(ax, x + w / 2, reward[1] + reward[3] - 16, "$r_t=\\sum_k\\lambda_k r_t^k$", size=16.8, weight="bold", color=COL["red_dark"])
    overview_arrow(
        ax,
        [(policy[0] + policy[2] / 2, reward[1]), (policy[0] + policy[2] / 2, policy[1] + policy[3])],
        color=COL["orange"],
        lw=2.05,
        ms=16,
        ls=(0, (6, 4)),
    )


def draw_ranking_overview(ax):
    x, y, w, h = 1580, 50, 560, 720
    overview_panel(ax, x, y, w, h, "#e7e3ef", "c", ["Strict Item-Cold", "Ranking", "Inference"])
    overview_box(
        ax,
        x + 44,
        y + 138,
        w - 88,
        74,
        "Strict Item-Cold Inference\ncold-only full catalog",
        fc=COL["cream"],
        ec=COL["panel"],
        size=18,
        lw=2.5,
    )
    area = (x + 38, y + 246, w - 58, 392)
    overview_box(ax, *area, "", fc="#fbfbf7", ec=COL["line"], ls=(0, (9, 6)), lw=2.6, z=2)

    user_y = y + 324
    cold_y = y + 484
    input_x = x + 86
    overview_text(ax, input_x + 76, user_y - 44, "user\nembedding", size=14.8, weight="bold", color=COL["orange"])
    user_round_icon(ax, input_x - 22, user_y, size=34, color=COL["line"], lw=1.6, z=10)
    overview_vector(ax, input_x + 18, user_y - 12, w=116, h=22, colors=["#ffffff", "#efd8c8", "#c87b4c", "#8f4628"])
    overview_text(ax, input_x + 76, user_y + 42, "$z_u$", size=19, weight="bold", color=COL["orange"])

    snowflake_icon(ax, input_x - 34, cold_y, r=6.5, color=COL["ice_dark"], lw=1.0, z=10)
    book_open_icon(ax, input_x - 4, cold_y, size=26, color=COL["line"], lw=1.25, z=10)
    overview_text(ax, input_x + 78, cold_y - 56, "cold-course\nbank", size=14.8, weight="bold", color=COL["blue"])
    overview_vector(ax, input_x + 18, cold_y - 12, w=116, h=22)
    overview_text(ax, input_x + 78, cold_y + 48, "$Z_{cold}$", size=19, weight="bold", color=COL["blue"])

    score = (x + 240, y + 338, 154, 130)
    overview_box(ax, *score, "", fc=COL["paper"], ec=COL["line"], lw=2.35, z=8)
    overview_text(ax, score[0] + score[2] / 2, score[1] + 32, "dot-product\nscoring", size=14.6, weight="bold")
    overview_text(ax, score[0] + score[2] / 2, score[1] + 82, "$s(u,i)$", size=17.2, weight="bold")
    overview_text(ax, score[0] + score[2] / 2, score[1] + 108, "$=z_u^Tz_i^{cold}$", size=14.2, weight="bold")

    topk = (x + 426, y + 326, 108, 154)
    overview_box(ax, *topk, "", fc="#faf9f4", ec=COL["panel"], lw=2.3, z=8)
    overview_text(ax, topk[0] + topk[2] / 2, topk[1] + 24, "Top-K", size=16.2, weight="bold")
    rows = [(1, "C-17"), (2, "C-04"), (3, "C-31"), ("K", "C-K")]
    for j, (rank, course) in enumerate(rows):
        yy = topk[1] + 46 + j * 25
        ax.add_patch(
            patches.Rectangle(
                (topk[0] + 7, yy),
                topk[2] - 14,
                21,
                facecolor="#f4f7f7",
                edgecolor=COL["blue"] if j < 2 else COL["panel"],
                linewidth=1.25,
                zorder=9,
            )
        )
        course_marker(ax, topk[0] + 12, yy + 4, w=11, h=11, z=10)
        overview_text(ax, topk[0] + 34, yy + 11, str(rank), size=14.2, weight="bold", color=COL["blue"] if j < 2 else COL["muted"])
        overview_text(ax, topk[0] + 75, yy + 11, course, size=14.2, weight="bold", color=COL["blue"] if j < 2 else COL["ink"])

    elbow_x = score[0] - 42
    overview_arrow(ax, [(input_x + 134, user_y), (elbow_x, user_y), (elbow_x, score[1] + 45), (score[0], score[1] + 45)], color=COL["orange"])
    overview_arrow(ax, [(input_x + 134, cold_y), (elbow_x, cold_y), (elbow_x, score[1] + 92), (score[0], score[1] + 92)], color=COL["blue"])
    overview_arrow(ax, [(score[0] + score[2], score[1] + score[3] / 2), (topk[0], score[1] + score[3] / 2)], color=COL["flow"])


def draw_overview_routes(ax):
    red_boundary_arrow(ax, 582, 430, scale=0.88, z=31)
    overview_text(ax, 582, 390, "$q_i$", size=17, weight="bold", color=COL["green"], z=32)
    red_boundary_arrow(ax, 1560, 430, scale=0.88, z=31)


def draw_figure():
    fig = plt.figure(figsize=(12.0, 5.3), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.add_patch(patches.Rectangle((0, 0), W, H, facecolor=COL["paper"], edgecolor="none", zorder=0))

    draw_encoder_overview(ax)
    draw_simulation_overview(ax)
    draw_ranking_overview(ax)
    draw_overview_routes(ax)
    return fig


def save(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in [
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 300}),
        (".tiff", {"dpi": 600}),
    ]:
        final = base.with_suffix(suffix)
        tmp = base.with_name(base.name + "_tmp").with_suffix(suffix)
        if tmp.exists():
            tmp.unlink()
        fig.savefig(tmp, bbox_inches="tight", pad_inches=0.035, **kwargs)
        if final.exists():
            final.unlink()
        tmp.replace(final)


def main():
    out = Path(__file__).resolve().parent / "ckg_rl_framework_topconf"
    fig = draw_figure()
    save(fig, out)
    plt.close(fig)
    for suffix in [".svg", ".pdf", ".png", ".tiff"]:
        print(f"saved: {out.with_suffix(suffix)}")


if __name__ == "__main__":
    main()
