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
TEXT_SCALE = 1.34
MIN_TEXT_SIZE = 10.6
BOX_GEOM_SCALE = 1.06
PANEL_GEOM_SCALE = 1.025
FRAME_LINE_SCALE = 1.42
VECTOR_LINE_SCALE = 1.30

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
    "teal": "#2d6f67",
    "teal_soft": "#dceee8",
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

ARROW_STYLES = {
    "encoder": {"color": COL["flow"], "lw": 1.65, "ms": 12.0, "alpha": 1.0},
    "data": {"color": COL["flow"], "lw": 1.65, "ms": 12.0, "alpha": 1.0},
    "micro": {"color": COL["muted"], "lw": 1.0, "ms": 8.8, "alpha": 0.95},
    "action": {"color": COL["orange"], "lw": 1.55, "ms": 11.0, "alpha": 1.0},
    "reward": {"color": COL["orange"], "lw": 1.55, "ms": 11.0, "alpha": 1.0},
    "aux": {"color": COL["aux"], "lw": 1.1, "ms": 9.5, "ls": (0, (4.5, 3.2)), "alpha": 0.98},
    "update": {"color": COL["red"], "lw": 1.55, "ms": 11.0, "alpha": 1.0},
    "rank_user": {"color": COL["orange"], "lw": 1.55, "ms": 11.0, "alpha": 1.0},
    "rank_course": {"color": COL["red"], "lw": 1.55, "ms": 11.0, "alpha": 1.0},
    "rank_out": {"color": COL["flow"], "lw": 1.55, "ms": 11.0, "alpha": 1.0},
}


def font_size(size, floor=None):
    size = size * TEXT_SCALE
    return max(size, floor) if floor is not None else size


def grow_rect(x, y, w, h, scale):
    if scale == 1:
        return x, y, w, h
    dw = w * (scale - 1)
    dh = h * (scale - 1)
    return x - dw / 2, y - dh / 2, w + dw, h + dh


def stroke_width(lw, floor=1.05):
    return max(lw * FRAME_LINE_SCALE, floor)


def txt(ax, x, y, s, size=10, weight="normal", color=None, ha="center", va="center", z=20):
    size = font_size(size, floor=MIN_TEXT_SIZE)
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
    x, y, w, h = grow_rect(x, y, w, h, BOX_GEOM_SCALE)
    lw = stroke_width(lw, floor=1.2)
    r = min(max(r * 1.16, 0), 1.7)
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


def mini_clock(ax, cx, cy, r=6.0, color=None, lw=0.95, z=20, fc=None):
    color = color or COL["blue"]
    ax.add_patch(
        patches.Circle(
            (cx, cy),
            r,
            facecolor=fc or COL["paper"],
            edgecolor=color,
            lw=lw,
            zorder=z,
        )
    )
    ax.plot([cx, cx], [cy - r * 0.55, cy - r * 0.12], color=COL["line"], lw=lw * 0.8, zorder=z + 1)
    ax.plot([cx, cx + r * 0.45], [cy, cy - r * 0.28], color=COL["line"], lw=lw * 0.8, zorder=z + 1)


def time_chip(ax, x, y, w, h, label, accent=None, z=20):
    accent = accent or COL["blue"]
    box(ax, x, y, w, h, COL["paper"], ec=accent, lw=1.05, r=0.5, ls=(0, (5, 3)), z=z)
    mini_clock(ax, x + 12, y + h / 2, r=5.6, color=accent, z=z + 2)
    txt(ax, x + w / 2 + 9, y + h / 2, label, size=8.2, weight="bold", color=accent, z=z + 3)


def panel_box(ax, x, y, w, h, fc, accent=None):
    x, y, w, h = grow_rect(x, y, w, h, PANEL_GEOM_SCALE)
    patch = patches.Rectangle(
        (x, y),
        w,
        h,
        linewidth=stroke_width(2.25, floor=2.8),
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


def arrow_style(kind, **overrides):
    style = dict(ARROW_STYLES[kind])
    style.update(overrides)
    return style


def flow_arrow(ax, pts, kind="data", **overrides):
    return orth_arrow(ax, pts, **arrow_style(kind, **overrides))


def curved_arrow(ax, start, end, kind="aux", connectionstyle="arc3,rad=0.0", z=12, **overrides):
    style = arrow_style(kind, **overrides)
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        connectionstyle=connectionstyle,
        mutation_scale=style["ms"],
        linewidth=style["lw"],
        linestyle=style.get("ls", "-"),
        color=style["color"],
        shrinkA=0,
        shrinkB=0,
        alpha=style.get("alpha", 1.0),
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
    feedback_color = COL["red_dark"]
    ax.plot([x0, x1], [y, y], color=feedback_color, linewidth=1.45, linestyle=ls, alpha=0.78, zorder=27)
    for x in [x0, 740, x1]:
        ax.plot([x, x], [y, 96], color=feedback_color, linewidth=1.45, linestyle=ls, alpha=0.78, zorder=27)
    for x in [330, 720, 1080]:
        ax.add_patch(
            FancyArrowPatch(
                (x + 56, y - 11),
                (x + 8, y - 11),
                arrowstyle="->",
                mutation_scale=13,
                linewidth=1.25,
                color=feedback_color,
                alpha=0.78,
                zorder=28,
            )
        )
        ax.add_patch(
            FancyArrowPatch(
                (x + 56, y + 2),
                (x + 8, y + 2),
                arrowstyle="->",
                mutation_scale=13,
                linewidth=1.25,
                color=feedback_color,
                alpha=0.78,
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
        color=feedback_color,
        ha="left",
        z=29,
    )


COURSE_EMB_COLORS = ["#ffffff", "#dfe6ee", "#8ea7bd", "#2f5f89"]
CONTENT_EMB_COLORS = ["#ffffff", "#dfece8", "#8fb4ad", "#2d6f67"]
BEHAVIOR_EMB_COLORS = ["#ffffff", "#e5e8ec", "#aab4bf", "#66717e"]
USER_EMB_COLORS = ["#ffffff", "#ecd8c9", "#c77a4b", "#8e4628"]
STATE_EMB_COLORS = ["#f3f6f9", "#d8e5ee", "#8fb0c9", "#426f95"]
NEXT_STATE_EMB_COLORS = ["#f3f6f9", "#f5d5d2", "#d98582", "#b65a5a"]
PROGRESS_EMB_COLORS = ["#f3f6f9", "#d9e8dd", "#86b88c", "#3a7548"]
TRANSITION_NEXT_EMB_COLORS = ["#ffe2df", "#f2aaa4", "#de7476", "#c95562"]


def vector(ax, x, y, w=70, h=17, colors=None, z=8):
    colors = colors or COURSE_EMB_COLORS
    n = len(colors)
    clip = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=1.1",
        linewidth=0,
        facecolor="none",
        edgecolor="none",
        zorder=z,
    )
    ax.add_patch(clip)
    for i, c in enumerate(colors):
        segment = patches.Rectangle(
            (x + i * w / n, y),
            w / n,
            h,
            facecolor=c,
            edgecolor="none",
            linewidth=0,
            zorder=z,
        )
        segment.set_clip_path(clip)
        ax.add_patch(segment)
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=1.1",
            facecolor="none",
            edgecolor=COL["line"],
            linewidth=1.15 * VECTOR_LINE_SCALE,
            zorder=z + 2,
        )
    )


def vertical_embedding(ax, x, y, w=22, h=102, colors=None, z=8):
    colors = colors or COURSE_EMB_COLORS
    n = len(colors)
    clip = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=2.0",
        linewidth=0,
        facecolor="none",
        edgecolor="none",
        zorder=z,
    )
    ax.add_patch(clip)
    slot_h = h / n
    for i, c in enumerate(colors):
        segment = patches.Rectangle(
            (x, y + i * slot_h),
            w,
            slot_h + 0.15,
            facecolor=c,
            edgecolor="none",
            linewidth=0,
            zorder=z,
        )
        segment.set_clip_path(clip)
        ax.add_patch(segment)
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=2.0",
            facecolor="none",
            edgecolor=COL["line"],
            linewidth=1.15 * VECTOR_LINE_SCALE,
            zorder=z + 2,
        )
    )


def tilted_vector(ax, cx, cy, w=66, h=16, colors=None, angle=-45, z=12):
    colors = colors or STATE_EMB_COLORS
    n = len(colors)
    trans = mpl.transforms.Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData
    x0 = cx - w / 2
    y0 = cy - h / 2
    for i, c in enumerate(colors):
        ax.add_patch(
            patches.Rectangle(
                (x0 + i * w / n, y0),
                w / n,
                h,
                facecolor=c,
                edgecolor="none",
                transform=trans,
                zorder=z,
            )
        )
    ax.add_patch(
        patches.FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=3",
            facecolor="none",
            edgecolor=COL["line"],
            linewidth=1.0 * VECTOR_LINE_SCALE,
            transform=trans,
            zorder=z + 1,
        )
    )


def state_capsule(ax, x, y, w=42, h=110, label="$s_t$", colors=None, accent=None, band_labels=None, h_label_side="left", z=9):
    colors = colors or ["#f4f6f8", "#cfd8e3", "#6f879e", "#284f73"]
    accent = accent or COL["blue"]
    cap = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=2.2",
        linewidth=stroke_width(1.75, floor=2.1),
        edgecolor=accent,
        facecolor=COL["paper"],
        zorder=z,
    )
    ax.add_patch(cap)
    h_label = band_labels[0] if band_labels else "$\\mathbf{h}_t$"
    tau_label = band_labels[1] if band_labels and len(band_labels) > 1 else "$\\tau_t$"
    vec_w = min(26, w - 18)
    vec_h = h - 64
    vec_x = x + (w - vec_w) / 2
    vec_y = y + 12
    slot_h = vec_h / len(colors)
    for i, c in enumerate(reversed(colors)):
        ax.add_patch(
            patches.Rectangle(
                (vec_x, vec_y + i * slot_h),
                vec_w,
                slot_h + 0.2,
                facecolor=c,
                edgecolor="none",
                zorder=z + 1,
            )
        )
    ax.add_patch(
        patches.FancyBboxPatch(
            (vec_x, vec_y),
            vec_w,
            vec_h,
            boxstyle="round,pad=0.02,rounding_size=2.0",
            linewidth=1.0 * VECTOR_LINE_SCALE,
            edgecolor=COL["line"],
            facecolor="none",
            zorder=z + 2,
        )
    )
    is_next_state = "t+1" in h_label
    h_label_x = x + w + 8 if h_label_side == "right" else x - (2 if is_next_state else 8)
    ax.text(
        h_label_x,
        vec_y + vec_h / 2 - 18,
        h_label,
        fontsize=font_size(10.1 if "t+1" in h_label else 10.8),
        fontweight="bold",
        color=COL["ink"],
        ha="left" if h_label_side == "right" else "right",
        va="center",
        zorder=z + 3,
    )
    clock_r = 10.4
    clock_c = (x + w / 2, y + h - 28.5)
    ax.add_patch(
        patches.Circle(
            clock_c,
            clock_r,
            facecolor=COL["paper"],
            edgecolor=accent,
            lw=1.15 * VECTOR_LINE_SCALE,
            zorder=z + 3,
        )
    )
    ax.plot([clock_c[0], clock_c[0]], [clock_c[1] - 5.2, clock_c[1] - 1.0], color=COL["line"], lw=0.9, zorder=z + 4)
    ax.plot([clock_c[0], clock_c[0] + 4.2], [clock_c[1], clock_c[1] - 2.8], color=COL["line"], lw=0.9, zorder=z + 4)
    ax.text(
        clock_c[0],
        y + h - 6.4,
        tau_label,
        fontsize=font_size(7.0 if "t+1" in tau_label else 7.4),
        fontweight="bold",
        color=accent,
        ha="center",
        va="center",
        zorder=z + 5,
    )
    label_offset = 13 if h > 140 else 19
    txt(ax, x + w / 2, y + h + label_offset, label, size=15.0, weight="bold", color=accent)
    return cap


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


def mlp_vertical_wedge(ax, x, y, w=62, h=30, ec=None, label="MLP", label_size=10.0):
    inset = min(10, w * 0.18)
    poly = patches.Polygon(
        [[x, y], [x + w, y], [x + w - inset, y + h], [x + inset, y + h]],
        closed=True,
        facecolor=COL["paper"],
        edgecolor=ec or COL["blue"],
        linewidth=2.2,
        zorder=8,
    )
    ax.add_patch(poly)
    txt(ax, x + w / 2, y + h / 2 + 1, label, size=label_size, weight="bold")


def course_icon(ax, x, y, w=98, h=90, framed=True):
    if framed:
        box(ax, x, y, w, h, COL["card"], lw=1.4, r=7)
    snowflake_icon(ax, x + w - 22, y + 18, r=7.2, color=COL["ice_dark"], lw=1.05, z=11)
    book_open_icon(ax, x + w / 2, y + 43, size=52, color=COL["line"], lw=1.65, z=10)
    txt(ax, x + w / 2, y + 73, "course c", size=12.2, weight="bold")


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
    ax.text(cx - 2 * scale, cy + 1 * scale, "g", fontsize=font_size(11), fontweight="bold", color=color, ha="center", va="center", zorder=z + 2)


def gate_fusion_circuit(ax, x, y, w=320, h=66, left_x=None, right_x=None, out_x=None, z=18):
    left_x = left_x if left_x is not None else x + 0.28 * w
    right_x = right_x if right_x is not None else x + 0.72 * w
    out_x = out_x if out_x is not None else x + 0.50 * w
    line = COL["line"]
    side_pad = 24
    gate_w, gate_h = right_x - left_x + 2 * side_pad, 22
    gate_y = y + 24
    gate_x = left_x - side_pad
    gate_cy = gate_y + gate_h / 2

    box(ax, gate_x, gate_y, gate_w, gate_h, COL["paper"], ec=line, lw=1.12, r=0.25, z=z)
    txt(ax, out_x, gate_cy, "$\\sigma$ gate", size=8.2, weight="bold", color=line, z=z + 2)
    txt(ax, gate_x + gate_w + 34, gate_cy, "$\\mathbf{g}_c$", size=8.0, weight="bold", color=COL["muted"], z=z + 2)


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
    ax.text(cx, cy + 1 * scale, r"$\nabla$", fontsize=font_size(11), fontweight="bold", color=color, ha="center", va="center", zorder=z + 1)


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
    if kind == "target":
        for rr, lw in [(12, 1.35), (7, 1.15)]:
            ax.add_patch(patches.Circle((cx, cy), rr * scale, facecolor="none", edgecolor=color, lw=lw, zorder=z))
        ax.add_patch(patches.Circle((cx, cy), 2.6 * scale, facecolor=accent, edgecolor=color, lw=0.8, zorder=z + 1))
        ax.plot([cx + 4 * scale, cx + 14 * scale], [cy - 4 * scale, cy - 14 * scale], color=color, lw=1.4, zorder=z + 1)
        ax.add_patch(
            patches.Polygon(
                [(cx + 14 * scale, cy - 14 * scale), (cx + 8 * scale, cy - 13 * scale), (cx + 13 * scale, cy - 8 * scale)],
                facecolor=color,
                edgecolor=color,
                zorder=z + 2,
            )
        )
    elif kind == "progress":
        pts = [(cx - 12 * scale, cy + 9 * scale), (cx - 4 * scale, cy + 3 * scale), (cx + 3 * scale, cy + 5 * scale), (cx + 12 * scale, cy - 8 * scale)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=1.55, solid_capstyle="round", zorder=z)
        ax.add_patch(
            patches.Polygon(
                [(cx + 12 * scale, cy - 8 * scale), (cx + 5 * scale, cy - 7 * scale), (cx + 10 * scale, cy - 1 * scale)],
                facecolor=color,
                edgecolor=color,
                zorder=z + 1,
            )
        )
        for px, py in pts[:-1]:
            ax.add_patch(patches.Circle((px, py), 2.0 * scale, facecolor=accent, edgecolor=color, lw=0.8, zorder=z + 1))
    elif kind == "concept":
        nodes = [(cx - 10 * scale, cy - 7 * scale), (cx + 2 * scale, cy - 11 * scale), (cx + 11 * scale, cy), (cx - 3 * scale, cy + 9 * scale)]
        for i, j in [(0, 1), (1, 2), (0, 3), (2, 3)]:
            ax.plot([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]], color=color, lw=1.15, zorder=z)
        for px, py in nodes:
            ax.add_patch(patches.Circle((px, py), 3.4 * scale, facecolor=accent, edgecolor=color, lw=0.9, zorder=z + 1))
    elif kind == "coverage":
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
        steps = [(cx - 13 * scale, cy + 8 * scale), (cx - 3 * scale, cy + 1 * scale), (cx + 8 * scale, cy - 6 * scale)]
        for px, py in steps:
            ax.add_patch(patches.Rectangle((px - 4 * scale, py - 4 * scale), 8 * scale, 8 * scale, facecolor=accent, edgecolor=color, lw=0.95, zorder=z))
        ax.plot([steps[0][0] + 4 * scale, steps[1][0] - 4 * scale, steps[1][0] + 4 * scale, steps[2][0] - 4 * scale], [steps[0][1], steps[1][1], steps[1][1], steps[2][1]], color=color, lw=1.35, zorder=z + 1)
        ax.add_patch(patches.Polygon([(cx + 8 * scale, cy - 12 * scale), (cx + 3 * scale, cy - 6 * scale), (cx + 11 * scale, cy - 4 * scale)], facecolor=color, edgecolor=color, zorder=z + 2))
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
    box(ax, x, y, w, h, "#fbfaf6", ec=color, lw=1.25, r=1, z=z)
    ax.add_patch(patches.Rectangle((x + 1.5, y + 1.5), 4.2, h - 3, facecolor=accent, edgecolor="none", zorder=z + 1))
    reward_term_icon(ax, x + 23, y + h / 2 + 1, kind, color, accent, size=24, z=z + 5)
    ax.text(
        x + 48,
        y + 18,
        title,
        fontsize=font_size(10.8),
        fontweight="bold",
        color=COL["ink"],
        ha="left",
        va="center",
        zorder=z + 6,
    )
    ax.text(
        x + 48,
        y + 39,
        formula,
        fontsize=font_size(7.6),
        fontweight="normal",
        color=COL["muted"],
        ha="left",
        va="center",
        zorder=z + 6,
    )


def reward_mini_chip(ax, x, y, w, h, title, kind, color, accent, z=14):
    box(ax, x, y, w, h, "#fbfaf6", ec=color, lw=1.15, r=0.9, z=z)
    reward_term_icon(ax, x + 20, y + h / 2 + 0.5, kind, color, accent, size=17, z=z + 4)
    ax.text(
        x + 42,
        y + h / 2,
        title,
        fontsize=font_size(8.4),
        fontweight="bold",
        color=COL["ink"],
        ha="left",
        va="center",
        zorder=z + 5,
    )


def reward_output_badge(ax, cx, cy, r=28, z=16):
    ax.add_patch(
        patches.Circle(
            (cx, cy),
            r,
            facecolor="#fff7f2",
            edgecolor=COL["red_dark"],
            linewidth=1.55,
            zorder=z,
        )
    )
    trophy_icon(ax, cx, cy - 5, size=24, color=COL["red_dark"], accent=COL["red_soft"], z=z + 3)
    ax.text(cx, cy + 19, "feedback", fontsize=font_size(7.8), fontweight="bold", color=COL["red_dark"], ha="center", va="center", zorder=z + 4)


def reward_expanded_view(ax, x, y, w, h, z=13):
    box(ax, x, y, w, h, "#fbfaf6", ec=COL["rose"], lw=1.35, r=1, z=z)
    ax.text(
        x + 14,
        y + 16,
        "Expanded reward view",
        fontsize=font_size(9.2),
        fontweight="bold",
        color=COL["ink"],
        ha="left",
        va="center",
        zorder=z + 4,
    )

    lane_x = x + 22
    top_y = y + 45
    bot_y = y + 86

    book_open_icon(ax, lane_x + 3, top_y + 1, size=20, color=COL["line"], lw=1.0, z=z + 5)
    vector(ax, lane_x + 24, top_y - 8, w=50, h=14, z=z + 3)
    ax.text(lane_x + 49, top_y + 20, "$\\mathbf{e}_c$", fontsize=font_size(8.5), fontweight="bold", color=COL["blue"], ha="center", va="center", zorder=z + 5)

    vector(ax, lane_x + 24, bot_y - 8, w=50, h=14, colors=STATE_EMB_COLORS, z=z + 3)
    ax.text(lane_x + 49, bot_y + 20, "$\\mathbf{h}_{t+1}$", fontsize=font_size(8.2), fontweight="bold", color=COL["blue"], ha="center", va="center", zorder=z + 5)

    match_x = x + 151
    match_y = y + 65
    ax.plot([lane_x + 78, match_x - 25], [top_y - 1, top_y - 1], color=COL["rose"], lw=1.25, zorder=z + 3)
    ax.plot([lane_x + 78, match_x - 25], [bot_y - 1, bot_y - 1], color=COL["rose"], lw=1.25, zorder=z + 3)
    ax.plot([match_x - 25, match_x - 25], [top_y - 1, bot_y - 1], color=COL["rose"], lw=1.25, zorder=z + 3)
    flow_arrow(ax, [(match_x - 25, match_y), (match_x - 2, match_y)], kind="micro", color=COL["rose"], z=z + 4)
    reward_term_icon(ax, match_x + 18, match_y, "target", COL["rose"], COL["rose_soft"], size=30, z=z + 5)
    ax.text(match_x + 55, match_y - 9, "target", fontsize=font_size(8.6), fontweight="bold", color=COL["ink"], ha="left", va="center", zorder=z + 6)
    ax.text(match_x + 55, match_y + 9, "match", fontsize=font_size(8.6), fontweight="bold", color=COL["ink"], ha="left", va="center", zorder=z + 6)

    gain_y = y + h - 24
    vector(ax, x + 205, gain_y - 8, w=42, h=13, colors=PROGRESS_EMB_COLORS, z=z + 3)
    ax.text(x + 226, gain_y + 17, "$\\mathbf{h}_t$", fontsize=font_size(8.0), fontweight="bold", color=COL["green"], ha="center", va="center", zorder=z + 5)
    flow_arrow(ax, [(x + 249, gain_y - 1), (x + 281, gain_y - 1)], kind="micro", color=COL["green"], z=z + 4)
    reward_term_icon(ax, x + 300, gain_y - 1, "progress", COL["green"], COL["green_soft"], size=24, z=z + 5)
    ax.text(x + 321, gain_y - 1, "progress gain", fontsize=font_size(8.4), fontweight="bold", color=COL["ink"], ha="left", va="center", zorder=z + 6)


def draw_state_transition_function(ax, x, y, w, h, z=13):
    box(ax, x, y, w, h, "#f8f8f2", ec=COL["aux"], lw=1.5, r=0.5, ls=(0, (6, 4)), z=z)
    txt(ax, x + w / 2, y + 24, "State Transition Function", size=10.5, weight="bold")

    flow_cx = x + w / 2
    flow_gap = 58
    left_c = (flow_cx - flow_gap, y + 74)
    right_c = (flow_cx + flow_gap, y + 74)
    tilted_vector(ax, *left_c, w=63, h=16, angle=-48, z=z + 2)
    tilted_vector(
        ax,
        *right_c,
        w=63,
        h=16,
        colors=TRANSITION_NEXT_EMB_COLORS,
        angle=-48,
        z=z + 2,
    )
    flow_arrow(ax, [(left_c[0] + 34, y + 74), (right_c[0] - 34, y + 74)], kind="update", z=z + 4)

    guide_c = (flow_cx, y + 50)
    ax.add_patch(patches.Circle(guide_c, 10, facecolor=COL["orange_soft"], edgecolor=COL["orange"], lw=1.0, zorder=z + 5))
    ax.text(guide_c[0], guide_c[1], "$u_j$", fontsize=font_size(7.8), fontweight="bold", color=COL["orange"], ha="center", va="center", zorder=z + 6)
    curved_arrow(
        ax,
        (left_c[0] + 27, y + 52),
        (guide_c[0] - 13, y + 50),
        kind="action",
        connectionstyle="arc3,rad=-0.42",
        z=z + 4,
        lw=1.1,
        ms=9.2,
        alpha=0.95,
    )
    curved_arrow(
        ax,
        (guide_c[0] + 13, y + 50),
        (right_c[0] + 11, y + 52),
        kind="action",
        connectionstyle="arc3,rad=-0.42",
        z=z + 4,
        lw=1.1,
        ms=9.2,
        alpha=0.95,
    )

    txt(ax, left_c[0] - 4, y + 112, "$\\mathbf{h}_t$", size=10.4, weight="bold", color=COL["blue"])
    txt(ax, right_c[0], y + 112, "$\\mathbf{h}_{t+1}$", size=10.0, weight="bold", color=COL["red_dark"])
    txt(ax, x + w / 2, y + h - 33, "optimize with backpropagation", size=8.1, color=COL["muted"])
    txt(ax, x + w / 2, y + h - 12, "$u_j$ guides the update", size=8.2, weight="bold", color=COL["gold"])


def alignment_reward_panel(ax, x, y, w, h, z=14):
    box(ax, x, y, w, h, COL["paper"], ec=COL["rose"], lw=1.25, r=1, z=z)
    ax.text(x + w / 2, y + 15, "Emb. Alignment Reward", fontsize=font_size(9.2), fontweight="bold", color=COL["ink"], ha="center", va="center", zorder=z + 5)

    c1 = (x + 58, y + 67)
    c2 = (x + 174, y + 67)
    for c, ls in [(c1, "-"), (c2, (0, (4, 3)))]:
        ax.add_patch(patches.Circle(c, 34, facecolor="#fbfbf7", edgecolor=COL["blue"], lw=1.0, linestyle=ls, zorder=z + 1))
    pts1 = {"$\\mathbf{e}_c$": (c1[0] - 13, c1[1] + 9, COL["blue2"]), "$\\mathbf{h}_t$": (c1[0] + 19, c1[1] - 15, COL["green"])}
    pts2 = {"$\\mathbf{e}_c$": (c2[0] - 12, c2[1] + 7, COL["blue2"]), "$\\mathbf{h}_{t+1}$": (c2[0] + 19, c2[1] + 1, COL["green"])}
    for _label, (px, py, col) in pts1.items():
        ax.add_patch(patches.Circle((px, py), 5.1, facecolor=COL["green_soft"] if col == COL["green"] else COL["blue_soft"], edgecolor=col, lw=0.8, zorder=z + 4))
    for _label, (px, py, col) in pts2.items():
        ax.add_patch(patches.Circle((px, py), 5.1, facecolor=COL["green_soft"] if col == COL["green"] else COL["blue_soft"], edgecolor=col, lw=0.8, zorder=z + 4))
    ax.plot([pts1["$\\mathbf{e}_c$"][0], pts1["$\\mathbf{h}_t$"][0]], [pts1["$\\mathbf{e}_c$"][1], pts1["$\\mathbf{h}_t$"][1]], color=COL["red"], lw=1.0, zorder=z + 3)
    ax.plot([pts2["$\\mathbf{e}_c$"][0], pts2["$\\mathbf{h}_{t+1}$"][0]], [pts2["$\\mathbf{e}_c$"][1], pts2["$\\mathbf{h}_{t+1}$"][1]], color=COL["red"], lw=1.0, zorder=z + 3)
    ax.text(c1[0] - 24, c1[1] + 31, "$\\mathbf{e}_c$", fontsize=font_size(7.4), fontweight="bold", color=COL["blue2"], ha="center", va="center", zorder=z + 5)
    ax.text(c1[0] + 37, c1[1] - 19, "$\\mathbf{h}_t$", fontsize=font_size(7.4), fontweight="bold", color=COL["green"], ha="center", va="center", zorder=z + 5)
    ax.text(c2[0] - 24, c2[1] + 31, "$\\mathbf{e}_c$", fontsize=font_size(7.2), fontweight="bold", color=COL["blue2"], ha="center", va="center", zorder=z + 5)
    ax.text(
        c2[0] + 47,
        c2[1] + 30,
        "$\\mathbf{h}_{t+1}$",
        fontsize=font_size(6.3),
        fontweight="bold",
        color=COL["green"],
        ha="left",
        va="center",
        bbox={"facecolor": COL["paper"], "edgecolor": "none", "pad": 0.15},
        zorder=z + 6,
    )
    ax.text(c1[0], y + h - 12, "align gap", fontsize=font_size(6.8), fontweight="bold", color=COL["red_dark"], ha="center", va="center", zorder=z + 5)
    ax.text(c2[0], y + h - 12, "reduced gap", fontsize=font_size(6.8), fontweight="bold", color=COL["red_dark"], ha="center", va="center", zorder=z + 5)
    flow_arrow(ax, [(x + 98, y + 67), (x + 132, y + 67)], kind="micro", color=COL["aux"], z=z + 3)


def course_info_reward_panel(ax, x, y, w, h, z=14):
    box(ax, x, y, w, h, COL["paper"], ec=COL["green"], lw=1.25, r=1, z=z)
    ax.text(x + w / 2, y + 15, "Course-info Reward", fontsize=font_size(9.2), fontweight="bold", color=COL["ink"], ha="center", va="center", zorder=z + 5)

    course_cx = x + 47
    course_cy = y + 70
    book_open_icon(ax, course_cx, course_cy - 8, size=30, color=COL["line"], lw=1.1, z=z + 4)
    snowflake_icon(ax, course_cx + 18, course_cy - 23, r=3.8, color=COL["ice_dark"], lw=0.75, z=z + 6)
    ax.text(course_cx, course_cy + 25, "$c$", fontsize=font_size(8.2), fontweight="bold", color=COL["blue"], ha="center", va="center", zorder=z + 5)

    chips = [
        ("concept", "concept", COL["green"], COL["green_soft"]),
        ("prereq", "prereq", COL["orange"], COL["orange_soft"]),
        ("difficulty", "difficulty", COL["gold"], COL["gold_soft"]),
        ("repeat", "repeat", COL["muted"], "#eef2f6"),
    ]
    chip_x = x + 88
    chip_y = y + 35
    chip_w = 94
    chip_step = 108
    for idx, (label, kind, color, fill) in enumerate(chips):
        cx = chip_x + (idx % 2) * chip_step
        cy = chip_y + (idx // 2) * 43
        box(ax, cx, cy, chip_w, 30, "#fbfaf6", ec=color, lw=1.05, r=0.8, z=z + 1)
        reward_term_icon(ax, cx + 17, cy + 15, kind, color, fill, size=15, z=z + 4)
        ax.text(cx + 36, cy + 15, label, fontsize=font_size(7.0), fontweight="bold", color=COL["ink"], ha="left", va="center", zorder=z + 5)


def learner_action_set(ax, x, y, w=226, h=24, z=14):
    box(ax, x, y, w, h, "#fffaf1", ec=COL["line"], lw=1.1, r=1.0, z=z)
    ax.text(x + 64, y + h / 2, "Candidate users:", fontsize=font_size(7.5), fontweight="bold", color=COL["ink"], ha="center", va="center", zorder=z + 5)
    entries = [("$u_1$", False), ("$u_j$", True), ("$u_N$", False)]
    for idx, (label, selected) in enumerate(entries):
        cx = x + 142 + idx * 43
        cy = y + h / 2
        ax.add_patch(
            patches.Circle(
                (cx, cy),
                9.4 if selected else 8.4,
                facecolor=COL["orange_soft"],
                edgecolor=COL["orange"],
                linewidth=stroke_width(1.25 if selected else 0.9, floor=1.1),
                zorder=z + 3,
            )
        )
        ax.text(cx, cy + 0.2, label, fontsize=font_size(6.4), fontweight="bold", color=COL["orange"], ha="center", va="center", zorder=z + 4)
    ax.text(x + w - 45, y + h / 2 - 0.2, "$\\cdots$", fontsize=font_size(8.5), fontweight="bold", color=COL["muted"], ha="center", va="center", zorder=z + 4)
    x0 = x + w - 15
    y0 = y + h / 2
    ax.plot([x0 - 7, x0 + 7], [y0 - 7, y0 + 7], color=COL["red"], lw=1.75, zorder=z + 4)
    ax.plot([x0 + 7, x0 - 7], [y0 - 7, y0 + 7], color=COL["red"], lw=1.75, zorder=z + 4)


def selected_action_badge(ax, cx, cy, label="$a_t$", z=18):
    ax.add_patch(
        patches.Circle(
            (cx, cy),
            11.8,
            facecolor=COL["orange_soft"],
            edgecolor=COL["orange"],
            linewidth=stroke_width(1.25, floor=1.35),
            zorder=z,
        )
    )
    ax.text(cx, cy + 0.1, label, fontsize=font_size(8.1), fontweight="bold", color=COL["orange"], ha="center", va="center", zorder=z + 1)


def draw_reward_function_reference_style(ax, x, y, w, h, z=13):
    box(ax, x, y, w, h, "#f8f8f2", ec=COL["aux"], lw=1.5, r=0.5, ls=(0, (6, 4)), z=z)
    txt(ax, x + w / 2, y + 21, "Reward Function", size=13.0, weight="bold")
    align_w = 272
    course_w = w - align_w - 78
    alignment_reward_panel(ax, x + 20, y + 36, align_w, h - 58, z=z + 2)
    ax.text(x + align_w + 39, y + h / 2 + 6, "+", fontsize=font_size(15), fontweight="bold", color=COL["ink"], ha="center", va="center", zorder=z + 5)
    course_info_reward_panel(ax, x + align_w + 58, y + 36, course_w, h - 58, z=z + 2)


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
    ax.text(cx - 4 * scale, cy + 10 * scale, r"$\theta$", fontsize=font_size(8.5), fontweight="bold", color=COL["muted"], zorder=z + 1)


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
            linewidth=stroke_width(1.05, floor=1.25),
            zorder=z,
        )
    )
    divider_x = x + 20
    ax.plot([divider_x, divider_x], [y, y + h], color=ec, lw=0.95 * VECTOR_LINE_SCALE, zorder=z + 1)
    ax.text(
        x + 10.0,
        y + h / 2 + 0.4,
        str(rank),
        fontsize=font_size(10.8),
        fontweight="bold",
        color=COL["blue"] if cold else COL["muted"],
        ha="center",
        va="center",
        zorder=z + 3,
    )
    book_open_icon(ax, x + 34, y + h / 2 + 0.4, size=14.5, color=COL["line"], lw=1.0, z=z + 3)
    if cold:
        snowflake_icon(ax, x + 39, y + 8, r=2.3, color=COL["ice_dark"], lw=0.6, z=z + 4)
    ax.text(
        x + 50,
        y + h / 2 + 0.2,
        course,
        fontsize=font_size(10.9),
        fontweight="bold",
        color=COL["blue"] if cold else COL["ink"],
        ha="left",
        va="center",
        zorder=z + 3,
    )


def draw_panel_headers(ax):
    txt(ax, 40, 120, "a", size=14.5, weight="bold", ha="left")
    txt(ax, 260, 121, "Cold-course Evidence Encoder", size=11.2, weight="bold")
    txt(ax, 486, 120, "b", size=15.0, weight="bold", ha="left")
    txt(ax, 878, 121, "Course-knowledge Guided Simulation", size=13.2, weight="bold")
    txt(ax, 1344, 120, "c", size=15.0, weight="bold", ha="left")
    txt(ax, 1566, 121, "Strict Course-Cold Ranking", size=12.5, weight="bold")


def draw_left(ax):
    panel_box(ax, 28, 96, 378, 570, COL["left"], accent=COL["accent_a"])

    txt(ax, 217, 152, "Course Encoder", size=12.8, weight="bold", color=COL["blue"])
    course_icon(ax, 176, 158, 84, 80, framed=False)

    # Course evidence branches into content and behavior embeddings.
    course_cx = 218
    branch_y = 252
    tower_y = 286
    tower_h = 150
    tower_w = 162
    tower_gap = 20
    content_x = 44
    behavior_x = content_x + tower_w + tower_gap
    left_tower_cx = content_x + tower_w / 2
    right_tower_cx = behavior_x + tower_w / 2
    branch_stem_top_y = branch_y - 10
    encoder_lw = ARROW_STYLES["encoder"]["lw"]
    ax.plot(
        [course_cx, course_cx],
        [branch_stem_top_y, branch_y],
        color=COL["flow"],
        lw=encoder_lw,
        solid_capstyle="butt",
        solid_joinstyle="round",
        zorder=13,
    )
    ax.plot(
        [left_tower_cx, right_tower_cx],
        [branch_y, branch_y],
        color=COL["flow"],
        lw=encoder_lw,
        solid_capstyle="butt",
        solid_joinstyle="round",
        zorder=11,
    )
    flow_arrow(ax, [(left_tower_cx, branch_y), (left_tower_cx, tower_y)], kind="encoder", z=12)
    flow_arrow(ax, [(right_tower_cx, branch_y), (right_tower_cx, tower_y)], kind="encoder", z=12)

    box(ax, content_x, tower_y, tower_w, tower_h, "#f8f8f2", ec=COL["teal"], lw=1.8, r=0.5)
    txt(ax, left_tower_cx, tower_y + 20, "Content Emb.", size=10.8, weight="bold", color=COL["teal"])
    vector(ax, left_tower_cx - 46, tower_y + 38, w=92, h=16, colors=CONTENT_EMB_COLORS)
    txt(ax, left_tower_cx - 58, tower_y + 46, "$\\mathbf{x}_c$", size=8.4, weight="bold", color=COL["teal"], ha="right")
    mlp_vertical_wedge(ax, left_tower_cx - 32, tower_y + 66, w=64, h=28, ec=COL["teal"], label="MLP", label_size=8.8)
    vector(ax, left_tower_cx - 46, tower_y + 110, w=92, h=17, colors=CONTENT_EMB_COLORS)
    txt(ax, left_tower_cx + 58, tower_y + 118, "$\\mathbf{c}_c$", size=8.4, weight="bold", color=COL["teal"], ha="left")
    flow_arrow(ax, [(left_tower_cx, tower_y + 53), (left_tower_cx, tower_y + 66)], kind="micro")
    flow_arrow(ax, [(left_tower_cx, tower_y + 94), (left_tower_cx, tower_y + 110)], kind="micro")

    box(ax, behavior_x, tower_y, tower_w, tower_h, "#f8f8f2", ec=COL["panel"], lw=1.8, r=0.5)
    txt(ax, right_tower_cx, tower_y + 20, "Behavior Emb.", size=10.8, weight="bold", color=COL["muted"])
    vector(ax, right_tower_cx - 46, tower_y + 38, w=92, h=16, colors=BEHAVIOR_EMB_COLORS)
    txt(ax, right_tower_cx - 58, tower_y + 46, "$\\mathbf{v}_c$", size=8.4, weight="bold", color=COL["muted"], ha="right")
    box(ax, right_tower_cx - 35, tower_y + 66, 70, 26, COL["red_soft"], ec=COL["red_dark"], lw=1.25, r=1.0, z=9)
    txt(ax, right_tower_cx, tower_y + 79, "ID mask", size=7.6, weight="bold", color=COL["red_dark"], z=12)
    vector(ax, right_tower_cx - 46, tower_y + 110, w=92, h=17, colors=BEHAVIOR_EMB_COLORS)
    txt(ax, right_tower_cx + 58, tower_y + 118, "$\\bar{\\mathbf{v}}_c$", size=8.4, weight="bold", color=COL["muted"], ha="left")
    flow_arrow(ax, [(right_tower_cx, tower_y + 53), (right_tower_cx, tower_y + 68)], kind="micro")
    flow_arrow(ax, [(right_tower_cx, tower_y + 92), (right_tower_cx, tower_y + 110)], kind="micro")

    fusion_y = 440
    fusion_h = 72
    fusion_x = 64
    fusion_w = 320
    fusion_cx = (left_tower_cx + right_tower_cx) / 2
    gate_top_y = fusion_y + 24
    gate_bottom_y = gate_top_y + 22
    flow_arrow(ax, [(left_tower_cx, tower_y + tower_h), (left_tower_cx, gate_top_y)], kind="encoder", z=12)
    flow_arrow(ax, [(right_tower_cx, tower_y + tower_h), (right_tower_cx, gate_top_y)], kind="encoder", z=12)

    gate_fusion_circuit(
        ax,
        fusion_x,
        fusion_y,
        w=fusion_w,
        h=fusion_h,
        left_x=left_tower_cx,
        right_x=right_tower_cx,
        out_x=fusion_cx,
        z=18,
    )

    q_vec_y = 552
    flow_arrow(ax, [(fusion_cx, gate_bottom_y + 2), (fusion_cx, q_vec_y)], kind="encoder", z=12)
    vector(ax, fusion_cx - 38, q_vec_y, w=76, h=18)
    txt(ax, fusion_cx + 50, q_vec_y + 9, "$\\mathbf{e}_c$", size=12.0, weight="bold", color=COL["violet"], ha="left")

    txt(ax, 213, 594, "Learner Encoder", size=12.8, weight="bold", color=COL["orange"])
    learner_cy = 624
    user_icon(ax, 96, learner_cy - 22)
    vector(ax, 128, learner_cy - 8.5, w=66, h=17, colors=USER_EMB_COLORS)
    txt(ax, 161, 654, "$\\mathbf{e}_u$", size=11.0, weight="bold", color=COL["muted"])
    mlp_wedge(ax, 214, learner_cy - 19, w=58, h=38, ec=COL["orange"], label="MLP", label_size=11.5)
    vector(ax, 306, learner_cy - 8.5, w=66, h=17, colors=USER_EMB_COLORS)
    txt(ax, 339, 654, "$\\mathbf{z}_u$", size=11.5, weight="bold", color=COL["orange"])
    flow_arrow(ax, [(112, learner_cy), (128, learner_cy)], kind="micro")
    flow_arrow(ax, [(194, learner_cy), (214, learner_cy)], kind="micro")
    flow_arrow(ax, [(272, learner_cy), (306, learner_cy)], kind="micro")


def draw_middle(ax):
    panel_box(ax, 470, 96, 800, 570, COL["middle"], accent=COL["accent_b"])

    state_capsule(
        ax,
        545,
        186,
        w=50,
        h=168,
        label="$\\mathbf{s}_t$",
        colors=STATE_EMB_COLORS,
        band_labels=["$\\mathbf{h}_t$", "$\\tau_t$"],
        h_label_side="right",
        z=12,
    )
    box(ax, 635, 190, 260, 116, "#f8f8f2", ec=COL["aux"], lw=1.5, r=0.5, ls=(0, (6, 4)))
    txt(ax, 765, 215, "Exploration Set\nConstruction", size=11.6, weight="bold")
    box(ax, 646, 250, 66, 38, COL["card2"], ec=COL["green"], lw=1.3, r=0.5)
    txt(ax, 679, 269, "$\\mathcal{N}_M$", size=10.8, weight="bold", color=COL["green"])
    box(ax, 736, 250, 72, 38, COL["card2"], ec=COL["gold"], lw=1.3, r=0.5)
    txt(ax, 772, 269, "sample $N$", size=9.4, weight="bold", color=COL["gold"])
    box(ax, 824, 250, 66, 38, COL["red_soft"], ec=COL["red_dark"], lw=1.3, r=0.5)
    txt(ax, 857, 269, "$s_{sample}$", size=8.3, weight="bold", color=COL["red"])
    learner_action_set(ax, 610, 328, w=298, h=31, z=14)

    box(ax, 628, 372, 274, 66, "#f8f8f2", ec=COL["line"], lw=1.55, r=0.5)
    brain_circuit_icon(ax, 665, 405, size=28, color=COL["violet"], accent=COL["blue2"], z=18)
    txt(ax, 774, 396, "Actor-Critic Agent", size=12.4, weight="bold")
    txt(ax, 774, 421, "select learner $a_t\\sim\\pi_\\theta(\\cdot\\mid\\mathbf{s}_t)$", size=10.2, color=COL["muted"])

    draw_state_transition_function(ax, 918, 186, 242, 168, z=13)

    state_capsule(
        ax,
        1189,
        186,
        w=50,
        h=168,
        label="$\\mathbf{s}_{t+1}$",
        colors=NEXT_STATE_EMB_COLORS,
        accent=COL["red"],
        band_labels=["$\\mathbf{h}_{t+1}$", "$\\tau_{t+1}$"],
        z=12,
    )

    reward_y = 468
    draw_reward_function_reference_style(ax, 545, reward_y, 660, 178, z=13)

    flow_arrow(ax, [(595, 272), (635, 272)], kind="data")
    flow_arrow(ax, [(765, 306), (765, 328)], kind="data", z=22)
    flow_arrow(ax, [(765, 359), (765, 372)], kind="data", z=22)
    flow_arrow(ax, [(902, 405), (955, 405)], kind="action", z=22)
    txt(ax, 928, 390, "$a_t$", size=10.2, weight="bold", color=COL["orange"], z=24)
    selected_action_badge(ax, 970, 405, "$u_j$", z=23)
    flow_arrow(ax, [(985, 405), (1040, 405), (1040, 354)], kind="action", z=22)
    flow_arrow(ax, [(1150, 272), (1189, 272)], kind="data")
    flow_arrow(ax, [(800, reward_y - 1), (800, 434)], kind="reward", z=23)
    txt(ax, 824, 451, "$r_t$", size=10.4, weight="bold", color=COL["orange"], z=24)

    flow_arrow(ax, [(1214, 190), (1214, 166), (570, 166), (570, 190)], kind="aux", z=15)
    time_chip(ax, 1098, 390, 66, 30, "$-1$", accent=COL["blue"], z=26)
    flow_arrow(ax, [(1160, 405), (1175, 405), (1175, 326), (1192, 326)], kind="aux", z=25)
    flow_arrow(ax, [(1236, 272), (1260, 272), (1260, 560), (1205, 560)], kind="aux", z=20)


def draw_right(ax):
    dx = 14
    panel_box(ax, 1320 + dx, 96, 430, 570, COL["right"], accent=COL["accent_c"])

    source_x = 1396 + dx
    source_w = 72
    source_right = source_x + source_w
    score_x = 1502 + dx
    score_w = 130
    score_right = score_x + score_w
    score_cx = score_x + score_w / 2
    dot_r = 16
    dot_cy = 421
    topk_x = 1602 + dx
    topk_w = 116
    topk_cx = topk_x + topk_w / 2
    input_elbow_x = score_cx - dot_r - 46
    bank_elbow_x = input_elbow_x
    dot_gap = 9

    box(ax, 1340 + dx, 150, 390, 60, COL["cream"], ec=COL["panel"], lw=1.45, r=0.5)
    snowflake_icon(ax, 1378 + dx, 180, r=6.0, color=COL["ice_dark"], lw=1.0)
    txt(ax, 1554 + dx, 176, "Strict Course-Cold Inference", size=12.6, weight="bold")
    txt(ax, 1554 + dx, 198, "full catalog $\\mathcal{C}_u$", size=10.6, color=COL["muted"])

    box(ax, 1340 + dx, 238, 390, 378, "#f8f8f2", ec=COL["line"], lw=1.7, r=0.5, ls=(0, (7, 5)))
    txt(ax, 1535 + dx, 268, "Ranking Logit", size=13.2, weight="bold")

    user_icon(ax, source_x - 20, 315)
    txt(ax, source_x + source_w / 2, 302, "User Embedding", size=10.8, weight="bold", color=COL["orange"], ha="center")
    vector(ax, source_x, 324, w=source_w, h=17, colors=USER_EMB_COLORS)
    txt(ax, source_x + source_w / 2, 360, "$\\mathbf{z}_u$", size=11.8, weight="bold", color=COL["orange"])

    book_open_icon(ax, source_x - 20, 510, size=34, color=COL["line"], lw=1.35, z=18)
    txt(ax, source_x + source_w / 2, 482, "$\\mathbf{z}_{c,T}$", size=11.8, weight="bold", color=COL["red"])
    vector(ax, source_x, 501, w=source_w, h=17, colors=NEXT_STATE_EMB_COLORS)
    txt(ax, source_x + source_w / 2, 540, "Course Embedding", size=10.8, weight="bold", color=COL["red"], ha="center")

    ax.add_patch(
        patches.Circle(
            (score_cx, dot_cy),
            dot_r,
            facecolor="#f8f8f2",
            edgecolor=COL["line"],
            linewidth=stroke_width(1.5, floor=1.8),
            zorder=11,
        )
    )
    txt(ax, score_cx, dot_cy - 1, "$\\cdot$", size=20.0, weight="bold", color=COL["ink"], z=12)

    flow_arrow(
        ax,
        [(source_right, 333), (input_elbow_x, 333), (input_elbow_x, dot_cy - dot_gap), (score_cx - dot_r, dot_cy - dot_gap)],
        kind="rank_user",
    )
    flow_arrow(
        ax,
        [(source_right, 510), (bank_elbow_x, 510), (bank_elbow_x, dot_cy + dot_gap), (score_cx - dot_r, dot_cy + dot_gap)],
        kind="rank_course",
    )
    flow_arrow(ax, [(score_cx + dot_r, dot_cy), (topk_x, dot_cy)], kind="rank_out")

    box(ax, topk_x, 312, topk_w, 216, "#faf9f4", ec=COL["panel"], lw=1.55, r=0.5, z=3)
    txt(ax, topk_cx, 340, "Top-K $\\pi_u$", size=10.4, weight="bold", color=COL["ink"])
    recommendation_row(ax, topk_x + 4, 370, topk_w - 8, 34, 1, "C-17", "", cold=True)
    recommendation_row(ax, topk_x + 4, 408, topk_w - 8, 34, 2, "C-04", "", cold=True)
    recommendation_row(ax, topk_x + 4, 446, topk_w - 8, 34, 3, "C-31", "", cold=False)
    recommendation_row(ax, topk_x + 4, 484, topk_w - 8, 34, "K", "C-K", "", cold=False)


def draw_cross_panel_routes(ax):
    red_boundary_arrow(ax, 436, 344, scale=1.0, z=31)

    vertical_embedding(ax, 498, 198, w=26, h=110, colors=list(reversed(COURSE_EMB_COLORS)), z=29)
    ax.text(511, 190, "$\\mathrm{init}$", fontsize=font_size(10.2), fontweight="bold", color=COL["ink"], ha="center", va="center", zorder=31)
    ax.text(511, 319, "$\\mathbf{e}_c$", fontsize=font_size(10.8), fontweight="bold", color=COL["ink"], ha="center", va="center", zorder=31)
    flow_arrow(ax, [(524, 253), (545, 253)], kind="encoder", z=30)
    time_chip(ax, 492, 390, 58, 30, "$\\tau_0$", accent=COL["blue"], z=31)
    flow_arrow(ax, [(521, 390), (521, 326), (545, 326)], kind="aux", z=30)

    red_boundary_arrow(ax, 1300, 478, scale=1.0, z=31)


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
        fontsize=font_size(size),
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
    colors = colors or COURSE_EMB_COLORS
    step = w / len(colors)
    for i, c in enumerate(colors):
        ax.add_patch(
            patches.Rectangle((x + i * step, y), step, h, facecolor=c, edgecolor="none", linewidth=0, zorder=z)
        )
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="none", edgecolor=COL["line"], linewidth=1.35, zorder=z + 1))


def overview_panel(ax, x, y, w, h, fc, label, title_lines):
    band_h = 116
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=fc, edgecolor=COL["panel"], linewidth=3.2, zorder=1))
    ax.add_patch(patches.Rectangle((x, y), w, band_h, facecolor=fc, edgecolor=COL["panel"], linewidth=0, zorder=2))
    ax.plot([x, x + w], [y + band_h, y + band_h], color=COL["panel"], lw=2.4, zorder=3)
    overview_text(ax, x + 26, y + band_h / 2, f"({label})", size=24, weight="bold", ha="left", z=5)
    n = len(title_lines)
    gap = 34
    start = y + band_h / 2 - (n - 1) * gap / 2
    for j, line in enumerate(title_lines):
        overview_text(ax, x + 108, start + j * gap, line, size=14.5, weight="bold", ha="left", z=5)


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
    overview_text(ax, cx, y + 184, "cold course $c$", size=22, weight="bold")
    overview_text(ax, cx, y + 212, "no ID interaction signal", size=16, color=COL["muted"])

    side = (x + 50, y + 270, 190, 112)
    ident = (x + 280, y + 270, 190, 112)
    overview_box(ax, *side, "", fc="#fbfbf7", ec=COL["green"], lw=2.35, z=8)
    overview_box(ax, *ident, "", fc="#f4f4f0", ec=COL["muted"], lw=2.35, z=8)
    overview_text(ax, side[0] + side[2] / 2, side[1] + 34, "Content\nEmbedding", size=16.2, weight="bold", color=COL["green"])
    overview_text(ax, ident[0] + ident[2] / 2, ident[1] + 34, "Behavior\nEmbedding", size=15.4, weight="bold", color=COL["muted"])
    overview_vector(ax, side[0] + 43, side[1] + 78, w=104, h=18, colors=CONTENT_EMB_COLORS)
    overview_vector(
        ax,
        ident[0] + 43,
        ident[1] + 78,
        w=104,
        h=18,
        colors=BEHAVIOR_EMB_COLORS,
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
        "$\\sigma\\ \\mathrm{gate}\\rightarrow\\mathbf{g}_c$",
        size=16.5,
        color=COL["muted"],
    )

    overview_vector(ax, cx - 58, y + 566, w=116, h=22)
    overview_text(ax, cx, y + 618, "$\\mathbf{e}_c$ / $\\mathbf{z}_{c,T}$", size=22, weight="bold", color=COL["green"])

    trunk_y = y + 246
    overview_arrow(ax, [(cx, y + 224), (cx, trunk_y), (side[0] + side[2] / 2, trunk_y), (side[0] + side[2] / 2, side[1])], color=COL["flow"])
    overview_arrow(ax, [(cx, trunk_y), (ident[0] + ident[2] / 2, trunk_y), (ident[0] + ident[2] / 2, ident[1])], color=COL["flow"])
    merge_y = y + 404
    overview_arrow(ax, [(side[0] + side[2] / 2, side[1] + side[3]), (side[0] + side[2] / 2, merge_y), (fusion[0] + 92, merge_y), (fusion[0] + 92, fusion[1])], color=COL["flow"])
    overview_arrow(ax, [(ident[0] + ident[2] / 2, ident[1] + ident[3]), (ident[0] + ident[2] / 2, merge_y), (fusion[0] + 212, merge_y), (fusion[0] + 212, fusion[1])], color=COL["flow"])
    overview_arrow(ax, [(cx, fusion[1] + fusion[3]), (cx, y + 566)], color=COL["flow"])

    ax.plot([x + 34, x + w - 34], [y + 628, y + 628], color="#c9d6c0", lw=1.8, zorder=3)
    overview_text(ax, x + 52, y + 654, "User-history encoder", size=17.5, weight="bold", color=COL["orange"], ha="left")
    hist_y = y + 700
    user_round_icon(ax, x + 66, hist_y, size=30, color=COL["line"], lw=1.7, z=10)
    overview_box(ax, x + 118, hist_y - 24, 108, 48, "history", fc="#fbfbf7", ec=COL["orange"], color=COL["orange"], size=14.5)
    mlp_wedge(ax, x + 256, hist_y - 26, w=66, h=52, ec=COL["orange"], label="MLP", label_size=15.0)
    overview_vector(ax, x + 358, hist_y - 12, w=82, h=22, colors=USER_EMB_COLORS)
    overview_text(ax, x + 399, hist_y + 34, "$\\mathbf{z}_u$", size=19, weight="bold", color=COL["orange"])
    overview_arrow(ax, [(x + 226, hist_y), (x + 256, hist_y)], color=COL["flow"], lw=2.25, ms=16)
    overview_arrow(ax, [(x + 322, hist_y), (x + 358, hist_y)], color=COL["flow"], lw=2.25, ms=16)


def draw_simulation_overview(ax):
    x, y, w, h = 605, 50, 930, 720
    overview_panel(ax, x, y, w, h, "#dce9f2", "b", ["Knowledge-guided", "Learner-Course", "Simulation"])
    sim = (x + 46, y + 136, w - 92, 574)
    overview_box(ax, *sim, "", fc="#d6e6f1", ec=COL["blue"], lw=2.8, z=2)
    overview_text(ax, x + w / 2, y + 174, "Training-time MDP rollout", size=24, weight="bold")

    state = (x + 62, y + 236, 196, 84)
    policy = (x + 292, y + 220, 208, 116)
    action = (x + 540, y + 240, 176, 76)
    trans = (x + 754, y + 224, 150, 108)
    overview_box(ax, *state, "$\\mathbf{s}_t=[\\mathbf{h}_t;\\tau_t]$", fc="#fbfbf7", ec=COL["blue"], color=COL["blue"], size=15.5)
    overview_box(ax, *policy, "Actor-critic\npolicy\n$a_t\\sim\\pi_\\theta(\\cdot\\mid\\mathbf{s}_t)$", fc="#fbfbf7", ec=COL["line"], size=14.5)
    overview_box(ax, *action, "selected\nlearner $a_t$", fc="#fff8ee", ec=COL["orange"], color=COL["orange"], size=14.5)
    overview_box(ax, *trans, "$\\mathbf{h}_{t+1}=$\n$f(\\mathbf{h}_t,a_t,r_t)$", fc="#fbfbf7", ec=COL["blue"], color=COL["blue"], size=13.5)
    flow_y = state[1] + state[3] / 2
    overview_arrow(ax, [(state[0] + state[2], flow_y), (policy[0], flow_y)], color=COL["blue"])
    overview_arrow(ax, [(policy[0] + policy[2], flow_y), (action[0], flow_y)], color=COL["blue"])
    overview_arrow(ax, [(action[0] + action[2], flow_y), (trans[0], flow_y)], color=COL["blue"])

    reward = (x + 96, y + 430, w - 192, 252)
    overview_box(ax, *reward, "", fc="#fbfbf7", ec=COL["orange"], ls=(0, (9, 5)), lw=2.6, z=4)
    overview_text(ax, x + w / 2, reward[1] + 32, "Knowledge-guided reward signals", size=21, weight="bold", color=COL["orange"])
    labels = [
        "target\nmatch",
        "progress\ngain",
        "concept\ncoverage",
        "prereq\npath",
        "difficulty\nfit",
        "repeat\nguard",
    ]
    chip_w, chip_h = 212, 62
    sx, sy = reward[0] + 36, reward[1] + 72
    for r in range(2):
        for c in range(3):
            title = labels[r * 3 + c]
            bx, by = sx + c * 240, sy + r * 76
            overview_box(ax, bx, by, chip_w, chip_h, "", fc="#fffaf0", ec=COL["orange"], lw=1.85, z=8)
            overview_text(ax, bx + chip_w / 2, by + chip_h / 2, title, size=15.2, weight="bold", color=COL["ink"])
    overview_text(ax, x + w / 2, reward[1] + reward[3] - 16, "weighted feedback signals", size=16.8, weight="bold", color=COL["red_dark"])
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
    overview_panel(ax, x, y, w, h, "#e7e3ef", "c", ["Strict Course-Cold", "Ranking", "Inference"])
    overview_box(
        ax,
        x + 44,
        y + 150,
        w - 88,
        80,
        "Strict Course-Cold Inference\nfull catalog $\\mathcal{C}_u$",
        fc=COL["cream"],
        ec=COL["panel"],
        size=17,
        lw=2.5,
    )
    area = (x + 38, y + 256, w - 76, 446)
    overview_box(ax, *area, "", fc="#fbfbf7", ec=COL["line"], ls=(0, (9, 6)), lw=2.6, z=2)

    user_y = y + 360
    cold_y = y + 560
    input_x = x + 96
    overview_text(ax, input_x + 82, user_y - 54, "user\nembedding", size=15.5, weight="bold", color=COL["orange"])
    user_round_icon(ax, input_x - 26, user_y, size=40, color=COL["line"], lw=1.7, z=10)
    overview_vector(ax, input_x + 22, user_y - 14, w=128, h=26, colors=USER_EMB_COLORS)
    overview_text(ax, input_x + 86, user_y + 48, "$\\mathbf{z}_u$", size=20, weight="bold", color=COL["orange"])

    snowflake_icon(ax, input_x - 40, cold_y, r=7.5, color=COL["ice_dark"], lw=1.1, z=10)
    book_open_icon(ax, input_x - 6, cold_y, size=32, color=COL["line"], lw=1.4, z=10)
    overview_text(ax, input_x + 84, cold_y - 62, "cold-course\nbank", size=15.5, weight="bold", color=COL["blue"])
    overview_vector(ax, input_x + 22, cold_y - 14, w=128, h=26)
    overview_text(ax, input_x + 86, cold_y + 54, "$\\mathbf{z}_{c,T}$", size=20, weight="bold", color=COL["blue"])

    score_cx, score_cy, score_r = x + 340, y + 460, 26
    ax.add_patch(
        patches.Circle(
            (score_cx, score_cy),
            score_r,
            facecolor="#f8f8f2",
            edgecolor=COL["line"],
            linewidth=2.3,
            zorder=8,
        )
    )
    overview_text(ax, score_cx, score_cy - 1, "$\\cdot$", size=34, weight="bold", z=10)
    overview_text(ax, score_cx, score_cy + score_r + 20, "cosine\nlogit", size=13.5, weight="bold", color=COL["muted"], z=10)

    topk = (x + 398, y + 330, 118, 260)
    overview_box(ax, *topk, "", fc="#faf9f4", ec=COL["panel"], lw=2.3, z=8)
    overview_text(ax, topk[0] + topk[2] / 2, topk[1] + 28, "Top-K", size=17.5, weight="bold")
    rows = [(1, "C-17"), (2, "C-04"), (3, "C-31"), ("K", "C-K")]
    for j, (rank, course) in enumerate(rows):
        yy = topk[1] + 56 + j * 46
        ax.add_patch(
            patches.Rectangle(
                (topk[0] + 8, yy),
                topk[2] - 16,
                38,
                facecolor="#f4f7f7",
                edgecolor=COL["blue"] if j < 2 else COL["panel"],
                linewidth=1.4,
                zorder=9,
            )
        )
        course_marker(ax, topk[0] + 13, yy + 13, w=12, h=12, z=10)
        overview_text(ax, topk[0] + 40, yy + 19, str(rank), size=14.5, weight="bold", color=COL["blue"] if j < 2 else COL["muted"])
        overview_text(ax, topk[0] + 80, yy + 19, course, size=13.5, weight="bold", color=COL["blue"] if j < 2 else COL["ink"])

    elbow_x = score_cx - score_r - 46
    overview_arrow(ax, [(input_x + 150, user_y), (elbow_x, user_y), (elbow_x, score_cy - 11), (score_cx - score_r, score_cy - 11)], color=COL["orange"])
    overview_arrow(ax, [(input_x + 150, cold_y), (elbow_x, cold_y), (elbow_x, score_cy + 11), (score_cx - score_r, score_cy + 11)], color=COL["blue"])
    overview_arrow(ax, [(score_cx + score_r, score_cy), (topk[0], score_cy)], color=COL["flow"])


def draw_overview_routes(ax):
    red_boundary_arrow(ax, 582, 430, scale=0.88, z=31)
    overview_text(ax, 582, 390, "$\\mathbf{e}_c$", size=17, weight="bold", color=COL["green"], z=32)
    red_boundary_arrow(ax, 1560, 430, scale=0.88, z=31)


def draw_figure():
    fig = plt.figure(figsize=(15.6, 5.9), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 2180)
    ax.set_ylim(800, 20)
    ax.axis("off")
    ax.add_patch(patches.Rectangle((0, 20), 2180, 780, facecolor=COL["paper"], edgecolor="none", zorder=0))

    draw_encoder_overview(ax)
    draw_simulation_overview(ax)
    draw_ranking_overview(ax)
    draw_overview_routes(ax)
    return fig


def draw_figure_detailed():
    fig = plt.figure(figsize=(13.9, 5.8), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1794)
    ax.set_ylim(760, 0)
    ax.axis("off")
    ax.add_patch(patches.Rectangle((0, 0), 1794, 760, facecolor=COL["paper"], edgecolor="none", zorder=0))

    draw_panel_headers(ax)
    draw_left(ax)
    draw_middle(ax)
    draw_right(ax)
    draw_cross_panel_routes(ax)
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
