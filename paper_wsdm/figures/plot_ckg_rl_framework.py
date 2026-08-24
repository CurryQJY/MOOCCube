import matplotlib

matplotlib.use("Agg")

import math
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch


# Nature-figure contract
# Core conclusion: CKG-RL respects the strict item-cold evidence boundary by
# building cold-course representations from course-side evidence and then uses
# course-knowledge-guided simulation to optimize full-catalog item-cold ranking.
# Archetype: schematic-led composite / asymmetric mixed-modality method figure.
# Backend: Python only. Primary export: editable SVG.

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


W, H = 1500, 780

COLORS = {
    "ink": "#172033",
    "muted": "#566276",
    "line": "#232b35",
    "panel_line": "#a9b0bb",
    "left_bg": "#e4efd9",
    "mid_bg": "#dcecf9",
    "right_bg": "#ebeaf5",
    "right_top_bg": "#f4ecd6",
    "blue": "#2468a2",
    "blue_soft": "#cfe3f3",
    "blue_mid": "#6aa0c8",
    "orange": "#b7703c",
    "orange_soft": "#f2d5bd",
    "green": "#5f8f4b",
    "green_soft": "#dcedd2",
    "gold": "#c59a2f",
    "gold_soft": "#f0dfae",
    "violet": "#7367a8",
    "violet_soft": "#e3e0f2",
    "rose": "#a65c70",
    "rose_soft": "#f0d6df",
    "red": "#c71f2d",
    "red_soft": "#fff1f1",
    "white": "#ffffff",
    "black": "#111111",
    "grey": "#eef2f6",
}


def add_text(
    ax,
    x,
    y,
    text,
    size=10,
    weight="normal",
    color=None,
    ha="center",
    va="center",
    width=None,
    linespacing=1.05,
    style="normal",
    zorder=10,
    rotation=0,
):
    if width:
        text = "\n".join(textwrap.wrap(text, width=width, break_long_words=False))
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color or COLORS["ink"],
        ha=ha,
        va=va,
        linespacing=linespacing,
        fontstyle=style,
        zorder=zorder,
        rotation=rotation,
    )


def rect(ax, x, y, w, h, fill, edge=None, lw=1.5, radius=0, ls="-", zorder=1):
    if radius:
        patch = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.02,rounding_size={radius}",
            linewidth=lw,
            edgecolor=edge or COLORS["line"],
            facecolor=fill,
            linestyle=ls,
            zorder=zorder,
        )
    else:
        patch = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=lw,
            edgecolor=edge or COLORS["line"],
            facecolor=fill,
            linestyle=ls,
            zorder=zorder,
        )
    ax.add_patch(patch)
    return patch


def arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    color=None,
    lw=2.0,
    ms=14,
    style="-|>",
    ls="-",
    rad=0,
    zorder=9,
    alpha=1.0,
):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color or COLORS["line"],
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
        alpha=alpha,
    )
    ax.add_patch(arr)
    return arr


def stage_arrow(ax, x1, y1, x2, y2, label=None):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="simple,head_width=34,head_length=24,tail_width=14",
        mutation_scale=1,
        linewidth=2.7,
        edgecolor=COLORS["red"],
        facecolor=COLORS["white"],
        zorder=14,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arr)
    if label:
        add_text(ax, (x1 + x2) / 2, y1 - 21, label, size=8.5, weight="bold", color=COLORS["red"])
    return arr


def dashed_backbone(ax):
    y = 44
    arrow(ax, 1380, y, 215, y, color=COLORS["red"], lw=2.5, ms=20, ls=(0, (4, 4)), style="-|>")
    for x in [1200, 890, 610]:
        arrow(ax, x + 55, y - 19, x, y - 19, color=COLORS["red"], lw=2.5, ms=20, style="-|>")
    arrow(ax, 215, y, 215, 110, color=COLORS["red"], lw=2.0, ms=16, ls=(0, (4, 4)), style="-|>")
    add_text(
        ax,
        790,
        24,
        "training-time reward and loss signals",
        size=10,
        weight="bold",
        color=COLORS["red"],
    )


def vector_bar(ax, x, y, w=72, h=22, fills=None, edge=COLORS["line"]):
    fills = fills or ["#f7f7f7", "#dbe7f3", "#91bad7", "#2f78b7"]
    n = len(fills)
    for i, c in enumerate(fills):
        rect(ax, x + i * w / n, y, w / n, h, c, edge=edge, lw=1.0, zorder=5)
    rect(ax, x, y, w, h, "none", edge=edge, lw=1.1, zorder=6)


def mini_network(ax, x, y, w=116, h=76):
    rect(ax, x, y, w, h, COLORS["white"], edge=COLORS["line"], lw=1.4, radius=8, ls=(0, (4, 3)), zorder=5)
    pts = [
        (x + 24, y + 45),
        (x + 51, y + 20),
        (x + 82, y + 43),
        (x + 55, y + 62),
        (x + 95, y + 62),
    ]
    lines = [(0, 1), (1, 2), (0, 3), (3, 4), (1, 3), (2, 4)]
    for a, b in lines:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color=COLORS["line"], lw=1.4, zorder=6)
    node_cols = [COLORS["orange"], COLORS["blue_mid"], COLORS["orange"], COLORS["blue_mid"], COLORS["blue_mid"]]
    for (px, py), c in zip(pts, node_cols):
        ax.add_patch(patches.Circle((px, py), 7, facecolor=c, edgecolor=COLORS["line"], lw=1.2, zorder=7))
    add_text(ax, x + 16, y + 31, "u", size=9, weight="bold")
    add_text(ax, x + 60, y + 15, "i", size=9, weight="bold")


def small_mlp(ax, x, y, w=78, h=48, fill=COLORS["white"], edge=COLORS["green"]):
    poly = patches.Polygon(
        [[x, y], [x + w, y + 10], [x + w, y + h - 10], [x, y + h]],
        closed=True,
        facecolor=fill,
        edgecolor=edge,
        linewidth=2,
        zorder=7,
    )
    ax.add_patch(poly)
    add_text(ax, x + w / 2 + 3, y + h / 2, "MLP", size=13, weight="bold", color=COLORS["ink"])


def draw_icon_column(ax):
    # Item and user icon cell.
    rect(ax, 20, 155, 88, 158, COLORS["white"], edge=COLORS["line"], lw=1.4, radius=12, zorder=4)
    # Bag.
    rect(ax, 47, 179, 30, 35, "none", edge=COLORS["line"], lw=2.2, radius=4, zorder=6)
    ax.add_patch(patches.Arc((62, 179), 24, 26, theta1=180, theta2=360, lw=2.2, color=COLORS["line"], zorder=6))
    add_text(ax, 64, 235, "course", size=13, weight="bold")
    # User.
    ax.add_patch(patches.Circle((64, 263), 13, facecolor="none", edgecolor=COLORS["line"], lw=2.2, zorder=6))
    ax.add_patch(patches.Arc((64, 301), 52, 47, theta1=200, theta2=-20, lw=2.2, color=COLORS["line"], zorder=6))
    add_text(ax, 64, 306, "learner", size=13, weight="bold")

    # Text / graph cell.
    rect(ax, 20, 380, 88, 168, COLORS["white"], edge=COLORS["line"], lw=1.4, radius=12, zorder=4)
    rect(ax, 47, 404, 31, 45, "none", edge=COLORS["line"], lw=2.0, zorder=6)
    ax.plot([72, 78, 78], [404, 410, 449], color=COLORS["line"], lw=2, zorder=6)
    for k in range(4):
        ax.plot([54, 72], [417 + 7 * k, 417 + 7 * k], color=COLORS["line"], lw=1.3, zorder=6)
    add_text(ax, 64, 468, "text", size=13, weight="bold")
    # Knowledge icon.
    pts = [(45, 515), (65, 495), (82, 522), (60, 532)]
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=COLORS["line"], lw=1.5, zorder=6)
    for px, py in pts:
        ax.add_patch(patches.Circle((px, py), 5.5, facecolor=COLORS["green_soft"], edgecolor=COLORS["line"], lw=1.3, zorder=7))
    add_text(ax, 64, 548, "concepts", size=12, weight="bold")


def draw_left_panel(ax):
    rect(ax, 3, 72, 405, 628, COLORS["left_bg"], edge=COLORS["panel_line"], lw=2.6, zorder=1)
    add_text(ax, 40, 99, "Course Evidence Encoder", size=18, weight="bold", ha="left")
    draw_icon_column(ax)

    # Trainable and content features.
    vector_bar(ax, 124, 198, fills=["#eef2f7", "#c9d7e4", "#8bb6d3", "#1f77b4"])
    add_text(ax, 154, 242, "ID branch", size=11, weight="bold")
    vector_bar(ax, 124, 285, fills=["#f8e5d8", "#f1c5a4", "#e78d55", "#c44e25"])
    add_text(ax, 154, 329, "learner ID", size=11, weight="bold")
    vector_bar(ax, 124, 430, fills=["#eef7e8", "#cfe6c2", "#9cc17c", "#4e863e"])
    add_text(ax, 154, 474, "course text", size=11, weight="bold")
    vector_bar(ax, 124, 520, fills=["#fbf4d6", "#efe1a2", "#d6bb59", "#a5801e"])
    add_text(ax, 154, 562, "course knowledge", size=11, weight="bold", width=15)

    arrow(ax, 196, 209, 235, 209, lw=2.0)
    arrow(ax, 196, 296, 235, 296, lw=2.0)
    mini_network(ax, 238, 165, 136, 120)
    mini_network(ax, 238, 305, 136, 120)
    add_text(ax, 308, 156, "history graph", size=10.5, weight="bold")

    arrow(ax, 196, 441, 238, 441, lw=2.0)
    small_mlp(ax, 262, 401, edge=COLORS["green"])
    arrow(ax, 340, 425, 397, 425, lw=2.0)
    vector_bar(ax, 397, 414, fills=["#eef7e8", "#cfe6c2", "#9cc17c", "#4e863e"])
    add_text(ax, 434, 457, "c_i", size=12, weight="bold")

    arrow(ax, 196, 531, 238, 531, lw=2.0)
    small_mlp(ax, 262, 491, edge=COLORS["gold"])
    arrow(ax, 340, 515, 397, 515, lw=2.0)
    vector_bar(ax, 397, 504, fills=["#fbf4d6", "#efe1a2", "#d6bb59", "#a5801e"])
    add_text(ax, 434, 548, "signals", size=11, weight="bold")

    # Masking and fusion.
    rect(ax, 118, 112, 210, 45, COLORS["violet_soft"], edge="#d0cbe2", lw=2, zorder=5)
    add_text(ax, 223, 135, "training losses", size=15, weight="bold")
    arrow(ax, 328, 134, 382, 134, lw=2.6, style="<|-")

    rect(ax, 438, 158, 126, 94, COLORS["red_soft"], edge=COLORS["red"], lw=1.8, radius=8, ls=(0, (5, 4)), zorder=6)
    add_text(ax, 501, 184, "forced-cold", size=12, weight="bold")
    add_text(ax, 501, 203, "ID masking", size=12, weight="bold")
    add_text(ax, 501, 229, "xi = 1[p_i < 1]", size=10, weight="bold", color=COLORS["red"])
    arrow(ax, 376, 224, 438, 205, lw=2.0)
    arrow(ax, 438, 245, 408, 410, lw=1.8, rad=0.12)

    rect(ax, 438, 292, 126, 82, COLORS["violet_soft"], edge=COLORS["violet"], lw=1.8, radius=8, zorder=6)
    add_text(ax, 501, 313, "gated fusion", size=12, weight="bold")
    add_text(ax, 501, 334, "q_i = h_0", size=12.5, weight="bold", color=COLORS["violet"])
    add_text(ax, 501, 354, "masked ID + content", size=8.6, weight="bold")
    arrow(ax, 376, 340, 438, 333, lw=2.0)
    arrow(ax, 470, 425, 555, 374, lw=1.8, rad=-0.18)
    arrow(ax, 501, 252, 501, 292, lw=1.6)


def draw_middle_panel(ax):
    rect(ax, 420, 72, 710, 628, COLORS["mid_bg"], edge=COLORS["panel_line"], lw=2.6, zorder=1)
    add_text(ax, 462, 99, "Course-knowledge Guided User Simulation", size=18, weight="bold", ha="left")

    # Simulation boundary.
    rect(ax, 614, 214, 428, 350, "#c7e3f7", edge=COLORS["blue"], lw=3.0, radius=36, zorder=2)
    add_text(ax, 828, 244, "T-step simulator", size=16, weight="bold")

    # Retrieval and sampling row.
    rect(ax, 610, 117, 460, 80, COLORS["white"], edge=COLORS["line"], lw=1.5, radius=8, ls=(0, (4, 3)), zorder=5)
    add_text(ax, 640, 142, "retrieve top-M learners", size=11, weight="bold", ha="left")
    vector_bar(ax, 644, 156, w=76, h=18, fills=["#eef2f7", "#c9d7e4", "#8bb6d3", "#1f77b4"])
    vector_bar(ax, 734, 156, w=76, h=18, fills=["#f8e5d8", "#f1c5a4", "#e78d55", "#c44e25"])
    vector_bar(ax, 824, 156, w=76, h=18, fills=["#eef7e8", "#cfe6c2", "#9cc17c", "#4e863e"])
    rect(ax, 924, 132, 126, 44, COLORS["green_soft"], edge=COLORS["green"], lw=1.6, radius=8, zorder=6)
    add_text(ax, 987, 149, "course-aware", size=11, weight="bold")
    add_text(ax, 987, 166, "sample N users", size=10, weight="bold")
    arrow(ax, 560, 339, 614, 339, lw=2.6)
    arrow(ax, 568, 335, 622, 176, lw=1.55, rad=-0.18)
    arrow(ax, 900, 156, 924, 156, lw=1.7)

    # Reward/control branch.
    rect(ax, 474, 493, 290, 150, COLORS["white"], edge=COLORS["line"], lw=1.5, radius=8, ls=(0, (4, 3)), zorder=5)
    add_text(ax, 619, 518, "course-knowledge signals", size=14, weight="bold")
    chips = [
        ("concept +", COLORS["green_soft"], COLORS["green"]),
        ("prereq -", COLORS["orange_soft"], COLORS["orange"]),
        ("difficulty -", COLORS["gold_soft"], COLORS["gold"]),
        ("redundancy -", COLORS["rose_soft"], COLORS["rose"]),
    ]
    for idx, (label, fill, edge) in enumerate(chips):
        x = 497 + (idx % 2) * 128
        y = 544 + (idx // 2) * 45
        rect(ax, x, y, 112, 30, fill, edge=edge, lw=1.2, radius=7, zorder=6)
        add_text(ax, x + 56, y + 15, label, size=10.2, weight="bold")
    add_text(ax, 619, 626, "b(u,i) = w_c C - w_p P - w_d D - w_r R", size=10, weight="bold")
    arrow(ax, 764, 560, 930, 176, color=COLORS["green"], lw=2.0, rad=-0.16)

    # Actor-critic and state transition.
    rect(ax, 678, 290, 116, 56, COLORS["violet_soft"], edge=COLORS["violet"], lw=1.8, radius=8, zorder=6)
    add_text(ax, 736, 309, "actor-critic", size=12, weight="bold")
    add_text(ax, 736, 329, "select a_t", size=10, weight="bold")
    arrow(ax, 987, 176, 818, 286, lw=2.0, rad=0.16)

    rect(ax, 830, 280, 150, 76, COLORS["white"], edge=COLORS["line"], lw=1.5, radius=8, zorder=6)
    add_text(ax, 905, 303, "state update", size=12, weight="bold")
    add_text(ax, 905, 324, "h_t -> h_t+1", size=11, weight="bold")
    add_text(ax, 905, 344, "target anchor alpha_t", size=9.5, weight="bold", color=COLORS["muted"])
    arrow(ax, 794, 318, 830, 318, lw=2.0)

    # State progression.
    for i, label in enumerate(["h_1", "h_2", "...", "h_T"]):
        x = 663 + i * 88
        y = 401
        if label == "...":
            add_text(ax, x + 18, y + 18, "...", size=17, weight="bold")
            continue
        rect(ax, x, y, 52, 62, COLORS["white"], edge=COLORS["blue"], lw=1.5, radius=8, zorder=6)
        vector_bar(ax, x + 8, y + 12, w=36, h=12, fills=["#eaf2f8", "#b9d5e9", "#6aa0c8"])
        vector_bar(ax, x + 8, y + 33, w=36, h=12, fills=["#f6dfdf", "#de9eb0", "#a65c70"])
        add_text(ax, x + 26, y + 55, label, size=10, weight="bold")
    arrow(ax, 980, 332, 1008, 405, lw=2.0, rad=0.12)
    arrow(ax, 1008, 433, 968, 433, lw=2.0, style="<|-")
    arrow(ax, 715, 463, 715, 492, lw=1.7, color=COLORS["green"], style="<|-")
    arrow(ax, 905, 356, 905, 401, lw=1.8)
    arrow(ax, 715, 463, 895, 358, color=COLORS["violet"], lw=1.45, ls=(0, (5, 4)), rad=-0.28)

    # Reward expression inside simulator.
    rect(ax, 802, 493, 214, 48, COLORS["red_soft"], edge=COLORS["red"], lw=1.4, radius=8, ls=(0, (4, 3)), zorder=6)
    add_text(ax, 909, 511, "step reward", size=11.5, weight="bold")
    add_text(ax, 909, 530, "preserve target + progress + C - P - D - R", size=8.7, weight="bold")
    arrow(ax, 909, 493, 905, 356, color=COLORS["red"], lw=1.8, rad=-0.12)

    add_text(
        ax,
        742,
        681,
        "training-time simulated learner-course matching",
        size=9.5,
        color=COLORS["muted"],
        weight="bold",
    )


def draw_right_panel(ax):
    rect(ax, 1142, 72, 356, 628, COLORS["right_bg"], edge=COLORS["panel_line"], lw=2.6, zorder=1)
    rect(ax, 1150, 84, 340, 158, COLORS["right_top_bg"], edge=COLORS["panel_line"], lw=1.8, zorder=2)
    add_text(ax, 1180, 113, "Strict Item-cold Protocol", size=16, weight="bold", ha="left")
    rect(ax, 1168, 136, 284, 82, COLORS["white"], edge=COLORS["line"], lw=1.5, radius=8, ls=(0, (4, 3)), zorder=5)
    add_text(ax, 1310, 160, "zero target-course interactions", size=12, weight="bold")
    add_text(ax, 1310, 181, "full-catalog competition", size=11, weight="bold")
    add_text(ax, 1310, 202, "item-macro Recall / NDCG", size=11, weight="bold", color=COLORS["red"])

    add_text(ax, 1160, 278, "Ranking Phase", size=18, weight="bold", ha="left")
    rect(ax, 1168, 302, 284, 124, COLORS["white"], edge=COLORS["line"], lw=1.5, radius=8, ls=(0, (4, 3)), zorder=5)
    add_text(ax, 1188, 328, "final course state", size=10.5, weight="bold", ha="left")
    vector_bar(ax, 1300, 316, w=82, h=20, fills=["#eaf2f8", "#b9d5e9", "#6aa0c8", "#2468a2"])
    add_text(ax, 1341, 348, "h_T", size=11, weight="bold")
    add_text(ax, 1188, 376, "learner representation", size=10.5, weight="bold", ha="left")
    vector_bar(ax, 1300, 364, w=82, h=20, fills=["#f8e5d8", "#f1c5a4", "#e78d55", "#c44e25"])
    add_text(ax, 1341, 396, "z_u", size=11, weight="bold")
    arrow(ax, 1382, 340, 1422, 340, lw=1.8)
    arrow(ax, 1382, 388, 1422, 388, lw=1.8)
    add_text(ax, 1426, 365, "score", size=10.5, weight="bold", rotation=90)
    add_text(ax, 1310, 414, "ell(u,i) = cos(z_u, h_T) / tau", size=10, weight="bold")

    add_text(ax, 1160, 472, "Training Objectives", size=18, weight="bold", ha="left")
    rect(ax, 1168, 494, 284, 156, COLORS["white"], edge=COLORS["line"], lw=1.5, radius=8, ls=(0, (4, 3)), zorder=5)
    rows = [
        ("CE ranking", COLORS["blue_soft"], COLORS["blue"]),
        ("PPO / GAE", COLORS["violet_soft"], COLORS["violet"]),
        ("ID-content InfoNCE", COLORS["green_soft"], COLORS["green"]),
        ("prereq margin loss", COLORS["orange_soft"], COLORS["orange"]),
    ]
    for i, (label, fill, edge) in enumerate(rows):
        y = 514 + 32 * i
        rect(ax, 1190, y, 214, 22, fill, edge=edge, lw=1.0, radius=5, zorder=6)
        add_text(ax, 1297, y + 11, label, size=10.5, weight="bold")
    add_text(ax, 1310, 636, "L = L_rank + L_PPO + L_aux + L_pre", size=9.8, weight="bold")

    # Vertical note similar to reference's easy-to-hard strip.
    rect(ax, 1461, 303, 21, 125, COLORS["red_soft"], edge=COLORS["red"], lw=1.4, zorder=6)
    add_text(ax, 1472, 365, "cold-risk first", size=10, weight="bold", color=COLORS["red"], rotation=90)


def draw_bottom_strip(ax):
    rect(ax, 456, 719, 588, 35, COLORS["white"], edge=COLORS["line"], lw=1.2, radius=8, zorder=5)
    add_text(
        ax,
        750,
        736.5,
        "content anchoring + simulated matching + educational constraints",
        size=9.6,
        weight="bold",
    )


def make_figure():
    fig = plt.figure(figsize=(15, 7.8), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    rect(ax, 0, 0, W, H, COLORS["white"], edge=COLORS["white"], lw=0)

    dashed_backbone(ax)
    draw_left_panel(ax)
    draw_middle_panel(ax)
    draw_right_panel(ax)
    stage_arrow(ax, 407, 407, 439, 407, label="represent")
    stage_arrow(ax, 1114, 407, 1143, 407, label="rank")
    draw_bottom_strip(ax)

    # Panel-level labels for manuscript reference.
    for label, x in [("a", 17), ("b", 434), ("c", 1158)]:
        add_text(ax, x, 92, label, size=16, weight="bold", color=COLORS["black"])

    return fig


def save_outputs(fig, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    exports = [
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 300}),
        (".tiff", {"dpi": 600}),
    ]
    for suffix, kwargs in exports:
        final_path = out_base.with_suffix(suffix)
        tmp_path = out_base.with_name(f"{out_base.name}_tmp").with_suffix(suffix)
        if tmp_path.exists():
            tmp_path.unlink()
        fig.savefig(tmp_path, bbox_inches="tight", pad_inches=0.035, **kwargs)
        if final_path.exists():
            final_path.unlink()
        tmp_path.replace(final_path)


def main():
    out_base = Path(__file__).resolve().parent / "ckg_rl_framework"
    fig = make_figure()
    save_outputs(fig, out_base)
    plt.close(fig)
    print(f"saved: {out_base.with_suffix('.svg')}")
    print(f"saved: {out_base.with_suffix('.pdf')}")
    print(f"saved: {out_base.with_suffix('.png')}")
    print(f"saved: {out_base.with_suffix('.tiff')}")


if __name__ == "__main__":
    main()
