import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 9,
        "legend.frameon": False,
    }
)


W, H = 1600, 720

COL = {
    "ink": "#172033",
    "muted": "#5a6578",
    "line": "#273241",
    "panel_line": "#94a0af",
    "paper": "#ffffff",
    "left": "#e4f0da",
    "middle": "#dbeefa",
    "right": "#e9e7f5",
    "soft": "#f7fafc",
    "blue": "#1f6fa8",
    "blue_soft": "#dceefa",
    "green": "#4f853f",
    "green_soft": "#e4f2df",
    "violet": "#6f5ea3",
    "violet_soft": "#ebe8f5",
    "orange": "#c45f2a",
    "orange_soft": "#f4d7bf",
    "gold": "#b89129",
    "gold_soft": "#f4e8bd",
    "red": "#c61b2a",
    "red_soft": "#fff3f4",
    "gray": "#e8edf3",
}

PANELS = {
    "a": {"x": 30, "y": 58, "w": 360, "h": 540, "fill": COL["left"]},
    "b": {"x": 412, "y": 58, "w": 770, "h": 540, "fill": COL["middle"]},
    "c": {"x": 1204, "y": 58, "w": 366, "h": 540, "fill": COL["right"]},
}

LAYOUT_SPEC = {
    "canvas": {"width": W, "height": H},
    "panels": {
        "a": {
            "title": "Cold-course evidence encoder",
            "blocks": [
                "target cold course i -> content x_i -> MLP -> gated item vector",
                "cold mask / ID masked -> gated item vector",
                "training history only -> MLP -> z_u",
            ],
        },
        "b": {
            "title": "Course-knowledge guided simulation",
            "layers": [
                "top-M item-vector retrieval -> sampled candidates S_i",
                "actor-critic policy -> grad update",
                "h_0 -> h_1 -> h_2 -> ... -> h_T -> cached item bank",
            ],
        },
        "c": {
            "title": "Strict Item-Cold Ranking",
            "layers": [
                "strict item-cold protocol",
                "z_u and z_j^cold -> cosine scoring -> ranked list",
                "Recall@K and NDCG@K evaluation output",
            ],
        },
    },
    "visual_grammar": {
        "black_solid": "forward computation",
        "green_solid": "cold-side retrieval / course-knowledge guidance",
        "red_dashed": "training optimization signal",
        "red_top_route": "training-only feedback route, styled after the reference layout",
    },
}


def txt(ax, x, y, s, size=10, weight="normal", color=None, ha="center", va="center", z=20):
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


def panel(ax, p):
    ax.add_patch(
        patches.Rectangle(
            (p["x"], p["y"]),
            p["w"],
            p["h"],
            linewidth=3.0,
            edgecolor=COL["panel_line"],
            facecolor=p["fill"],
            zorder=1,
        )
    )


def box(ax, x, y, w, h, fc=COL["paper"], ec=None, lw=1.8, r=7, ls="-", z=4):
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


def arrow(ax, pts, color=None, lw=2.25, ms=15, ls="-", z=12):
    color = color or COL["line"]
    if len(pts) < 2:
        raise ValueError("arrow requires at least two points")
    if len(pts) > 2:
        xs, ys = zip(*pts[:-1])
        ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, zorder=z)
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
        zorder=z,
    )
    ax.add_patch(arr)
    return arr


def red_boundary_arrow(ax, x, y, scale=1.0, z=24):
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
            linewidth=1.8,
            zorder=z,
        )
    )


def draw_top_training_route(ax):
    y = 30
    x0, x1 = 150, 1208
    ls = (0, (6, 5))
    ax.plot([x0, x1], [y, y], color=COL["red"], linewidth=2.0, linestyle=ls, zorder=22)
    for x in [150, 705, 1208]:
        ax.plot([x, x], [y, 58], color=COL["red"], linewidth=2.0, linestyle=ls, zorder=22)
    for x in [340, 700, 1045]:
        ax.add_patch(
            FancyArrowPatch(
                (x + 56, y - 11),
                (x + 8, y - 11),
                arrowstyle="->",
                mutation_scale=17,
                linewidth=1.75,
                color=COL["red"],
                zorder=23,
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
                zorder=23,
            )
        )
    txt(
        ax,
        975,
        46,
        "training-only feedback",
        size=8.2,
        weight="bold",
        color=COL["red"],
        ha="left",
        z=24,
    )


def vector(ax, x, y, w=64, h=18, colors=None, z=8):
    colors = colors or ["#f7fafc", "#d9e6f2", "#8fb9d8", "#2f7db7"]
    n = len(colors)
    for i, c in enumerate(colors):
        ax.add_patch(
            patches.Rectangle(
                (x + i * w / n, y),
                w / n,
                h,
                facecolor=c,
                edgecolor=COL["line"],
                linewidth=1.0,
                zorder=z,
            )
        )
    ax.add_patch(
        patches.Rectangle((x, y), w, h, facecolor="none", edgecolor=COL["line"], linewidth=0.9, zorder=z + 1)
    )


def snowflake(ax, cx, cy, r=7, color="#367fa8", lw=1.25, z=18):
    for angle in [0, 60, 120]:
        x0 = cx - r * mpl.transforms.Affine2D().rotate_deg(angle).transform((1, 0))[0]
        y0 = cy - r * mpl.transforms.Affine2D().rotate_deg(angle).transform((1, 0))[1]
        x1 = cx + r * mpl.transforms.Affine2D().rotate_deg(angle).transform((1, 0))[0]
        y1 = cy + r * mpl.transforms.Affine2D().rotate_deg(angle).transform((1, 0))[1]
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, zorder=z)


def course_icon(ax, x, y, w=64, h=64):
    box(ax, x, y, w, h, COL["paper"], lw=1.6, r=6, z=6)
    cx = x + w / 2
    ax.plot([cx, cx], [y + 16, y + h - 12], color=COL["line"], linewidth=1.35, zorder=9)
    left = patches.FancyBboxPatch(
        (x + 15, y + 16),
        16,
        30,
        boxstyle="round,pad=0.02,rounding_size=3",
        facecolor="#f8fafc",
        edgecolor=COL["line"],
        linewidth=1.25,
        zorder=8,
    )
    right = patches.FancyBboxPatch(
        (x + 33, y + 16),
        16,
        30,
        boxstyle="round,pad=0.02,rounding_size=3",
        facecolor="#f8fafc",
        edgecolor=COL["line"],
        linewidth=1.25,
        zorder=8,
    )
    ax.add_patch(left)
    ax.add_patch(right)
    for yy in [y + 25, y + 34]:
        ax.plot([x + 20, x + 27], [yy, yy + 1.5], color=COL["muted"], linewidth=1.1, zorder=9)
        ax.plot([x + 38, x + 45], [yy + 1.5, yy], color=COL["muted"], linewidth=1.1, zorder=9)


def user_icon(ax, x, y, w=96, h=36):
    box(ax, x, y, w, h, COL["paper"], lw=1.5, r=6, z=6)
    ax.add_patch(patches.Circle((x + 18, y + 14), 7, fill=False, edgecolor=COL["line"], linewidth=1.35, zorder=9))
    ax.add_patch(patches.Arc((x + 18, y + 31), 24, 21, theta1=200, theta2=-20, color=COL["line"], linewidth=1.35, zorder=9))


def mlp(ax, x, y, w=54, h=48, color=COL["blue"]):
    pts = [(x + 9, y), (x + w, y + 8), (x + w - 9, y + h), (x, y + h - 8)]
    ax.add_patch(patches.Polygon(pts, closed=True, facecolor=COL["paper"], edgecolor=color, linewidth=1.85, zorder=7))
    txt(ax, x + w / 2, y + h / 2, "MLP", size=9.5, weight="bold", color=COL["ink"], z=10)


def gate_icon(ax, x, y, w=74, h=56):
    box(ax, x, y, w, h, COL["violet_soft"], ec=COL["violet"], lw=1.45, r=6, z=7)
    for dx, dy in [(20, 18), (20, 36), (39, 27), (55, 18), (55, 36)]:
        ax.add_patch(patches.Circle((x + dx, y + dy), 3.2, facecolor=COL["paper"], edgecolor=COL["violet"], linewidth=1.1, zorder=10))
    ax.plot([x + 23, x + 39, x + 52], [y + 18, y + 27, y + 18], color=COL["violet"], linewidth=1.15, zorder=9)
    ax.plot([x + 23, x + 39, x + 52], [y + 36, y + 27, y + 36], color=COL["violet"], linewidth=1.15, zorder=9)
    txt(ax, x + w / 2, y + h + 14, "gated item", size=8.7, weight="bold", color=COL["violet"])
    txt(ax, x + w / 2, y + h + 28, "vector", size=8.7, weight="bold", color=COL["violet"])


def hidden_state(ax, x, y, label):
    box(ax, x, y, 78, 58, COL["paper"], ec=COL["blue"], lw=1.8, r=8, z=7)
    vector(ax, x + 15, y + 14, w=48, h=13, colors=["#f8fafc", "#d9e6f2", "#8fb9d8"], z=9)
    vector(ax, x + 15, y + 32, w=48, h=13, colors=["#f8fafc", "#d9e6f2", "#8fb9d8"], z=9)
    txt(ax, x + 39, y + 74, label, size=9.6, weight="bold")


def cached_bank(ax, x, y):
    box(ax, x, y, 92, 66, COL["paper"], ec=COL["panel_line"], lw=1.5, r=7, z=7)
    for k, colors in enumerate(
        [
            ["#f8fafc", "#d9e6f2", "#8fb9d8", "#2f7db7"],
            ["#f8fafc", "#e3f2d9", "#a3cf94", "#4f853f"],
            ["#f8fafc", "#f4d7bf", "#ee9f62", "#c45f2a"],
        ]
    ):
        vector(ax, x + 16, y + 12 + 17 * k, w=58, h=11, colors=colors, z=9)
    txt(ax, x + 46, y + 84, "cached item bank", size=8.4, weight="bold", color=COL["muted"])


def cosine_icon(ax, cx, cy):
    ax.add_patch(patches.Circle((cx, cy), 23, facecolor=COL["paper"], edgecolor=COL["panel_line"], linewidth=1.2, zorder=8))
    ax.plot([cx - 13, cx + 14], [cy + 11, cy - 12], color=COL["line"], linewidth=1.25, zorder=10)
    ax.add_patch(FancyArrowPatch((cx - 2, cy + 7), (cx + 13, cy - 6), arrowstyle="-|>", mutation_scale=10, color=COL["blue"], linewidth=1.1, zorder=11))
    ax.add_patch(FancyArrowPatch((cx - 2, cy + 7), (cx + 2, cy - 12), arrowstyle="-|>", mutation_scale=10, color=COL["gold"], linewidth=1.1, zorder=11))
    txt(ax, cx, cy + 38, "cos", size=10.5, weight="bold")


def draw_headers(ax):
    for letter, title in [
        ("a", "Cold-course evidence encoder"),
        ("b", "Course-knowledge guided simulation"),
        ("c", "Strict Item-Cold Ranking"),
    ]:
        p = PANELS[letter]
        txt(ax, p["x"] + 28, p["y"] + 31, letter, size=18, weight="bold")
        txt(ax, p["x"] + 62, p["y"] + 31, title, size=16.5, weight="bold", ha="left")


def draw_panel_a(ax):
    p = PANELS["a"]
    box(ax, p["x"] + 30, p["y"] + 74, 300, 318, COL["paper"], ec="#c7d2c1", lw=1.45, r=8, z=3)
    txt(ax, p["x"] + 53, p["y"] + 96, "target cold course i", size=9.5, weight="bold", color=COL["muted"], ha="left")
    snowflake(ax, p["x"] + 205, p["y"] + 96, r=6)

    course_icon(ax, p["x"] + 48, p["y"] + 125)
    txt(ax, p["x"] + 80, p["y"] + 204, "course i", size=9.1, weight="bold")

    vector(ax, p["x"] + 140, p["y"] + 148, w=58, h=19)
    txt(ax, p["x"] + 169, p["y"] + 181, "content $x_i$", size=8.8, weight="bold")
    mlp(ax, p["x"] + 218, p["y"] + 128, w=52, h=55, color=COL["blue"])
    gate_icon(ax, p["x"] + 278, p["y"] + 127, w=64, h=56)

    arrow(ax, [(p["x"] + 112, p["y"] + 157), (p["x"] + 140, p["y"] + 157)], lw=2.0, ms=13)
    arrow(ax, [(p["x"] + 198, p["y"] + 157), (p["x"] + 218, p["y"] + 157)], lw=2.0, ms=13)
    arrow(ax, [(p["x"] + 270, p["y"] + 157), (p["x"] + 278, p["y"] + 157)], lw=2.0, ms=13)

    box(ax, p["x"] + 48, p["y"] + 252, 122, 70, COL["soft"], ec=COL["line"], lw=1.25, r=6, z=5)
    vector(ax, p["x"] + 66, p["y"] + 268, w=74, h=18, colors=["#f8fafc", "#dce3ec", "#b9c4d0", "#7d8998"], z=8)
    txt(ax, p["x"] + 109, p["y"] + 301, "cold mask / ID masked", size=8.3, weight="bold", color=COL["muted"])
    txt(ax, p["x"] + 109, p["y"] + 315, "$v_i=0$", size=9.0, weight="bold", color=COL["blue"])
    arrow(
        ax,
        [
            (p["x"] + 170, p["y"] + 286),
            (p["x"] + 205, p["y"] + 286),
            (p["x"] + 205, p["y"] + 183),
            (p["x"] + 278, p["y"] + 183),
        ],
        lw=2.05,
        ms=13,
    )

    txt(ax, p["x"] + 284, p["y"] + 220, "$c_i=\\mathrm{Encode}(x_i,v_i=0)$", size=8.0, weight="bold", color=COL["violet"])
    txt(ax, p["x"] + 284, p["y"] + 239, "$h_0=\\mathrm{Init}(c_i)$", size=8.7, weight="bold", color=COL["violet"])

    box(ax, p["x"] + 30, p["y"] + 414, 300, 94, COL["paper"], ec="#c2cad8", lw=1.45, r=8, z=3)
    txt(ax, p["x"] + 53, p["y"] + 436, "separate user encoder", size=9.4, weight="bold", color=COL["muted"], ha="left")
    user_icon(ax, p["x"] + 52, p["y"] + 454, w=116, h=38)
    txt(ax, p["x"] + 124, p["y"] + 468, "training history", size=7.7, weight="bold", color=COL["muted"])
    txt(ax, p["x"] + 124, p["y"] + 482, "only", size=7.7, weight="bold", color=COL["muted"])
    mlp(ax, p["x"] + 190, p["y"] + 448, w=52, h=50, color=COL["orange"])
    vector(ax, p["x"] + 266, p["y"] + 463, w=54, h=18, colors=["#f8fafc", "#f0d2bd", "#ee9f62", "#c84f1e"], z=8)
    txt(ax, p["x"] + 293, p["y"] + 496, "$z_u$", size=9.5, weight="bold", color=COL["orange"])
    arrow(ax, [(p["x"] + 168, p["y"] + 473), (p["x"] + 190, p["y"] + 473)], lw=2.0, ms=13)
    arrow(ax, [(p["x"] + 242, p["y"] + 473), (p["x"] + 266, p["y"] + 473)], lw=2.0, ms=13)


def draw_panel_b(ax):
    p = PANELS["b"]
    ax.add_patch(
        patches.FancyBboxPatch(
            (p["x"] + 52, p["y"] + 58),
            666,
            398,
            boxstyle="round,pad=0.02,rounding_size=18",
            linewidth=2.7,
            edgecolor=COL["blue"],
            facecolor="none",
            zorder=2,
        )
    )
    txt(ax, p["x"] + 385, p["y"] + 79, "T-step learner-course simulator", size=13.8, weight="bold", z=4)
    box(ax, p["x"] + 60, p["y"] + 82, 650, 102, COL["paper"], ec="#9fc795", lw=1.35, r=8, z=3)
    snowflake(ax, p["x"] + 82, p["y"] + 113, r=6)
    txt(ax, p["x"] + 102, p["y"] + 113, "cold-side learner retrieval", size=10.5, weight="bold", ha="left")
    box(ax, p["x"] + 150, p["y"] + 132, 190, 38, COL["soft"], ec=COL["green"], lw=1.25, r=6, z=6)
    txt(ax, p["x"] + 245, p["y"] + 151, "top-M item-vector retrieval", size=9.1, weight="bold")
    box(ax, p["x"] + 430, p["y"] + 132, 186, 38, COL["green_soft"], ec=COL["green"], lw=1.35, r=6, z=6)
    txt(ax, p["x"] + 523, p["y"] + 151, "sampled candidates $S_i$", size=9.3, weight="bold", color=COL["green"])
    arrow(ax, [(p["x"] + 340, p["y"] + 151), (p["x"] + 430, p["y"] + 151)], color=COL["green"], lw=2.15, ms=14)

    box(ax, p["x"] + 178, p["y"] + 234, 220, 76, COL["paper"], ec=COL["blue"], lw=1.45, r=8, z=5)
    txt(ax, p["x"] + 288, p["y"] + 253, "actor-critic policy", size=10.5, weight="bold")
    txt(ax, p["x"] + 288, p["y"] + 275, "$a_t\\sim\\pi_\\theta(a\\mid h_t,S_i)$", size=9.1, weight="bold", color=COL["muted"])
    txt(ax, p["x"] + 288, p["y"] + 294, "choose learner action $a_t$", size=8.4, weight="bold", color=COL["muted"])

    box(ax, p["x"] + 458, p["y"] + 234, 220, 76, COL["paper"], ec=COL["blue"], lw=1.45, r=8, z=5)
    txt(ax, p["x"] + 568, p["y"] + 253, "grad update", size=10.5, weight="bold")
    txt(ax, p["x"] + 568, p["y"] + 275, "$h_{t+1}=h_t+\\eta\\nabla_{h_t}A_t$", size=8.7, weight="bold", color=COL["muted"])
    txt(ax, p["x"] + 568, p["y"] + 294, "target anchor $\\alpha_t$", size=8.4, weight="bold", color=COL["muted"])

    arrow(ax, [(p["x"] + 523, p["y"] + 170), (p["x"] + 523, p["y"] + 219), (p["x"] + 288, p["y"] + 219), (p["x"] + 288, p["y"] + 234)], color=COL["green"], lw=2.0, ms=13)
    arrow(ax, [(p["x"] + 398, p["y"] + 272), (p["x"] + 458, p["y"] + 272)], lw=2.05, ms=14)

    txt(
        ax,
        p["x"] + 378,
        p["y"] + 322,
        "$A_t=(1-\\alpha_t)h_t^\\top z_{a_t}+\\alpha_t h_t^\\top c_i$",
        size=8.6,
        weight="bold",
        color=COL["muted"],
    )

    y = p["y"] + 345
    hidden_state(ax, p["x"] + 78, y, "h_0")
    hidden_state(ax, p["x"] + 205, y, "h_1")
    hidden_state(ax, p["x"] + 332, y, "h_2")
    txt(ax, p["x"] + 474, y + 28, "...", size=20, weight="bold", color=COL["muted"])
    hidden_state(ax, p["x"] + 525, y, "h_T")
    cached_bank(ax, p["x"] + 635, y - 4)
    arrow(ax, [(p["x"] + 156, y + 29), (p["x"] + 205, y + 29)], lw=2.15, ms=14)
    arrow(ax, [(p["x"] + 283, y + 29), (p["x"] + 332, y + 29)], lw=2.15, ms=14)
    arrow(ax, [(p["x"] + 410, y + 29), (p["x"] + 454, y + 29)], lw=2.15, ms=14)
    arrow(ax, [(p["x"] + 494, y + 29), (p["x"] + 525, y + 29)], lw=2.15, ms=14)
    arrow(ax, [(p["x"] + 603, y + 29), (p["x"] + 635, y + 29)], lw=2.15, ms=14)

    box(ax, p["x"] + 82, p["y"] + 464, 292, 72, COL["paper"], ec=COL["line"], lw=1.35, r=8, z=4)
    txt(ax, p["x"] + 228, p["y"] + 484, "reward terms", size=10.1, weight="bold")
    txt(ax, p["x"] + 228, p["y"] + 509, "concept bonus / prereq gap", size=8.5, weight="bold", color=COL["muted"])
    txt(ax, p["x"] + 228, p["y"] + 526, "difficulty gap / redundancy penalty", size=8.5, weight="bold", color=COL["muted"])

    box(ax, p["x"] + 396, p["y"] + 464, 292, 72, COL["paper"], ec=COL["line"], lw=1.35, r=8, z=4)
    txt(ax, p["x"] + 542, p["y"] + 484, "training losses", size=10.1, weight="bold")
    txt(ax, p["x"] + 542, p["y"] + 508, "rank CE + InfoNCE aux", size=8.5, weight="bold", color=COL["red"])
    txt(ax, p["x"] + 542, p["y"] + 526, "PPO + prereq aux", size=8.5, weight="bold", color=COL["red"])

    red_ls = (0, (6, 5))
    arrow(ax, [(p["x"] + 172, p["y"] + 464), (p["x"] + 172, p["y"] + 326), (p["x"] + 250, p["y"] + 326), (p["x"] + 250, p["y"] + 310)], color=COL["red"], lw=1.9, ms=12, ls=red_ls, z=6)
    arrow(ax, [(p["x"] + 604, p["y"] + 464), (p["x"] + 604, p["y"] + 326), (p["x"] + 568, p["y"] + 326), (p["x"] + 568, p["y"] + 310)], color=COL["red"], lw=1.9, ms=12, ls=red_ls, z=6)


def draw_panel_c(ax):
    p = PANELS["c"]
    box(ax, p["x"] + 28, p["y"] + 86, 310, 76, "#fbf4dc", ec=COL["panel_line"], lw=1.45, r=7, z=3)
    snowflake(ax, p["x"] + 57, p["y"] + 112, r=6)
    txt(ax, p["x"] + 183, p["y"] + 112, "strict item-cold protocol", size=12.0, weight="bold")
    txt(ax, p["x"] + 183, p["y"] + 139, "zero target-course interactions", size=8.6, weight="bold", color=COL["muted"])
    txt(ax, p["x"] + 183, p["y"] + 154, "full-catalog, item-macro ranking", size=8.4, weight="bold", color=COL["muted"])

    box(ax, p["x"] + 28, p["y"] + 188, 310, 242, COL["paper"], ec=COL["line"], lw=1.55, r=8, z=3)
    txt(ax, p["x"] + 50, p["y"] + 211, "cosine scoring", size=10.8, weight="bold", ha="left")

    box(ax, p["x"] + 48, p["y"] + 238, 96, 52, COL["soft"], ec=COL["line"], lw=1.15, r=6, z=5)
    vector(ax, p["x"] + 62, p["y"] + 252, w=68, h=17, colors=["#f8fafc", "#f0d2bd", "#ee9f62", "#c84f1e"], z=8)
    txt(ax, p["x"] + 96, p["y"] + 281, "$z_u$", size=9.4, weight="bold", color=COL["orange"])
    txt(ax, p["x"] + 96, p["y"] + 302, "from user encoder", size=7.9, weight="bold", color=COL["muted"])

    box(ax, p["x"] + 48, p["y"] + 340, 96, 52, COL["soft"], ec=COL["line"], lw=1.15, r=6, z=5)
    vector(ax, p["x"] + 62, p["y"] + 354, w=68, h=17, z=8)
    txt(ax, p["x"] + 96, p["y"] + 383, "$z_j^{cold}$", size=9.1, weight="bold", color=COL["blue"])
    txt(ax, p["x"] + 96, p["y"] + 404, "from cached bank", size=7.9, weight="bold", color=COL["muted"])

    box(ax, p["x"] + 170, p["y"] + 268, 82, 82, COL["violet_soft"], ec=COL["violet"], lw=1.35, r=8, z=5)
    cosine_icon(ax, p["x"] + 211, p["y"] + 300)

    box(ax, p["x"] + 274, p["y"] + 246, 38, 132, COL["soft"], ec=COL["line"], lw=1.15, r=5, z=5)
    for k, yy in enumerate([258, 281, 304, 327, 350]):
        fill = COL["blue_soft"] if k == 1 else "#d9dee7"
        box(ax, p["x"] + 281, p["y"] + yy, 24, 12, fill, ec=COL["line"], lw=0.8, r=2, z=7)
    txt(ax, p["x"] + 293, p["y"] + 393, "ranked list", size=8.6, weight="bold", color=COL["muted"])

    arrow(ax, [(p["x"] + 144, p["y"] + 264), (p["x"] + 156, p["y"] + 264), (p["x"] + 156, p["y"] + 288), (p["x"] + 170, p["y"] + 288)], lw=1.9, ms=12)
    arrow(ax, [(p["x"] + 144, p["y"] + 366), (p["x"] + 156, p["y"] + 366), (p["x"] + 156, p["y"] + 326), (p["x"] + 170, p["y"] + 326)], lw=1.9, ms=12)
    arrow(ax, [(p["x"] + 252, p["y"] + 309), (p["x"] + 274, p["y"] + 309)], lw=2.05, ms=13)
    txt(ax, p["x"] + 183, p["y"] + 415, "$s(u,j)=\\cos(z_u,z_j^{cold})/\\tau$", size=9.4, weight="bold")

    box(ax, p["x"] + 28, p["y"] + 452, 310, 68, COL["paper"], ec=COL["line"], lw=1.45, r=8, z=3)
    txt(ax, p["x"] + 48, p["y"] + 472, "evaluation output", size=10.6, weight="bold", ha="left")
    txt(ax, p["x"] + 318, p["y"] + 472, "no training feedback", size=7.5, weight="bold", color=COL["muted"], ha="right")
    box(ax, p["x"] + 54, p["y"] + 490, 106, 22, COL["blue_soft"], ec=COL["blue"], lw=1.1, r=5, z=6)
    txt(ax, p["x"] + 107, p["y"] + 501, "Recall@K", size=8.7, weight="bold")
    box(ax, p["x"] + 206, p["y"] + 490, 106, 22, COL["violet_soft"], ec=COL["violet"], lw=1.1, r=5, z=6)
    txt(ax, p["x"] + 259, p["y"] + 501, "NDCG@K", size=8.7, weight="bold")


def draw_cross_panel_routes(ax):
    arrow(ax, [(390, 432), (490, 432)], lw=2.45, ms=16, z=14)
    red_boundary_arrow(ax, 394, 432, scale=0.55, z=18)
    txt(ax, 440, 414, "$h_0$", size=8.6, weight="bold", color=COL["muted"])
    arrow(ax, [(1119, 432), (1228, 432), (1228, 424), (1252, 424)], lw=2.45, ms=16, z=14)
    red_boundary_arrow(ax, 1185, 424, scale=0.55, z=18)


def draw_legend(ax):
    box(ax, 490, 622, 620, 50, COL["paper"], ec=COL["panel_line"], lw=1.25, r=7, z=20)
    y = 641
    ax.plot([520, 570], [y, y], color=COL["line"], linewidth=2.35, zorder=22)
    txt(ax, 582, y, "forward computation", size=8.4, weight="bold", color=COL["muted"], ha="left", z=22)
    ax.plot([750, 800], [y, y], color=COL["green"], linewidth=2.35, zorder=22)
    txt(ax, 812, y, "cold-side retrieval / guidance", size=8.4, weight="bold", color=COL["muted"], ha="left", z=22)
    ax.plot([520, 570], [660, 660], color=COL["red"], linewidth=2.0, linestyle=(0, (6, 5)), zorder=22)
    txt(ax, 582, 660, "training optimization signal", size=8.4, weight="bold", color=COL["muted"], ha="left", z=22)


def draw_figure():
    fig = plt.figure(figsize=(16, 7.2), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.add_patch(patches.Rectangle((0, 0), W, H, facecolor=COL["paper"], edgecolor="none", zorder=0))
    for p in PANELS.values():
        panel(ax, p)
    draw_top_training_route(ax)
    draw_headers(ax)
    draw_panel_a(ax)
    draw_panel_b(ax)
    draw_panel_c(ax)
    draw_cross_panel_routes(ax)
    draw_legend(ax)
    return fig


def save_outputs(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in [(".svg", {}), (".pdf", {}), (".png", {"dpi": 300})]:
        final = base.with_suffix(suffix)
        tmp = base.with_name(base.name + "_tmp").with_suffix(suffix)
        if tmp.exists():
            tmp.unlink()
        fig.savefig(tmp, facecolor=COL["paper"], **kwargs)
        if final.exists():
            final.unlink()
        tmp.replace(final)


def save_spec(base: Path):
    spec_dir = base.parent / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"{base.name}_spec.json"
    spec_path.write_text(json.dumps(LAYOUT_SPEC, indent=2), encoding="utf-8")
    return spec_path


def main():
    base = Path(__file__).resolve().parent / "ckg_rl_framework_topconf_relayout"
    fig = draw_figure()
    save_outputs(fig, base)
    plt.close(fig)
    spec_path = save_spec(base)
    for suffix in [".svg", ".pdf", ".png"]:
        print(f"saved: {base.with_suffix(suffix)}")
    print(f"saved: {spec_path}")


if __name__ == "__main__":
    main()
