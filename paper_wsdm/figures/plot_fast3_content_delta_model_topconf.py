import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch


# Top-conference style overview for the current main code path:
# usim_feedback_fast3_content_delta.py::Fast3FeedbackUSIM.
# The layout is fixed and deterministic so SVG/PDF exports remain editable.

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


W, H = 2000, 960

COL = {
    "ink": "#172033",
    "muted": "#596679",
    "line": "#253244",
    "panel": "#94a0af",
    "paper": "#ffffff",
    "left": "#e5efdc",
    "middle": "#dcedf7",
    "right": "#ebe7f4",
    "bottom": "#f6f2e8",
    "soft": "#f8fafc",
    "blue": "#1f6fa8",
    "blue_soft": "#dceefa",
    "green": "#4d7f43",
    "green_soft": "#e4f2df",
    "violet": "#6655a1",
    "violet_soft": "#ebe8f5",
    "orange": "#b95f2e",
    "orange_soft": "#f4d7bf",
    "gold": "#ad8421",
    "gold_soft": "#f4e7bc",
    "red": "#c61b2a",
    "red_soft": "#fff0f1",
    "gray": "#e8edf3",
    "dark_green": "#2d5c3b",
}


FIGURE_SPEC = {
    "title": "USIM-Feedback FAST3 ContentDelta model overview",
    "source_files": [
        "usim_feedback_fast3_content_delta.py",
        "fast3_delta/config.py",
        "fast3_delta/course_artifacts.py",
        "fast3_delta/eval.py",
        "run_usim_feedback_fast3_content_delta_static.ps1",
    ],
    "main_class": "Fast3FeedbackUSIM",
    "panels": {
        "state": [
            "static item-cold split",
            "target course, user history, content embeddings, LLM scores, course relations",
        ],
        "A": [
            "course ID/content/LLM fusion",
            "bounded cold-only ContentDelta",
            "user history encoder and user bank",
        ],
        "B": [
            "top-M retrieval and knowledge-biased candidate sampling",
            "actor-critic rollout",
            "gradient state update and reward feedback",
        ],
        "C": [
            "strict item-cold full catalog scoring",
            "item-macro Recall@K and NDCG@K",
        ],
        "training": [
            "rank CE",
            "InfoNCE",
            "PPO",
            "course/prereq rewards",
            "PAAC",
            "SAGE aux",
            "CGRC recon aux",
            "ContentDelta regularization",
        ],
    },
    "visual_grammar": {
        "black": "forward computation",
        "green": "course-knowledge guidance and candidate sampling",
        "blue": "representation/state flow",
        "red_dashed": "training-only optimization or feedback",
        "orange": "user/action signal",
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
        linespacing=1.06,
        zorder=z,
    )


def panel(ax, x, y, w, h, fc, title, tag=None):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=18",
            linewidth=2.8,
            edgecolor=COL["panel"],
            facecolor=fc,
            zorder=1,
        )
    )
    if tag:
        txt(ax, x + 26, y + 34, tag, size=17, weight="bold")
        txt(ax, x + 64, y + 34, title, size=14.5, weight="bold", ha="left")
    else:
        txt(ax, x + w / 2, y + 34, title, size=14.5, weight="bold")


def box(ax, x, y, w, h, fc=None, ec=None, lw=1.55, r=7, ls="-", z=4):
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec or COL["line"],
        facecolor=fc or COL["paper"],
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, pts, color=None, lw=2.15, ms=14, ls="-", z=12, alpha=1.0):
    color = color or COL["line"]
    if len(pts) > 2:
        xs, ys = zip(*pts[:-1])
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=lw,
            linestyle=ls,
            solid_capstyle="butt",
            alpha=alpha,
            zorder=z,
        )
    ax.add_patch(
        FancyArrowPatch(
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
            zorder=z + 1,
        )
    )


def vector(ax, x, y, w=76, h=18, colors=None, z=8):
    colors = colors or ["#ffffff", "#dbe8f3", "#8eb9d7", "#2f7db7"]
    step = w / len(colors)
    for i, c in enumerate(colors):
        ax.add_patch(
            patches.Rectangle(
                (x + i * step, y),
                step,
                h,
                facecolor=c,
                edgecolor=COL["line"],
                linewidth=0.95,
                zorder=z,
            )
        )
    ax.add_patch(
        patches.Rectangle((x, y), w, h, facecolor="none", edgecolor=COL["line"], linewidth=1.05, zorder=z + 1)
    )


def snowflake(ax, cx, cy, r=7, color=None, lw=1.15, z=18):
    color = color or COL["blue"]
    for angle in [0, 60, 120]:
        t = mpl.transforms.Affine2D().rotate_deg(angle).transform((1, 0))
        x0, y0 = cx - r * t[0], cy - r * t[1]
        x1, y1 = cx + r * t[0], cy + r * t[1]
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, zorder=z)


def course_icon(ax, x, y, w=74, h=66, cold=False):
    box(ax, x, y, w, h, COL["paper"], ec=COL["line"], lw=1.35, r=7, z=6)
    cx = x + w / 2
    ax.plot([cx, cx], [y + 15, y + h - 14], color=COL["line"], linewidth=1.2, zorder=10)
    ax.add_patch(
        patches.FancyBboxPatch(
            (x + 16, y + 15),
            19,
            34,
            boxstyle="round,pad=0.02,rounding_size=3",
            facecolor="#fbfcfe",
            edgecolor=COL["line"],
            linewidth=1.1,
            zorder=9,
        )
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (x + 39, y + 15),
            19,
            34,
            boxstyle="round,pad=0.02,rounding_size=3",
            facecolor="#fbfcfe",
            edgecolor=COL["line"],
            linewidth=1.1,
            zorder=9,
        )
    )
    for yy in [y + 24, y + 34]:
        ax.plot([x + 21, x + 31], [yy, yy + 1.5], color=COL["muted"], linewidth=1.0, zorder=10)
        ax.plot([x + 43, x + 53], [yy + 1.5, yy], color=COL["muted"], linewidth=1.0, zorder=10)
    if cold:
        snowflake(ax, x + w - 13, y + 12, r=5.5, z=15)


def user_icon(ax, x, y, w=88, h=42):
    box(ax, x, y, w, h, COL["paper"], ec=COL["line"], lw=1.25, r=7, z=6)
    ax.add_patch(patches.Circle((x + 22, y + 15), 7, fill=False, edgecolor=COL["line"], linewidth=1.2, zorder=10))
    ax.add_patch(patches.Arc((x + 22, y + 33), 25, 20, theta1=198, theta2=-18, color=COL["line"], linewidth=1.2, zorder=10))


def database_icon(ax, x, y, w=62, h=56):
    box(ax, x, y, w, h, "#fbfbf7", ec=COL["line"], lw=1.2, r=5, z=6)
    for k in range(3):
        yy = y + 12 + k * 12
        ax.add_patch(
            patches.Ellipse((x + w / 2, yy), w - 18, 10, facecolor="#edf2f7", edgecolor=COL["line"], linewidth=1.0, zorder=8)
        )
        ax.plot([x + 9, x + 9], [yy, yy + 12], color=COL["line"], lw=1.0, zorder=8)
        ax.plot([x + w - 9, x + w - 9], [yy, yy + 12], color=COL["line"], lw=1.0, zorder=8)


def mlp(ax, x, y, w=58, h=54, label="MLP", color=None):
    color = color or COL["blue"]
    pts = [(x + 10, y), (x + w, y + 9), (x + w - 10, y + h), (x, y + h - 9)]
    ax.add_patch(patches.Polygon(pts, closed=True, facecolor=COL["paper"], edgecolor=color, linewidth=1.8, zorder=7))
    txt(ax, x + w / 2, y + h / 2, label, size=9.5, weight="bold", z=10)


def gate(ax, x, y, w=78, h=58, label="gate"):
    box(ax, x, y, w, h, COL["violet_soft"], ec=COL["violet"], lw=1.35, r=7, z=7)
    nodes = [(20, 18), (20, 40), (41, 29), (58, 18), (58, 40)]
    for dx, dy in nodes:
        ax.add_patch(patches.Circle((x + dx, y + dy), 3.1, facecolor=COL["paper"], edgecolor=COL["violet"], linewidth=1.0, zorder=10))
    ax.plot([x + 23, x + 41, x + 55], [y + 18, y + 29, y + 18], color=COL["violet"], linewidth=1.05, zorder=9)
    ax.plot([x + 23, x + 41, x + 55], [y + 40, y + 29, y + 40], color=COL["violet"], linewidth=1.05, zorder=9)
    txt(ax, x + w / 2, y + h + 14, label, size=8.6, weight="bold", color=COL["violet"])


def state_stack(ax, x, y, label, color=None):
    color = color or COL["blue"]
    box(ax, x, y, 86, 60, COL["paper"], ec=color, lw=1.65, r=8, z=7)
    vector(ax, x + 16, y + 14, w=54, h=13, z=9)
    vector(ax, x + 16, y + 34, w=54, h=13, colors=["#ffffff", "#e8f1dd", "#a8cc9f", "#4d7f43"], z=9)
    txt(ax, x + 43, y + 78, label, size=10.5, weight="bold", color=color)


def chip(ax, x, y, w, h, label, color, fc=None, size=8.6):
    box(ax, x, y, w, h, fc or COL["paper"], ec=color, lw=1.15, r=6, z=8)
    txt(ax, x + w / 2, y + h / 2, label, size=size, weight="bold", color=COL["ink"], z=12)


def red_boundary_arrow(ax, x, y, scale=1.0, z=25):
    pts = [
        (x - 19 * scale, y - 14 * scale),
        (x + 3 * scale, y - 14 * scale),
        (x + 3 * scale, y - 23 * scale),
        (x + 31 * scale, y),
        (x + 3 * scale, y + 23 * scale),
        (x + 3 * scale, y + 14 * scale),
        (x - 19 * scale, y + 14 * scale),
    ]
    ax.add_patch(patches.Polygon(pts, closed=True, facecolor=COL["paper"], edgecolor=COL["red"], linewidth=2.0, zorder=z))


def draw_state_band(ax):
    ax.add_patch(
        patches.FancyBboxPatch(
            (58, 26),
            1884,
            116,
            boxstyle="round,pad=0.02,rounding_size=18",
            linewidth=2.8,
            edgecolor=COL["panel"],
            facecolor="#f9faf8",
            zorder=1,
        )
    )
    txt(ax, 1000, 52, r"State $s$: strict item-cold split + target course + learner history", size=14.8, weight="bold")

    database_icon(ax, 92, 70, w=58, h=48)
    txt(ax, 121, 127, "MOOCCube/X", size=8.2, weight="bold")
    arrow(ax, [(158, 93), (198, 93)], color=COL["line"], lw=1.8, ms=12)

    box(ax, 205, 68, 205, 50, COL["blue_soft"], ec=COL["blue"], lw=1.2, r=6)
    txt(ax, 307, 88, "static split", size=10.4, weight="bold", color=COL["blue"])
    txt(ax, 307, 109, "train / val / test", size=8.7, color=COL["muted"])
    txt(ax, 307, 126, "cold iff train pop = 0", size=8.5, weight="bold", color=COL["blue"])

    arrow(ax, [(418, 93), (464, 93)], color=COL["line"], lw=1.8, ms=12)
    user_icon(ax, 472, 72, w=86, h=42)
    txt(ax, 521, 126, "user u", size=8.7, weight="bold")
    txt(ax, 591, 92, "+", size=18, weight="bold", color=COL["muted"])
    for j, (lab, col) in enumerate(
        [
            ("hist.", COL["orange_soft"]),
            ("course", COL["blue_soft"]),
            ("concept", COL["green_soft"]),
            ("LLM", COL["violet_soft"]),
            ("pop.", COL["gold_soft"]),
        ]
    ):
        box(ax, 626 + j * 96, 72, 78, 38, col, ec=COL["line"], lw=1.0, r=5, z=7)
        txt(ax, 665 + j * 96, 91, lab, size=8.8, weight="bold")
    arrow(ax, [(1112, 92), (1178, 92)], color=COL["line"], lw=1.9, ms=13)
    course_icon(ax, 1192, 64, w=78, h=62, cold=True)
    txt(ax, 1231, 133, "target course i", size=8.6, weight="bold", color=COL["blue"])

    box(ax, 1350, 70, 530, 48, COL["soft"], ec=COL["panel"], lw=1.15, r=6)
    txt(ax, 1615, 88, "course artifacts", size=9.3, weight="bold", color=COL["muted"])
    txt(
        ax,
        1615,
        106,
        "concept overlap, prerequisite graph, difficulty, redundancy",
        size=8.3,
        weight="bold",
        color=COL["dark_green"],
    )
    txt(ax, 1615, 123, "from fast3_delta.course_artifacts", size=7.9, color=COL["muted"])


def draw_encoder_panel(ax):
    x, y, w, h = 58, 168, 520, 608
    panel(ax, x, y, w, h, COL["left"], "Evidence Encoders", tag="A")

    box(ax, x + 34, y + 74, 452, 250, COL["paper"], ec="#b8caae", lw=1.35, r=10, z=3)
    txt(ax, x + 58, y + 98, "cold-course representation", size=10.4, weight="bold", color=COL["muted"], ha="left")
    course_icon(ax, x + 58, y + 125, cold=True)
    txt(ax, x + 95, y + 202, "$i$", size=10.8, weight="bold", color=COL["blue"])

    box(ax, x + 168, y + 118, 104, 48, COL["soft"], ec=COL["blue"], lw=1.15, r=6)
    vector(ax, x + 182, y + 132, w=74, h=17)
    txt(ax, x + 220, y + 184, "$x_i$ content", size=8.7, weight="bold", color=COL["blue"])
    mlp(ax, x + 312, y + 112, w=58, h=58, label="proj", color=COL["blue"])
    gate(ax, x + 398, y + 112, w=70, h=58, label="gate_net")
    arrow(ax, [(x + 132, y + 156), (x + 168, y + 142)], lw=1.9, ms=12)
    arrow(ax, [(x + 272, y + 142), (x + 312, y + 142)], lw=1.9, ms=12)
    arrow(ax, [(x + 370, y + 142), (x + 398, y + 142)], lw=1.9, ms=12)

    box(ax, x + 158, y + 218, 110, 44, "#f6f6f3", ec=COL["line"], lw=1.05, r=6)
    vector(ax, x + 178, y + 230, w=70, h=14, colors=["#ffffff", "#dce3ec", "#b8c3cf", "#7d8998"], z=8)
    txt(ax, x + 214, y + 278, "ID emb", size=8.2, weight="bold", color=COL["muted"])
    txt(ax, x + 214, y + 292, "masked for cold", size=7.8, color=COL["muted"])
    box(ax, x + 292, y + 218, 83, 44, COL["violet_soft"], ec=COL["violet"], lw=1.05, r=6)
    txt(ax, x + 334, y + 235, "LLM", size=9.2, weight="bold", color=COL["violet"])
    txt(ax, x + 334, y + 253, "$g(l_i)$", size=8.4, weight="bold", color=COL["violet"])
    arrow(ax, [(x + 268, y + 240), (x + 398, y + 160)], color=COL["muted"], lw=1.45, ms=11)
    arrow(ax, [(x + 375, y + 240), (x + 398, y + 160)], color=COL["violet"], lw=1.45, ms=11)

    box(ax, x + 75, y + 342, 410, 122, COL["paper"], ec=COL["green"], lw=1.35, r=10, z=3)
    txt(ax, x + 98, y + 365, "ContentDelta residual", size=10.2, weight="bold", color=COL["dark_green"], ha="left")
    chip(ax, x + 98, y + 388, 96, 42, "delta_i\nembedding", COL["green"], COL["green_soft"], size=8.0)
    chip(ax, x + 218, y + 388, 118, 42, "delta MLP\nprojector", COL["green"], COL["green_soft"], size=8.0)
    chip(ax, x + 358, y + 388, 104, 42, "norm cap\ncold-only", COL["green"], COL["green_soft"], size=8.0)
    txt(ax, x + 278, y + 449, r"$q_i=\mathrm{Norm}(base_i+\Delta_i)$", size=10.0, weight="bold", color=COL["dark_green"])
    arrow(ax, [(x + 432, y + 170), (x + 432, y + 342)], color=COL["green"], lw=2.0, ms=13)

    box(ax, x + 34, y + 492, 452, 84, COL["paper"], ec="#c1cad7", lw=1.35, r=10, z=3)
    txt(ax, x + 58, y + 514, "learner/user encoder", size=10.1, weight="bold", color=COL["muted"], ha="left")
    user_icon(ax, x + 58, y + 529, w=94, h=36)
    txt(ax, x + 119, y + 547, "history", size=7.8, weight="bold", color=COL["muted"])
    mlp(ax, x + 196, y + 522, w=56, h=48, label="user\nproj", color=COL["orange"])
    vector(ax, x + 300, y + 537, w=74, h=18, colors=["#ffffff", "#f0d2bd", "#ee9f62", "#b95f2e"])
    txt(ax, x + 337, y + 569, "$z_u$", size=10.1, weight="bold", color=COL["orange"])
    box(ax, x + 393, y + 526, 70, 40, COL["orange_soft"], ec=COL["orange"], lw=1.05, r=6)
    txt(ax, x + 428, y + 546, "user\nbank", size=8.5, weight="bold")
    arrow(ax, [(x + 152, y + 547), (x + 196, y + 547)], color=COL["line"], lw=1.8, ms=12)
    arrow(ax, [(x + 252, y + 547), (x + 300, y + 547)], color=COL["line"], lw=1.8, ms=12)
    arrow(ax, [(x + 374, y + 547), (x + 393, y + 547)], color=COL["orange"], lw=1.8, ms=12)


def draw_agent_panel(ax):
    x, y, w, h = 610, 168, 878, 608
    panel(ax, x, y, w, h, COL["middle"], "FAST3 Feedback Rollout", tag="B")

    box(ax, x + 38, y + 76, 802, 122, COL["paper"], ec=COL["green"], lw=1.45, r=10, z=3)
    txt(ax, x + 64, y + 100, "knowledge-biased candidate sampler", size=10.4, weight="bold", color=COL["dark_green"], ha="left")
    state_stack(ax, x + 74, y + 120, "$h_t$")
    box(ax, x + 210, y + 113, 144, 50, COL["soft"], ec=COL["green"], lw=1.15, r=7)
    txt(ax, x + 282, y + 132, "top-M retrieval", size=9.3, weight="bold")
    txt(ax, x + 282, y + 151, "$h_t^T Z_U$", size=8.7, weight="bold", color=COL["muted"])
    box(ax, x + 390, y + 104, 236, 72, COL["green_soft"], ec=COL["green"], lw=1.15, r=7)
    txt(ax, x + 508, y + 122, "probability mix", size=9.4, weight="bold", color=COL["dark_green"])
    txt(ax, x + 508, y + 144, "retrieval + course fit", size=8.4, weight="bold", color=COL["muted"])
    txt(ax, x + 508, y + 162, "optional SAGE-lite / CGRC", size=8.2, weight="bold", color=COL["violet"])
    box(ax, x + 674, y + 113, 126, 50, COL["soft"], ec=COL["green"], lw=1.15, r=7)
    txt(ax, x + 737, y + 133, "$S_t$", size=10.5, weight="bold", color=COL["green"])
    txt(ax, x + 737, y + 152, "$N$ candidates", size=8.5, weight="bold", color=COL["muted"])
    arrow(ax, [(x + 160, y + 149), (x + 210, y + 138)], color=COL["green"], lw=2.0, ms=13)
    arrow(ax, [(x + 354, y + 138), (x + 390, y + 140)], color=COL["green"], lw=2.0, ms=13)
    arrow(ax, [(x + 626, y + 140), (x + 674, y + 138)], color=COL["green"], lw=2.0, ms=13)

    box(ax, x + 88, y + 244, 706, 160, COL["paper"], ec=COL["blue"], lw=1.55, r=12, z=3)
    txt(ax, x + 442, y + 270, "T-step actor-critic simulation", size=12.3, weight="bold", color=COL["blue"])
    box(ax, x + 124, y + 300, 174, 66, COL["soft"], ec=COL["blue"], lw=1.15, r=7)
    txt(ax, x + 211, y + 320, "policy", size=10.3, weight="bold")
    txt(ax, x + 211, y + 342, r"$a_t\sim\pi_\theta(a|h_t,S_t)$", size=8.8, weight="bold", color=COL["muted"])
    box(ax, x + 360, y + 300, 132, 66, COL["orange_soft"], ec=COL["orange"], lw=1.15, r=7)
    txt(ax, x + 426, y + 322, "action", size=10.0, weight="bold", color=COL["orange"])
    txt(ax, x + 426, y + 344, "selected user\n$a_t$", size=8.4, weight="bold")
    box(ax, x + 556, y + 294, 202, 78, COL["soft"], ec=COL["blue"], lw=1.15, r=7)
    txt(ax, x + 657, y + 315, "state update", size=10.0, weight="bold", color=COL["blue"])
    txt(ax, x + 657, y + 338, r"$h_{t+1}=h_t+\eta\nabla A_t$", size=8.5, weight="bold", color=COL["muted"])
    txt(ax, x + 657, y + 358, "adaptive target anchor", size=8.0, weight="bold", color=COL["muted"])
    arrow(ax, [(x + 737, y + 163), (x + 737, y + 222), (x + 211, y + 222), (x + 211, y + 300)], color=COL["green"], lw=1.8, ms=12)
    arrow(ax, [(x + 298, y + 333), (x + 360, y + 333)], color=COL["blue"], lw=2.1, ms=14)
    arrow(ax, [(x + 492, y + 333), (x + 556, y + 333)], color=COL["blue"], lw=2.1, ms=14)
    arrow(ax, [(x + 657, y + 372), (x + 657, y + 430), (x + 118, y + 430), (x + 118, y + 180)], color=COL["blue"], lw=1.65, ms=11, ls=(0, (6, 4)), alpha=0.9)

    box(ax, x + 38, y + 444, 802, 116, "#fbfcfe", ec=COL["orange"], lw=1.35, r=10, z=3)
    txt(ax, x + 64, y + 468, "reward design", size=10.4, weight="bold", color=COL["orange"], ha="left")
    reward_labels = [
        ("target align", COL["blue"], COL["blue_soft"]),
        ("step gain", COL["green"], COL["green_soft"]),
        ("concept bonus", COL["green"], COL["green_soft"]),
        ("prereq gap", COL["orange"], COL["orange_soft"]),
        ("difficulty gap", COL["gold"], COL["gold_soft"]),
        ("redundancy / dup", COL["red"], COL["red_soft"]),
    ]
    for j, (lab, color, fill) in enumerate(reward_labels):
        chip(ax, x + 74 + j * 125, y + 495, 104, 38, lab, color, fill, size=7.6)
    txt(ax, x + 442, y + 548, r"$r_t=\sum_k \lambda_k r_t^k$", size=10.5, weight="bold", color=COL["red"])

    box(ax, x + 116, y + 578, 648, 58, COL["paper"], ec=COL["red"], lw=1.25, r=8, ls=(0, (7, 5)), z=4)
    txt(ax, x + 440, y + 598, "PPO feedback updates actor-critic", size=10.2, weight="bold", color=COL["red"])
    txt(ax, x + 440, y + 620, "GAE, advantage normalization, value clipping", size=8.4, weight="bold", color=COL["muted"])
    arrow(ax, [(x + 442, y + 578), (x + 442, y + 405)], color=COL["red"], lw=1.75, ms=12, ls=(0, (7, 5)))


def draw_ranking_panel(ax):
    x, y, w, h = 1520, 168, 422, 608
    panel(ax, x, y, w, h, COL["right"], "Strict Item-Cold Ranking", tag="C")

    box(ax, x + 34, y + 76, 354, 78, COL["gold_soft"], ec=COL["gold"], lw=1.25, r=8, z=4)
    snowflake(ax, x + 62, y + 108, r=6.0)
    txt(ax, x + 216, y + 101, "static item-cold inference", size=11.4, weight="bold")
    txt(ax, x + 216, y + 128, "full catalog, item-macro metrics", size=8.8, weight="bold", color=COL["muted"])

    box(ax, x + 34, y + 192, 354, 252, COL["paper"], ec=COL["line"], lw=1.45, r=10, z=3)
    txt(ax, x + 58, y + 216, "score all candidate courses", size=10.2, weight="bold", color=COL["muted"], ha="left")
    user_icon(ax, x + 60, y + 252, w=102, h=40)
    vector(ax, x + 74, y + 308, w=74, h=18, colors=["#ffffff", "#f0d2bd", "#ee9f62", "#b95f2e"])
    txt(ax, x + 111, y + 342, "$z_u$", size=10.3, weight="bold", color=COL["orange"])
    box(ax, x + 58, y + 374, 108, 44, COL["soft"], ec=COL["blue"], lw=1.1, r=6)
    vector(ax, x + 75, y + 388, w=74, h=16)
    txt(ax, x + 112, y + 432, "$Z_{cold}$ bank", size=8.5, weight="bold", color=COL["blue"])

    box(ax, x + 207, y + 292, 102, 78, COL["violet_soft"], ec=COL["violet"], lw=1.25, r=8, z=6)
    txt(ax, x + 258, y + 315, "dot product", size=9.5, weight="bold")
    txt(ax, x + 258, y + 340, r"$s(u,j)$", size=10.5, weight="bold", color=COL["violet"])
    txt(ax, x + 258, y + 360, r"$=z_u^T z_j/\tau$", size=8.2, weight="bold", color=COL["muted"])
    arrow(ax, [(x + 166, y + 272), (x + 190, y + 272), (x + 190, y + 317), (x + 207, y + 317)], color=COL["orange"], lw=1.75, ms=12)
    arrow(ax, [(x + 166, y + 397), (x + 190, y + 397), (x + 190, y + 349), (x + 207, y + 349)], color=COL["blue"], lw=1.75, ms=12)

    box(ax, x + 326, y + 262, 40, 138, COL["soft"], ec=COL["line"], lw=1.1, r=6, z=5)
    for k, yy in enumerate([275, 299, 323, 347, 371]):
        fill = COL["blue_soft"] if k < 2 else "#dfe5ec"
        box(ax, x + 333, y + yy, 26, 13, fill, ec=COL["line"], lw=0.8, r=2, z=8)
    txt(ax, x + 346, y + 418, "Top-K\nlist", size=8.5, weight="bold", color=COL["muted"])
    arrow(ax, [(x + 309, y + 331), (x + 326, y + 331)], color=COL["line"], lw=1.9, ms=12)

    box(ax, x + 34, y + 478, 354, 74, COL["paper"], ec=COL["line"], lw=1.35, r=9, z=3)
    txt(ax, x + 58, y + 499, "evaluation output", size=10.3, weight="bold", ha="left")
    chip(ax, x + 70, y + 520, 108, 24, "Recall@K", COL["blue"], COL["blue_soft"], size=8.6)
    chip(ax, x + 244, y + 520, 108, 24, "NDCG@K", COL["violet"], COL["violet_soft"], size=8.6)


def draw_training_bar(ax):
    x, y, w, h = 58, 804, 1884, 106
    panel(ax, x, y, w, h, COL["bottom"], "")
    txt(ax, x + 38, y + 28, "Training objectives and trace", size=12.3, weight="bold", ha="left")
    txt(ax, x + 38, y + 64, "Loss:", size=10.6, weight="bold", ha="left", color=COL["red"])
    chips = [
        ("rank CE", COL["red"], COL["red_soft"]),
        ("ID-content InfoNCE", COL["blue"], COL["blue_soft"]),
        ("PPO", COL["violet"], COL["violet_soft"]),
        ("prereq aux", COL["orange"], COL["orange_soft"]),
        ("PAAC", COL["gold"], COL["gold_soft"]),
        ("SAGE aux", COL["green"], COL["green_soft"]),
        ("CGRC recon aux", COL["violet"], COL["violet_soft"]),
        ("Delta reg", COL["green"], COL["green_soft"]),
    ]
    start = x + 102
    widths = [94, 154, 74, 100, 72, 96, 132, 96]
    xx = start
    for (label, color, fill), cw in zip(chips, widths):
        chip(ax, xx, y + 46, cw, 34, label, color, fill, size=8.1)
        xx += cw + 18
    txt(
        ax,
        x + 1540,
        y + 29,
        "static_protocol_manifest.json records split, env flags, course artifacts, and optional modules",
        size=8.8,
        weight="bold",
        color=COL["muted"],
    )
    txt(
        ax,
        x + 942,
        y + 92,
        "code path: usim_feedback_fast3_content_delta.py::Fast3FeedbackUSIM | fast3_delta/config.py | fast3_delta/eval.py",
        size=8.3,
        color=COL["muted"],
    )
    arrow(ax, [(x + 398, y + 46), (x + 398, 776)], color=COL["red"], lw=1.7, ms=12, ls=(0, (7, 5)), z=9)
    arrow(ax, [(x + 894, y + 46), (x + 894, 776)], color=COL["red"], lw=1.7, ms=12, ls=(0, (7, 5)), z=9)


def draw_cross_routes(ax):
    red_boundary_arrow(ax, 590, 470, scale=0.58, z=25)
    txt(ax, 637, 451, "$h_0=q_i$", size=9.0, weight="bold", color=COL["dark_green"])
    arrow(ax, [(578, 470), (610, 470)], color=COL["blue"], lw=2.35, ms=15)

    arrow(ax, [(1488, 468), (1520, 468)], color=COL["blue"], lw=2.35, ms=15)
    red_boundary_arrow(ax, 1494, 468, scale=0.58, z=25)
    txt(ax, 1470, 447, "$z_j$", size=9.0, weight="bold", color=COL["blue"])

    arrow(ax, [(1390, 715), (1390, 748), (1870, 748), (1870, 552)], color=COL["red"], lw=1.55, ms=11, ls=(0, (6, 5)))
    txt(ax, 1630, 734, "best checkpoint by full cold N@K", size=8.0, weight="bold", color=COL["red"])


def draw_legend(ax):
    box(ax, 645, 926, 710, 26, COL["paper"], ec=COL["panel"], lw=1.0, r=6, z=20)
    y = 939
    ax.plot([670, 710], [y, y], color=COL["line"], lw=2.1, zorder=21)
    txt(ax, 720, y, "forward", size=8.1, weight="bold", color=COL["muted"], ha="left", z=22)
    ax.plot([815, 855], [y, y], color=COL["green"], lw=2.1, zorder=21)
    txt(ax, 865, y, "knowledge guidance", size=8.1, weight="bold", color=COL["muted"], ha="left", z=22)
    ax.plot([1045, 1085], [y, y], color=COL["red"], lw=1.9, linestyle=(0, (6, 5)), zorder=21)
    txt(ax, 1095, y, "training-only update", size=8.1, weight="bold", color=COL["muted"], ha="left", z=22)
    ax.plot([1275, 1315], [y, y], color=COL["orange"], lw=2.1, zorder=21)
    txt(ax, 1325, y, "user/action signal", size=8.1, weight="bold", color=COL["muted"], ha="left", z=22)


def draw_figure():
    fig = plt.figure(figsize=(14.2, 6.8), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.add_patch(patches.Rectangle((0, 0), W, H, facecolor=COL["paper"], edgecolor="none", zorder=0))

    draw_state_band(ax)
    draw_encoder_panel(ax)
    draw_agent_panel(ax)
    draw_ranking_panel(ax)
    draw_training_bar(ax)
    draw_cross_routes(ax)
    draw_legend(ax)
    return fig


def save_outputs(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in [(".svg", {}), (".pdf", {}), (".png", {"dpi": 300})]:
        final = base.with_suffix(suffix)
        tmp = base.with_name(base.name + "_tmp").with_suffix(suffix)
        if tmp.exists():
            tmp.unlink()
        fig.savefig(tmp, bbox_inches="tight", pad_inches=0.035, facecolor=COL["paper"], **kwargs)
        if final.exists():
            final.unlink()
        tmp.replace(final)


def save_spec(base: Path):
    spec_dir = base.parent / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"{base.name}_spec.json"
    spec_path.write_text(json.dumps(FIGURE_SPEC, indent=2), encoding="utf-8")
    return spec_path


def main():
    base = Path(__file__).resolve().parent / "fast3_content_delta_model_topconf"
    fig = draw_figure()
    save_outputs(fig, base)
    plt.close(fig)
    spec_path = save_spec(base)
    for suffix in [".svg", ".pdf", ".png"]:
        print(f"saved: {base.with_suffix(suffix)}")
    print(f"saved: {spec_path}")


if __name__ == "__main__":
    main()
