import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch


# NeurIPS/ICLR-style simplified method overview for CKG-RL.
# The layout is deterministic and intentionally sparse: one main flow,
# compact RL training module, and explicit notation/visual legends.

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 10,
        "legend.frameon": False,
    }
)


W, H = 2480, 1180

COL = {
    "ink": "#172033",
    "muted": "#596579",
    "line": "#263244",
    "paper": "#ffffff",
    "panel": "#2b3544",
    "soft": "#f8fafc",
    "repr": "#2563eb",
    "repr_soft": "#eaf2ff",
    "repr_mid": "#b9d5ff",
    "learn": "#d97706",
    "learn_soft": "#fff3df",
    "learn_mid": "#ffd59a",
    "infer": "#7c3aed",
    "infer_soft": "#f2edff",
    "infer_mid": "#d8ccff",
    "green": "#327a52",
    "green_soft": "#e8f5ed",
    "gold": "#a16207",
    "gold_soft": "#fff7d6",
    "red": "#c2410c",
    "red_soft": "#ffebe3",
    "gray": "#e7ecf2",
    "gray2": "#d5dce6",
}


FIGURE_SPEC = {
    "canvas": {"width": W, "height": H},
    "purpose": (
        "Simplified top-conference framework figure: make the training-to-inference "
        "main story explicit, collapse overloaded reward details, unify notation, "
        "and separate RL training from cold-start retrieval."
    ),
    "visual_grammar": {
        "blue": "representation learning and embeddings",
        "orange": "RL training, simulator, reward, policy learning",
        "purple": "cold-start inference and retrieval",
        "solid_arrow": "forward pass / data flow",
        "dashed_arrow": "training signal",
        "red_arrow": "stage transition",
    },
    "panels": [
        {
            "id": "a",
            "title": "Representation Learning",
            "main_modules": [
                "content tower",
                "masked ID tower",
                "fusion gate",
                "history-only user encoder",
            ],
            "outputs": ["course representation q_i", "user embedding z_u"],
        },
        {
            "id": "b",
            "title": "RL-based Training Simulator",
            "main_modules": [
                "environment / simulator",
                "actor-critic agent",
                "compact multi-component reward",
            ],
            "note": "RL is used for training only; it is not called during retrieval.",
        },
        {
            "id": "c",
            "title": "Cold-start Retrieval",
            "main_modules": [
                "strict item-cold course bank",
                "single embedding-matching score",
                "Top-K cold courses",
            ],
            "score": "s(u,i)=z_u^T z_i^cold",
        },
    ],
    "notation": {
        "q_i": "course representation",
        "z_u": "user embedding",
        "s_t": "simulator state",
        "h_t": "learner hidden state",
        "S_t": "exploration candidate set",
    },
}


def txt(ax, x, y, s, size=14, weight="normal", color=None, ha="center", va="center", z=20):
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


def box(ax, x, y, w, h, fc, ec=None, lw=1.5, r=10, ls="-", z=2):
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


def panel(ax, x, y, w, h, fc, ec, tag, title, subtitle):
    box(ax, x, y, w, h, fc, ec, lw=2.4, r=12, z=1)
    txt(ax, x + 26, y + 34, tag, size=22, weight="bold", ha="left")
    txt(ax, x + 88, y + 32, title, size=19.5, weight="bold", ha="left")
    txt(ax, x + 88, y + 64, subtitle, size=12.2, color=COL["muted"], ha="left")


def arrow(ax, x1, y1, x2, y2, color=None, lw=2.2, ls="-", ms=16, rad=0.0, z=15):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=ls,
        color=color or COL["line"],
        shrinkA=0,
        shrinkB=0,
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def line(ax, x1, y1, x2, y2, color=None, lw=1.6, ls="-", z=12):
    ax.plot([x1, x2], [y1, y2], color=color or COL["line"], linewidth=lw, linestyle=ls, zorder=z)


def pill(ax, x, y, w, h, s, fc, ec, color=None, size=11):
    box(ax, x, y, w, h, fc, ec, lw=1.1, r=h / 2, z=5)
    txt(ax, x + w / 2, y + h / 2, s, size=size, weight="bold", color=color or ec)


def mini_vector(ax, x, y, n=5, colors=None, w=72, h=22):
    colors = colors or [COL["paper"], COL["gray"], COL["repr_mid"], COL["repr"], "#174ea6"]
    sw = w / n
    for i in range(n):
        ax.add_patch(
            patches.Rectangle(
                (x + i * sw, y),
                sw,
                h,
                linewidth=1.0,
                edgecolor=COL["line"],
                facecolor=colors[i % len(colors)],
                zorder=8,
            )
        )


def user_icon(ax, x, y, scale=1.0, color=None):
    c = color or COL["line"]
    ax.add_patch(patches.Circle((x, y), 11 * scale, linewidth=2.2, edgecolor=c, facecolor=COL["paper"], zorder=8))
    ax.add_patch(
        patches.Arc((x, y + 28 * scale), 48 * scale, 40 * scale, theta1=205, theta2=-25, linewidth=2.2, color=c, zorder=8)
    )


def book_icon(ax, x, y, scale=1.0, color=None):
    c = color or COL["line"]
    ax.add_patch(
        patches.FancyBboxPatch(
            (x - 28 * scale, y - 22 * scale),
            25 * scale,
            44 * scale,
            boxstyle=f"round,pad=0.02,rounding_size={4 * scale}",
            linewidth=2.1,
            edgecolor=c,
            facecolor=COL["paper"],
            zorder=8,
        )
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (x + 3 * scale, y - 22 * scale),
            25 * scale,
            44 * scale,
            boxstyle=f"round,pad=0.02,rounding_size={4 * scale}",
            linewidth=2.1,
            edgecolor=c,
            facecolor=COL["paper"],
            zorder=8,
        )
    )
    line(ax, x, y - 20 * scale, x, y + 25 * scale, c, lw=1.4, z=9)
    line(ax, x - 20 * scale, y - 7 * scale, x - 8 * scale, y - 7 * scale, c, lw=1.5, z=9)
    line(ax, x + 9 * scale, y - 7 * scale, x + 21 * scale, y - 7 * scale, c, lw=1.5, z=9)


def snowflake(ax, x, y, scale=1.0):
    c = "#1f6f9f"
    for ang in [0, 60, 120]:
        dx = 14 * scale
        dy = 0
        import math

        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        x1, y1 = x - dx * ca + dy * sa, y - dx * sa - dy * ca
        x2, y2 = x + dx * ca - dy * sa, y + dx * sa + dy * ca
        line(ax, x1, y1, x2, y2, c, lw=1.7, z=10)
    ax.add_patch(patches.Circle((x, y), 2.2 * scale, facecolor=c, edgecolor=c, zorder=10))


def draw_representation_panel(ax):
    x, y, w, h = 60, 130, 650, 740
    panel(
        ax,
        x,
        y,
        w,
        h,
        COL["repr_soft"],
        COL["repr"],
        "a",
        "Representation Learning",
        "disentangled cold-course encoder",
    )

    book_icon(ax, x + 326, y + 130, 1.05, COL["line"])
    snowflake(ax, x + 365, y + 101, 0.9)
    txt(ax, x + 326, y + 178, "cold course $i$", size=14, weight="bold")
    arrow(ax, x + 326, y + 196, x + 326, y + 240, COL["repr"], lw=2.2)

    line(ax, x + 180, y + 240, x + 470, y + 240, COL["line"], lw=2.0)
    arrow(ax, x + 180, y + 240, x + 180, y + 266, COL["line"], lw=1.8)
    arrow(ax, x + 470, y + 240, x + 470, y + 266, COL["line"], lw=1.8)

    box(ax, x + 82, y + 270, 250, 140, COL["paper"], COL["repr"], lw=2.0, r=8)
    txt(ax, x + 207, y + 300, "content tower", size=14, weight="bold", color=COL["repr"])
    txt(ax, x + 207, y + 331, "$x_i \\rightarrow c_i$", size=15, color=COL["ink"])
    mini_vector(ax, x + 154, y + 360, colors=[COL["paper"], COL["gray"], "#d6e5ff", COL["repr_mid"], COL["repr"]])

    box(ax, x + 350, y + 270, 250, 140, COL["paper"], "#536173", lw=2.0, r=8)
    txt(ax, x + 475, y + 300, "ID-factor tower", size=14, weight="bold", color="#536173")
    txt(ax, x + 475, y + 331, "$v_i \\odot m_i$", size=15, color=COL["ink"])
    mini_vector(ax, x + 422, y + 360, colors=[COL["paper"], COL["gray"], COL["gray2"], COL["gray2"], "#8391a3"])
    line(ax, x + 422, y + 383, x + 494, y + 360, COL["red"], lw=3.0, z=12)
    txt(ax, x + 515, y + 383, "forced\ncold mask", size=10.5, color=COL["red"], ha="left")

    arrow(ax, x + 207, y + 410, x + 276, y + 474, COL["repr"], lw=2.0)
    arrow(ax, x + 475, y + 410, x + 432, y + 474, COL["repr"], lw=2.0)

    box(ax, x + 155, y + 474, 380, 100, COL["soft"], "#4f46e5", lw=1.8, r=8)
    txt(ax, x + 345, y + 504, "fusion gate", size=15, weight="bold", color="#4f46e5")
    txt(ax, x + 345, y + 536, "$q_i=Gate(c_i, v_i \\odot m_i)$", size=16)
    arrow(ax, x + 345, y + 574, x + 345, y + 606, COL["repr"], lw=2.2)
    mini_vector(ax, x + 300, y + 614, colors=[COL["paper"], "#e0ecff", COL["repr_mid"], COL["repr"], "#174ea6"], w=90)
    txt(ax, x + 345, y + 656, "course representation $q_i$", size=13, color=COL["repr"], weight="bold")

    line(ax, x + 60, y + 676, x + w - 60, y + 676, COL["repr_mid"], lw=1.2)
    user_icon(ax, x + 135, y + 710, 0.66, COL["line"])
    txt(ax, x + 236, y + 710, "history-only\nuser encoder", size=11.4, weight="bold", color=COL["muted"])
    arrow(ax, x + 304, y + 710, x + 370, y + 710, COL["line"], lw=1.8)
    box(ax, x + 370, y + 686, 84, 48, COL["learn_soft"], COL["learn"], lw=1.6, r=6)
    txt(ax, x + 412, y + 710, "MLP", size=12.6, weight="bold")
    arrow(ax, x + 454, y + 710, x + 506, y + 710, COL["line"], lw=1.8)
    mini_vector(ax, x + 506, y + 699, colors=[COL["paper"], "#f8d5b2", "#d99662", "#ad5b2a", "#7c3f1b"], w=78)
    txt(ax, x + 588, y + 710, "$z_u$", size=14, color=COL["learn"], weight="bold")


def draw_training_panel(ax):
    x, y, w, h = 760, 130, 900, 740
    panel(
        ax,
        x,
        y,
        w,
        h,
        COL["learn_soft"],
        COL["learn"],
        "b",
        "RL-based Training Simulator",
        "environment + agent + compact reward",
    )
    pill(ax, x + w - 238, y + 74, 172, 28, "training only", COL["paper"], COL["learn"], COL["learn"], size=10.2)

    box(ax, x + 55, y + 126, 390, 410, COL["paper"], COL["learn"], lw=2.0, r=10)
    txt(ax, x + 250, y + 156, "Environment / simulator", size=16, weight="bold", color=COL["learn"])

    box(ax, x + 90, y + 195, 320, 76, COL["soft"], COL["line"], lw=1.2, r=6)
    txt(ax, x + 250, y + 218, "state", size=11.5, weight="bold", color=COL["muted"])
    txt(ax, x + 250, y + 247, "$s_t=[h_t, q_i, l_t]$", size=17, color=COL["ink"])

    box(ax, x + 90, y + 304, 320, 88, COL["green_soft"], COL["green"], lw=1.6, r=6)
    txt(ax, x + 250, y + 329, "exploration set $S_t$", size=14, weight="bold", color=COL["green"])
    txt(ax, x + 250, y + 360, "Top-M retrieval  |  course prior  |  stop", size=11.6, color=COL["muted"])
    arrow(ax, x + 250, y + 271, x + 250, y + 304, COL["learn"], lw=1.8)

    box(ax, x + 90, y + 426, 320, 76, COL["soft"], COL["line"], lw=1.2, r=6)
    txt(ax, x + 250, y + 448, "transition", size=11.5, weight="bold", color=COL["muted"])
    txt(ax, x + 250, y + 477, "$s_{t+1}=\\rho(s_t,a_t)$", size=17)
    arrow(ax, x + 250, y + 392, x + 250, y + 426, COL["learn"], lw=1.8)

    arrow(ax, x + 415, y + 233, x + 505, y + 233, COL["line"], lw=2.0)
    arrow(ax, x + 415, y + 350, x + 505, y + 350, COL["line"], lw=2.0)

    box(ax, x + 505, y + 164, 340, 190, COL["paper"], COL["learn"], lw=2.0, r=10)
    txt(ax, x + 675, y + 197, "RL agent", size=16, weight="bold", color=COL["learn"])
    txt(ax, x + 675, y + 238, "Actor: $\\pi_\\theta(a\\mid s_t,S_t)$", size=15)
    txt(ax, x + 675, y + 277, "Critic: $V_\\phi(s_t)$", size=15)
    pill(ax, x + 615, y + 309, 120, 28, "action $a_t$", COL["red_soft"], COL["red"], COL["red"], size=10.5)

    arrow(ax, x + 615, y + 337, x + 410, y + 464, COL["red"], lw=2.0, rad=0.16)

    box(ax, x + 505, y + 410, 340, 126, COL["gold_soft"], COL["gold"], lw=1.8, r=10)
    txt(ax, x + 675, y + 437, "multi-component reward", size=15, weight="bold", color=COL["gold"])
    txt(ax, x + 675, y + 468, "alignment, progress, concept", size=12.2)
    txt(ax, x + 675, y + 495, "prereq, difficulty, repeat", size=12.2)
    txt(ax, x + 675, y + 520, "$r_t=\\sum_k \\lambda_k r_t^k$", size=15, color=COL["red"])
    arrow(ax, x + 675, y + 410, x + 675, y + 354, COL["gold"], lw=2.0, ls=(0, (4, 4)), ms=14)

    box(ax, x + 84, y + 574, 752, 74, "#fffaf2", "#dfb66d", lw=1.1, r=8)
    txt(
        ax,
        x + 460,
        y + 594,
        "Policy learning shapes representations",
        size=12.6,
        weight="bold",
        color=COL["learn"],
    )
    txt(ax, x + 460, y + 622, "under full-catalog competition; rollout is not used at inference.", size=10.8, color=COL["muted"])


def draw_retrieval_panel(ax):
    x, y, w, h = 1710, 130, 710, 740
    panel(
        ax,
        x,
        y,
        w,
        h,
        COL["infer_soft"],
        COL["infer"],
        "c",
        "Cold-start Retrieval",
        "embedding matching at inference",
    )
    pill(ax, x + w - 230, y + 74, 172, 28, "inference only", COL["paper"], COL["infer"], COL["infer"], size=10.2)

    box(ax, x + 65, y + 132, 580, 70, COL["paper"], COL["infer"], lw=1.8, r=8)
    txt(ax, x + 355, y + 157, "strict item-cold protocol", size=15.5, weight="bold", color=COL["infer"])
    txt(ax, x + 355, y + 184, "candidate bank contains cold courses only", size=11.8, color=COL["muted"])

    box(ax, x + 70, y + 260, 210, 112, COL["paper"], COL["learn"], lw=1.6, r=8)
    user_icon(ax, x + 112, y + 309, 0.78, COL["line"])
    mini_vector(ax, x + 150, y + 296, colors=[COL["paper"], "#f8d5b2", "#d99662", "#ad5b2a", "#7c3f1b"], w=92)
    txt(ax, x + 175, y + 349, "$z_u$", size=16, color=COL["learn"], weight="bold")

    box(ax, x + 70, y + 454, 210, 120, COL["paper"], COL["repr"], lw=1.6, r=8)
    book_icon(ax, x + 112, y + 504, 0.7, COL["line"])
    snowflake(ax, x + 148, y + 480, 0.75)
    mini_vector(ax, x + 150, y + 493, colors=[COL["paper"], "#e0ecff", COL["repr_mid"], COL["repr"], "#174ea6"], w=92)
    txt(ax, x + 175, y + 548, "$\\{z_i^{cold}\\}$", size=16, color=COL["repr"], weight="bold")

    box(ax, x + 328, y + 330, 210, 150, COL["soft"], COL["line"], lw=1.8, r=8)
    txt(ax, x + 433, y + 365, "similarity", size=14.2, weight="bold")
    txt(ax, x + 433, y + 414, "$s(u,i)=z_u^{\\top}z_i^{cold}$", size=13.3, color=COL["ink"])
    txt(ax, x + 433, y + 452, "single score", size=10.8, color=COL["muted"])

    arrow(ax, x + 280, y + 316, x + 328, y + 382, COL["learn"], lw=2.2)
    arrow(ax, x + 280, y + 516, x + 328, y + 427, COL["repr"], lw=2.2)

    box(ax, x + 568, y + 300, 95, 230, COL["paper"], COL["infer"], lw=1.8, r=6)
    txt(ax, x + 615, y + 328, "Top-K", size=15, weight="bold", color=COL["infer"])
    for i, label in enumerate(["C-17", "C-04", "C-31", "C-K"]):
        yy = y + 360 + i * 40
        box(ax, x + 584, yy, 63, 28, "#f8fbff", "#a9b8d0", lw=1.0, r=3)
        prefix = f"{i + 1}" if i < 3 else "K"
        txt(ax, x + 596, yy + 14, prefix, size=11, weight="bold", color=COL["muted"])
        txt(ax, x + 627, yy + 14, label, size=11.5, weight="bold", color=COL["ink"])

    arrow(ax, x + 538, y + 405, x + 568, y + 405, COL["infer"], lw=2.4)
    txt(ax, x + 355, y + 620, "No simulator state or RL rollout is used here.", size=11.8, color=COL["muted"])


def draw_main_flow(ax):
    y = 500
    arrow(ax, 710, y, 760, y, COL["red"], lw=4.0, ms=24)
    txt(ax, 735, y - 36, "$q_i, z_u$", size=12.5, color=COL["red"], weight="bold")
    arrow(ax, 1660, y, 1710, y, COL["red"], lw=4.0, ms=24)
    txt(ax, 1685, y - 42, "trained\nencoder", size=11.5, color=COL["red"], weight="bold")

    arrow(ax, 620, 112, 2215, 112, COL["red"], lw=2.6, ms=18)
    txt(
        ax,
        1420,
        92,
        "Main story: Encoder -> simulator training -> policy learning -> strict cold retrieval",
        size=12.8,
        weight="bold",
        color=COL["red"],
    )


def draw_bottom_strip(ax):
    y = 915
    box(ax, 60, y, 1140, 190, COL["soft"], "#a7b3c3", lw=1.2, r=10)
    txt(ax, 92, y + 28, "Key contributions", size=15, weight="bold", ha="left", color=COL["ink"])

    contribs = [
        ("Content-anchored\ncold representations", COL["repr_soft"], COL["repr"]),
        ("RL curriculum\nsimulation", COL["learn_soft"], COL["learn"]),
        ("Strict cold-only\nretrieval", COL["infer_soft"], COL["infer"]),
    ]
    for i, (label, fc, ec) in enumerate(contribs):
        xx = 95 + i * 350
        box(ax, xx, y + 60, 295, 92, fc, ec, lw=1.5, r=8)
        txt(ax, xx + 147.5, y + 106, label, size=13.5, weight="bold", color=ec)

    box(ax, 1240, y, 1180, 190, COL["soft"], "#a7b3c3", lw=1.2, r=10)
    txt(ax, 1272, y + 28, "Notation and visual grammar", size=15, weight="bold", ha="left")

    notation = [
        ("$q_i$: course representation", 1280, y + 68),
        ("$z_u$: user embedding", 1608, y + 68),
        ("$s_t$: simulator state", 1280, y + 110),
        ("$h_t$: learner hidden state", 1608, y + 110),
        ("$S_t$: exploration set", 1280, y + 152),
    ]
    for label, xx, yy in notation:
        txt(ax, xx, yy, label, size=10.8, color=COL["ink"], ha="left")

    # Arrow legend.
    lx, ly = 2035, y + 64
    line(ax, lx, ly, lx + 62, ly, COL["line"], lw=2.0)
    arrow(ax, lx + 62, ly, lx + 96, ly, COL["line"], lw=2.0, ms=12)
    txt(ax, lx + 110, ly, "forward", size=11.5, ha="left", color=COL["muted"])
    line(ax, lx, ly + 40, lx + 92, ly + 40, COL["gold"], lw=2.0, ls=(0, (4, 4)))
    arrow(ax, lx + 62, ly + 40, lx + 96, ly + 40, COL["gold"], lw=2.0, ls=(0, (4, 4)), ms=12)
    txt(ax, lx + 110, ly + 40, "training signal", size=11.5, ha="left", color=COL["muted"])
    line(ax, lx, ly + 80, lx + 92, ly + 80, COL["red"], lw=2.6)
    arrow(ax, lx + 62, ly + 80, lx + 96, ly + 80, COL["red"], lw=2.6, ms=12)
    txt(ax, lx + 110, ly + 80, "stage transition", size=11.5, ha="left", color=COL["muted"])


def draw():
    fig = plt.figure(figsize=(W / 200, H / 200), dpi=200, facecolor=COL["paper"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    txt(ax, 60, 38, "CKG-RL framework", size=27, weight="bold", ha="left")
    txt(
        ax,
        60,
        72,
        "Simplified training-to-inference view for strict item-cold recommendation",
        size=14,
        color=COL["muted"],
        ha="left",
    )

    draw_main_flow(ax)
    draw_representation_panel(ax)
    draw_training_panel(ax)
    draw_retrieval_panel(ax)
    draw_bottom_strip(ax)

    out_dir = Path(__file__).resolve().parent
    spec_dir = out_dir / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    stem = "ckg_rl_framework_neurips_simplified"
    (spec_dir / f"{stem}_spec.json").write_text(json.dumps(FIGURE_SPEC, indent=2), encoding="utf-8")

    for ext in ["svg", "pdf", "png"]:
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)

    plt.close(fig)


if __name__ == "__main__":
    draw()
