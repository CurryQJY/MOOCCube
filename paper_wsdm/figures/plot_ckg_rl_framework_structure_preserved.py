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
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "stixsans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "legend.frameon": False,
    }
)


W, H = 2600, 1000

COL = {
    "ink": "#18212f",
    "muted": "#5d6876",
    "line": "#263241",
    "panel_line": "#2f3945",
    "paper": "#ffffff",
    "panel_a": "#e7f0dd",
    "panel_b": "#dcebf6",
    "panel_c": "#e9e7f2",
    "card": "#fbfaf5",
    "blue": "#1e5f8d",
    "blue_soft": "#e5f1f8",
    "green": "#3f7551",
    "green_soft": "#e3f1e3",
    "orange": "#b3652e",
    "orange_soft": "#f6dfca",
    "red": "#c92835",
    "red_soft": "#fae4e5",
    "violet": "#67527c",
    "violet_soft": "#eee8f5",
    "gold": "#8b6b1d",
    "gold_soft": "#f2e8c6",
    "dash": "#718193",
}

PANELS = {
    "a": {"x": 40, "y": 55, "w": 620, "h": 890, "fill": COL["panel_a"]},
    "b": {"x": 695, "y": 55, "w": 1130, "h": 890, "fill": COL["panel_b"]},
    "c": {"x": 1860, "y": 55, "w": 700, "h": 890, "fill": COL["panel_c"]},
}

LAYOUT_SPEC = {
    "canvas": {"width": W, "height": H},
    "purpose": "Structure-preserving visual redraw of the cold-course evidence encoder, course-knowledge guided simulation, and strict item-cold ranking method figure.",
    "panels": [
        {
            "title": "(a) Cold-course evidence encoder",
            "required_modules": [
                "Dropout-style course encoder",
                "cold course i",
                "side-feature tower",
                "ID-factor tower",
                "dropout / mask",
                "fusion MLP + gate",
                "q_i",
                "user-history encoder",
                "history only",
                "MLP",
                "z_u",
            ],
            "required_relations": [
                "x_i -> c_i",
                "v_i odot m_i",
                "q_i = Gate(c_i, v_i odot m_i)",
                "user history encoder -> z_u",
            ],
        },
        {
            "title": "(b) Course-knowledge guided simulation",
            "required_modules": [
                "T-step learner-course simulator",
                "s_t = [h_t, q_i, l_t]",
                "Exploration set construction",
                "Top-M",
                "course prior",
                "a_end",
                "actor-critic agent",
                "pi_theta(a | s_t, S_t)",
                "State transition",
                "h_{t+1} = rho(h_t, a_t, q_i)",
                "l_{t+1} = l_t - 1",
                "s_{t+1} = [h_{t+1}, q_i, l_{t+1}]",
                "Reward computation with six components",
                "r_t = sum_k lambda_k r_t^k",
            ],
        },
        {
            "title": "(c) Strict Item-Cold Ranking",
            "required_modules": [
                "strict item-cold inference",
                "cold-only full catalog",
                "dot-product scoring",
                "user embedding z_u",
                "cold bank / cold-course bank",
                "Z_cold",
                "Top-K courses",
            ],
            "required_relation": "s(u, i) = z_u^T z_i^cold",
        },
    ],
    "visual_grammar": {
        "blue": "state, user embedding, transition",
        "green": "course, cold course, cold bank",
        "orange": "course knowledge, reward, prior",
        "red": "dropout, mask, main transition arrows",
    },
}


def text(ax, x, y, s, size=8, weight="normal", color=None, ha="center", va="center", z=30):
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
        clip_on=False,
    )


def box(ax, x, y, w, h, fc=None, ec=None, lw=1.5, r=9, ls="-", z=5):
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


def panel(ax, key):
    p = PANELS[key]
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


def arrow(ax, start, end, color=None, lw=2.0, ms=16, ls="-", z=20, alpha=1.0, rad=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=ls,
        color=color or COL["line"],
        shrinkA=0,
        shrinkB=0,
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
        alpha=alpha,
    )
    ax.add_patch(arr)
    return arr


def line(ax, pts, color=None, lw=1.8, ls="-", z=12, alpha=1.0):
    xs, ys = zip(*pts)
    ax.plot(
        xs,
        ys,
        color=color or COL["line"],
        linewidth=lw,
        linestyle=ls,
        solid_capstyle="round",
        dash_capstyle="round",
        zorder=z,
        alpha=alpha,
    )


def orth_arrow(ax, pts, color=None, lw=2.0, ms=15, ls="-", z=20, alpha=1.0):
    if len(pts) < 2:
        raise ValueError("orth_arrow requires at least two points")
    if len(pts) > 2:
        line(ax, pts[:-1], color=color, lw=lw, ls=ls, z=z, alpha=alpha)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ms=ms, ls=ls, z=z, alpha=alpha)


def vector_bar(ax, x, y, n=4, w=22, h=19, colors=None, stroke=None, z=25):
    colors = colors or [COL["paper"], COL["blue_soft"], "#90a9bd", COL["blue"]]
    for i in range(n):
        ax.add_patch(
            patches.Rectangle(
                (x + i * w, y),
                w,
                h,
                facecolor=colors[i % len(colors)],
                edgecolor=stroke or COL["line"],
                linewidth=1.4,
                zorder=z,
            )
        )


def pill(ax, x, y, w, h, label, fc, ec, color=None, size=7.5):
    box(ax, x, y, w, h, fc=fc, ec=ec, lw=1.6, r=5, z=12)
    text(ax, x + w / 2, y + h / 2, label, size=size, weight="bold", color=color or ec, z=40)


def red_hollow_arrow(ax, x, y, scale=1.0, z=45):
    pts = [
        (x - 28 * scale, y - 16 * scale),
        (x + 0 * scale, y - 16 * scale),
        (x + 0 * scale, y - 31 * scale),
        (x + 42 * scale, y),
        (x + 0 * scale, y + 31 * scale),
        (x + 0 * scale, y + 16 * scale),
        (x - 28 * scale, y + 16 * scale),
    ]
    ax.add_patch(
        patches.Polygon(
            pts,
            closed=True,
            facecolor=COL["paper"],
            edgecolor=COL["red"],
            linewidth=4.0,
            zorder=z,
        )
    )


def draw_panel_titles(ax):
    p = PANELS["a"]
    text(ax, p["x"] + 28, p["y"] + 28, "(a) Cold-course evidence", size=10.8, weight="bold", ha="left")
    text(ax, p["x"] + 72, p["y"] + 56, "encoder", size=10.8, weight="bold", ha="left")

    p = PANELS["b"]
    text(ax, p["x"] + 28, p["y"] + 36, "(b) Course-knowledge guided simulation", size=11.7, weight="bold", ha="left")

    p = PANELS["c"]
    text(ax, p["x"] + 28, p["y"] + 36, "(c) Strict Item-Cold Ranking", size=11.7, weight="bold", ha="left")


def draw_course_icon(ax, cx, cy, scale=1.0):
    w, h = 36 * scale, 46 * scale
    ax.add_patch(
        patches.FancyBboxPatch(
            (cx - w - 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=4",
            linewidth=2.0,
            edgecolor=COL["line"],
            facecolor=COL["paper"],
            zorder=10,
        )
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (cx + 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=4",
            linewidth=2.0,
            edgecolor=COL["line"],
            facecolor=COL["paper"],
            zorder=10,
        )
    )
    line(ax, [(cx, cy - h / 2), (cx, cy + h / 2)], color=COL["line"], lw=1.5, z=11)
    for yy in [-12, 0, 12]:
        line(ax, [(cx - w + 7, cy + yy), (cx - 9, cy + yy + 1)], color=COL["muted"], lw=1.1, z=11)
        line(ax, [(cx + 9, cy + yy + 1), (cx + w - 7, cy + yy)], color=COL["muted"], lw=1.1, z=11)


def draw_state_stack(ax, x, y, w, h, rows, stroke=COL["blue"], label=None, label_color=None):
    box(ax, x, y, w, h, fc=COL["paper"], ec=stroke, lw=2.4, r=7, z=16)
    row_h = h / len(rows)
    for idx, (formula, fc, color) in enumerate(rows):
        yy = y + idx * row_h
        if idx > 0:
            line(ax, [(x, yy), (x + w, yy)], color=stroke, lw=1.2, z=18)
        ax.add_patch(
            patches.Rectangle(
                (x + 2, yy + 2),
                w - 4,
                row_h - 4,
                facecolor=fc,
                edgecolor="none",
                zorder=17,
            )
        )
        text(ax, x + w / 2, yy + row_h / 2, formula, size=8.5, color=color, z=35)
    if label:
        text(ax, x + w / 2, y + h + 28, label, size=8.5, weight="bold", color=label_color or stroke)


def draw_panel_a(ax):
    x, y = PANELS["a"]["x"], PANELS["a"]["y"]

    text(ax, x + 310, y + 100, "Dropout-style course encoder", size=9.8, weight="bold", color=COL["blue"])

    draw_course_icon(ax, x + 300, y + 165, scale=0.85)
    text(ax, x + 300, y + 228, "cold course i", size=9.5, weight="bold", color=COL["ink"])
    text(ax, x + 300, y + 253, "cold", size=8.5, weight="bold", color=COL["green"])

    split_y = y + 285
    line(ax, [(x + 300, y + 255), (x + 300, split_y)], color=COL["line"], lw=2.0, z=15)
    orth_arrow(ax, [(x + 300, split_y), (x + 190, split_y), (x + 190, y + 325)], color=COL["line"], lw=2.0)
    orth_arrow(ax, [(x + 300, split_y), (x + 420, split_y), (x + 420, y + 325)], color=COL["line"], lw=2.0)

    side_x, side_y, side_w, side_h = x + 85, y + 315, 205, 122
    id_x, id_y, id_w, id_h = x + 335, y + 315, 205, 122
    box(ax, side_x, side_y, side_w, side_h, fc=COL["paper"], ec=COL["blue"], lw=2.3, r=7)
    text(ax, side_x + side_w / 2, side_y + 28, "side-feature", size=9.3, weight="bold", color=COL["blue"])
    text(ax, side_x + side_w / 2, side_y + 54, "tower", size=9.3, weight="bold", color=COL["blue"])
    vector_bar(ax, side_x + 43, side_y + 76, n=4, w=27, h=25)
    text(ax, side_x + side_w / 2, side_y + 110, r"$x_i \rightarrow c_i$", size=8.5, color=COL["blue"])

    box(ax, id_x, id_y, id_w, id_h, fc=COL["paper"], ec=COL["line"], lw=2.0, r=7)
    text(ax, id_x + id_w / 2, id_y + 28, "ID-factor", size=9.3, weight="bold", color=COL["muted"])
    text(ax, id_x + id_w / 2, id_y + 54, "tower", size=9.3, weight="bold", color=COL["muted"])
    vector_bar(
        ax,
        id_x + 42,
        id_y + 76,
        n=4,
        w=27,
        h=25,
        colors=[COL["paper"], "#e9edf2", "#a8b4c0", "#596b7a"],
    )
    line(ax, [(id_x + 72, id_y + 88), (id_x + 145, id_y + 88)], color=COL["red"], lw=4.0, z=38)
    line(ax, [(id_x + 72, id_y + 101), (id_x + 145, id_y + 76)], color=COL["red"], lw=4.0, z=38)
    text(ax, id_x + id_w / 2, id_y + 110, r"$v_i \odot m_i$", size=8.5, color=COL["muted"])
    pill(ax, id_x + 78, id_y - 42, 106, 34, "dropout", COL["red"], COL["red"], color=COL["paper"], size=8.0)
    text(ax, id_x + 132, id_y - 58, "dropout / mask", size=7.6, weight="bold", color=COL["red"])

    fusion_x, fusion_y, fusion_w, fusion_h = x + 115, y + 505, 390, 92
    orth_arrow(ax, [(side_x + side_w / 2, side_y + side_h), (side_x + side_w / 2, fusion_y)], color=COL["line"], lw=2.0)
    orth_arrow(ax, [(id_x + id_w / 2, id_y + id_h), (id_x + id_w / 2, fusion_y)], color=COL["line"], lw=2.0)
    box(ax, fusion_x, fusion_y, fusion_w, fusion_h, fc=COL["paper"], ec=COL["violet"], lw=2.3, r=7)
    text(ax, fusion_x + fusion_w / 2, fusion_y + 28, "fusion MLP + gate", size=10.0, weight="bold", color=COL["violet"])
    text(
        ax,
        fusion_x + fusion_w / 2,
        fusion_y + 62,
        r"$q_i = \mathrm{Gate}(c_i,\; v_i \odot m_i)$",
        size=9.0,
        color=COL["violet"],
    )

    out_y = y + 654
    orth_arrow(ax, [(fusion_x + fusion_w / 2, fusion_y + fusion_h), (fusion_x + fusion_w / 2, out_y - 24)], color=COL["line"], lw=2.0)
    vector_bar(ax, x + 246, out_y - 28, n=4, w=28, h=26, colors=[COL["paper"], COL["green_soft"], "#9bbda5", COL["green"]])
    text(ax, x + 302, out_y + 30, r"$q_i$", size=10, weight="bold", color=COL["green"])

    line(ax, [(x + 70, y + 690), (x + 530, y + 690)], color="#cfdbc7", lw=1.4, z=10)
    text(ax, x + 300, y + 720, "user-history encoder", size=10.5, weight="bold", color=COL["orange"])

    hist_y = y + 768
    ax.add_patch(patches.Circle((x + 105, hist_y + 2), 14, facecolor=COL["paper"], edgecolor=COL["line"], linewidth=2.0, zorder=12))
    line(ax, [(x + 105, hist_y + 17), (x + 105, hist_y + 49)], color=COL["line"], lw=2.0, z=12)
    line(ax, [(x + 84, hist_y + 62), (x + 105, hist_y + 42), (x + 126, hist_y + 62)], color=COL["line"], lw=2.0, z=12)
    text(ax, x + 174, hist_y + 4, "history", size=8.5, weight="bold", color=COL["muted"])
    text(ax, x + 174, hist_y + 32, "only", size=8.5, weight="bold", color=COL["muted"])
    arrow(ax, (x + 210, hist_y + 18), (x + 270, hist_y + 18), color=COL["line"], lw=2.0, ms=15)
    box(ax, x + 270, hist_y - 8, 85, 52, fc=COL["paper"], ec=COL["orange"], lw=2.1, r=4)
    text(ax, x + 312, hist_y + 18, "MLP", size=9.5, weight="bold", color=COL["orange"])
    arrow(ax, (x + 355, hist_y + 18), (x + 420, hist_y + 18), color=COL["line"], lw=2.0, ms=15)
    vector_bar(
        ax,
        x + 420,
        hist_y + 4,
        n=4,
        w=27,
        h=28,
        colors=[COL["paper"], "#f2e4d8", "#ca7b4b", "#994f2b"],
    )
    text(ax, x + 442, hist_y + 72, r"user embedding $z_u$", size=7.4, weight="bold", color=COL["orange"])


def draw_reward_component(ax, x, y, w, h, title, formula, icon_color):
    box(ax, x, y, w, h, fc=COL["paper"], ec=COL["line"], lw=1.5, r=5, z=12)
    ax.add_patch(
        patches.Circle((x + 24, y + 29), 9, facecolor="#f7fbfd", edgecolor=icon_color, linewidth=1.8, zorder=18)
    )
    text(ax, x + 48, y + 22, title, size=8.0, weight="bold", color=COL["ink"], ha="left")
    text(ax, x + w / 2 + 10, y + 47, formula, size=7.5, color=COL["muted"])


def draw_panel_b(ax):
    x, y = PANELS["b"]["x"], PANELS["b"]["y"]

    sim_x, sim_y, sim_w, sim_h = x + 42, y + 78, 966, 805
    box(ax, sim_x, sim_y, sim_w, sim_h, fc="#d7e8f4", ec=COL["blue"], lw=3.0, r=7, z=3)
    text(ax, sim_x + sim_w / 2, sim_y + 54, "T-step learner-course simulator", size=14, weight="bold", color=COL["ink"])

    text(ax, sim_x + 72, sim_y + 154, r"$t = 0,\ldots,T-1$", size=8.5, weight="bold", color=COL["gold"])
    box(ax, sim_x + 40, sim_y + 126, 130, 35, fc=COL["gold_soft"], ec=COL["gold"], lw=1.6, r=6, z=13)
    arrow(ax, (sim_x + 106, sim_y + 161), (sim_x + 106, sim_y + 218), color=COL["gold"], lw=2.0, ms=14)

    state_x, state_y = sim_x + 70, sim_y + 214
    draw_state_stack(
        ax,
        state_x,
        state_y,
        78,
        146,
        [
            (r"$h_t$", COL["paper"], COL["ink"]),
            (r"$q_i$", COL["blue_soft"], COL["blue"]),
            (r"$l_t$", "#fbfaf0", COL["ink"]),
        ],
        stroke=COL["blue"],
        label=r"$s_t = [h_t, q_i, l_t]$",
        label_color=COL["blue"],
    )

    exp_x, exp_y, exp_w, exp_h = sim_x + 225, sim_y + 135, 320, 162
    box(ax, exp_x, exp_y, exp_w, exp_h, fc=COL["paper"], ec=COL["dash"], lw=2.0, r=6, ls=(0, (6, 5)), z=9)
    text(ax, exp_x + exp_w / 2, exp_y + 27, "Exploration set", size=10.0, weight="bold")
    text(ax, exp_x + exp_w / 2, exp_y + 55, "construction", size=10.0, weight="bold")
    pill(ax, exp_x + 25, exp_y + 92, 82, 52, "Top-M", COL["green_soft"], COL["green"], size=8.5)
    pill(ax, exp_x + 127, exp_y + 92, 102, 52, "course\nprior", COL["gold_soft"], COL["gold"], size=8.2)
    pill(ax, exp_x + 249, exp_y + 92, 56, 52, r"$a_{end}$", COL["red_soft"], COL["red"], size=8.5)
    text(ax, exp_x + exp_w / 2, exp_y + exp_h + 27, r"exploration set $S_t$", size=7.6, color=COL["muted"])

    agent_x, agent_y, agent_w, agent_h = sim_x + 230, sim_y + 365, 330, 100
    box(ax, agent_x, agent_y, agent_w, agent_h, fc=COL["paper"], ec=COL["line"], lw=2.2, r=7, z=11)
    text(ax, agent_x + agent_w / 2, agent_y + 32, "actor-critic agent", size=10.2, weight="bold")
    text(ax, agent_x + agent_w / 2, agent_y + 68, r"$\pi_\theta(a \mid s_t, S_t)$", size=9.0, color=COL["muted"])

    trans_x, trans_y, trans_w, trans_h = sim_x + 610, sim_y + 135, 302, 330
    box(ax, trans_x, trans_y, trans_w, trans_h, fc=COL["paper"], ec=COL["dash"], lw=2.0, r=6, ls=(0, (6, 5)), z=9)
    text(ax, trans_x + trans_w / 2, trans_y + 42, "State transition", size=10.5, weight="bold")
    vector_bar(ax, trans_x + 38, trans_y + 92, n=4, w=22, h=22)
    arrow(ax, (trans_x + 135, trans_y + 103), (trans_x + 174, trans_y + 103), color=COL["blue"], lw=2.0, ms=14)
    vector_bar(ax, trans_x + 194, trans_y + 92, n=4, w=22, h=22, colors=[COL["paper"], COL["blue_soft"], "#a8bdce", "#396d97"])
    text(ax, trans_x + 80, trans_y + 154, r"$h_t$", size=8.7, color=COL["blue"])
    text(ax, trans_x + 234, trans_y + 154, r"$h_{t+1}$", size=8.7, color=COL["blue"])
    text(ax, trans_x + trans_w / 2, trans_y + 211, r"$h_{t+1} = \rho(h_t, a_t, q_i)$", size=8.8, color=COL["muted"])
    text(ax, trans_x + trans_w / 2, trans_y + 260, r"$l_{t+1} = l_t - 1$", size=9.2, color=COL["gold"])

    next_x, next_y = sim_x + 922, sim_y + 214
    draw_state_stack(
        ax,
        next_x,
        next_y,
        68,
        146,
        [
            (r"$h_{t+1}$", COL["paper"], COL["ink"]),
            (r"$q_i$", COL["red_soft"], COL["red"]),
            (r"$l_{t+1}$", "#fbfaf0", COL["ink"]),
        ],
        stroke=COL["red"],
        label=r"$s_{t+1}$",
        label_color=COL["red"],
    )

    arrow(ax, (state_x + 78, state_y + 73), (exp_x, exp_y + 112), color=COL["blue"], lw=2.4, ms=17)
    arrow(ax, (state_x + 78, state_y + 73), (agent_x, agent_y + 50), color=COL["blue"], lw=2.4, ms=17)
    arrow(ax, (exp_x + exp_w / 2, exp_y + exp_h), (agent_x + agent_w / 2, agent_y), color=COL["orange"], lw=2.4, ms=17)
    text(ax, agent_x + agent_w + 28, agent_y + 48, r"$a_t$", size=8.5, weight="bold", color=COL["muted"])
    arrow(ax, (agent_x + agent_w, agent_y + 50), (trans_x, trans_y + 220), color=COL["blue"], lw=2.4, ms=17)
    arrow(ax, (trans_x + trans_w, trans_y + 219), (next_x, next_y + 73), color=COL["blue"], lw=2.4, ms=17)

    reward_x, reward_y, reward_w, reward_h = sim_x + 100, sim_y + 515, 825, 250
    box(ax, reward_x, reward_y, reward_w, reward_h, fc="#fbfbf6", ec=COL["dash"], lw=2.0, r=6, ls=(0, (6, 5)), z=9)
    text(ax, reward_x + reward_w / 2, reward_y + 32, "Reward computation", size=11, weight="bold", color=COL["ink"])

    comp_w, comp_h = 228, 58
    gap_x, gap_y = 28, 18
    c0x, c0y = reward_x + 38, reward_y + 56
    draw_reward_component(ax, c0x, c0y, comp_w, comp_h, "target align.", r"$h_{t+1} \approx q_i$", COL["blue"])
    draw_reward_component(ax, c0x + comp_w + gap_x, c0y, comp_w, comp_h, "progress", r"$d_t - d_{t+1}$", COL["blue"])
    draw_reward_component(ax, c0x + 2 * (comp_w + gap_x), c0y, comp_w, comp_h, "concept", r"$s_u^T A_i^{con}$", COL["orange"])
    draw_reward_component(ax, c0x, c0y + comp_h + gap_y, comp_w, comp_h, "prereq gap", r"$A_i^{pre}$", COL["orange"])
    draw_reward_component(ax, c0x + comp_w + gap_x, c0y + comp_h + gap_y, comp_w, comp_h, "difficulty gap", r"$d_i - r_u$", COL["orange"])
    draw_reward_component(ax, c0x + 2 * (comp_w + gap_x), c0y + comp_h + gap_y, comp_w, comp_h, "repeat", r"$\mathbb{1}[a_t \in H_u]$", COL["red"])
    text(ax, reward_x + reward_w / 2, reward_y + reward_h - 22, r"$r_t = \sum_k \lambda_k r_t^k$", size=9.3, weight="bold", color=COL["red"])

    arrow(
        ax,
        (reward_x + reward_w / 2, reward_y),
        (agent_x + agent_w / 2, agent_y + agent_h),
        color=COL["orange"],
        lw=2.3,
        ms=17,
        rad=-0.15,
    )
    text(ax, agent_x + agent_w / 2 + 122, agent_y + agent_h + 22, "training feedback", size=7.5, weight="bold", color=COL["orange"])


def draw_panel_c(ax):
    x, y = PANELS["c"]["x"], PANELS["c"]["y"]

    inf_x, inf_y, inf_w, inf_h = x + 54, y + 78, 592, 94
    box(ax, inf_x, inf_y, inf_w, inf_h, fc="#f6f0dd", ec=COL["line"], lw=2.1, r=5, z=8)
    text(ax, inf_x + inf_w / 2, inf_y + 34, "strict item-cold inference", size=10.2, weight="bold")
    text(ax, inf_x + inf_w / 2, inf_y + 68, "cold-only full catalog", size=9.2, color=COL["muted"])

    score_x, score_y, score_w, score_h = x + 44, y + 215, 612, 600
    box(ax, score_x, score_y, score_w, score_h, fc="#fffef9", ec=COL["line"], lw=2.2, r=6, ls=(0, (7, 6)), z=8)
    text(ax, score_x + 36, score_y + 48, "dot-product scoring", size=10.6, weight="bold", ha="left")

    user_x, user_y = score_x + 50, score_y + 128
    ax.add_patch(patches.Circle((user_x, user_y + 4), 13, facecolor=COL["paper"], edgecolor=COL["blue"], linewidth=2.0, zorder=14))
    line(ax, [(user_x, user_y + 18), (user_x, user_y + 50)], color=COL["blue"], lw=2.0, z=14)
    line(ax, [(user_x - 20, user_y + 62), (user_x, user_y + 42), (user_x + 20, user_y + 62)], color=COL["blue"], lw=2.0, z=14)
    vector_bar(
        ax,
        user_x + 35,
        user_y - 4,
        n=4,
        w=28,
        h=25,
        colors=[COL["paper"], "#f2e4d8", "#cc7d4c", "#9a502d"],
    )
    text(ax, user_x + 107, user_y + 72, r"$z_u$", size=8.5, color=COL["orange"], weight="bold")
    text(ax, user_x + 112, user_y - 38, "user embedding", size=8.7, weight="bold", color=COL["orange"])

    bank_x, bank_y = score_x + 55, score_y + 360
    draw_course_icon(ax, bank_x + 30, bank_y - 28, scale=0.45)
    text(ax, bank_x - 5, bank_y - 64, "cold-course bank", size=7.8, weight="bold", color=COL["green"], ha="left")
    text(ax, bank_x - 5, bank_y - 36, "cold bank", size=9.0, weight="bold", color=COL["green"], ha="left")
    vector_bar(ax, bank_x + 10, bank_y + 8, n=5, w=28, h=27, colors=[COL["paper"], COL["green_soft"], "#bad5c0", "#82a987", COL["green"]])
    text(ax, bank_x + 78, bank_y + 80, r"$Z_{cold}$", size=9.0, weight="bold", color=COL["green"])

    dot_x, dot_y, dot_w, dot_h = score_x + 245, score_y + 214, 170, 165
    box(ax, dot_x, dot_y, dot_w, dot_h, fc=COL["paper"], ec=COL["blue"], lw=2.3, r=6, z=11)
    text(ax, dot_x + dot_w / 2, dot_y + 44, "dot-product", size=8.8, weight="bold")
    text(ax, dot_x + dot_w / 2, dot_y + 72, "scoring", size=8.8, weight="bold")
    text(ax, dot_x + dot_w / 2, dot_y + 112, r"$s(u,i)$", size=8.8, weight="bold", color=COL["muted"])
    text(ax, dot_x + dot_w / 2, dot_y + 138, r"$= z_u^T z_i^{cold}$", size=8.2, color=COL["muted"])

    top_x, top_y, top_w, top_h = score_x + 455, score_y + 150, 135, 310
    box(ax, top_x, top_y, top_w, top_h, fc=COL["paper"], ec=COL["line"], lw=2.2, r=6, z=11)
    text(ax, top_x + top_w / 2, top_y + 34, "Top-K", size=9.8, weight="bold")
    text(ax, top_x + top_w / 2, top_y + 64, "courses", size=9.8, weight="bold")
    rows = [("1", "C-17"), ("2", "C-04"), ("3", "C-31"), ("K", "C-K")]
    row_y = top_y + 96
    for idx, (rank, course) in enumerate(rows):
        yy = row_y + idx * 50
        fc = COL["blue_soft"] if idx < 2 else COL["paper"]
        box(ax, top_x + 12, yy, top_w - 24, 39, fc=fc, ec=COL["blue"], lw=1.3, r=3, z=12)
        text(ax, top_x + 31, yy + 20, rank, size=8.0, weight="bold", color=COL["muted"])
        line(ax, [(top_x + 50, yy), (top_x + 50, yy + 39)], color=COL["blue"], lw=1.2, z=14)
        text(ax, top_x + 88, yy + 20, course, size=8.1, weight="bold", color=COL["blue"])

    arrow(ax, (user_x + 147, user_y + 20), (dot_x, dot_y + 58), color=COL["orange"], lw=2.4, ms=17)
    orth_arrow(
        ax,
        [
            (bank_x + 150, bank_y + 24),
            (dot_x - 26, bank_y + 24),
            (dot_x - 26, dot_y + 112),
            (dot_x, dot_y + 112),
        ],
        color=COL["green"],
        lw=2.4,
        ms=17,
    )
    arrow(ax, (dot_x + dot_w, dot_y + dot_h / 2), (top_x, top_y + top_h / 2), color=COL["blue"], lw=2.4, ms=17)


def draw_inter_panel_links(ax):
    a = PANELS["a"]
    b = PANELS["b"]
    c = PANELS["c"]

    red_hollow_arrow(ax, a["x"] + a["w"] + 16, a["y"] + 355, scale=0.95)
    text(ax, a["x"] + a["w"] + 24, a["y"] + 310, r"$q_i$", size=8.5, weight="bold", color=COL["green"])

    red_hollow_arrow(ax, b["x"] + b["w"] + 14, b["y"] + 450, scale=1.0)



def build_figure():
    fig = plt.figure(figsize=(W / 300, H / 300), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(COL["paper"])
    ax.set_facecolor(COL["paper"])

    for key in ["a", "b", "c"]:
        panel(ax, key)
    draw_panel_titles(ax)
    draw_panel_a(ax)
    draw_panel_b(ax)
    draw_panel_c(ax)
    draw_inter_panel_links(ax)
    return fig


def main():
    out_dir = Path(__file__).resolve().parent
    spec_dir = out_dir / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)

    spec_path = spec_dir / "ckg_rl_framework_structure_preserved_spec.json"
    spec_path.write_text(json.dumps(LAYOUT_SPEC, indent=2), encoding="utf-8")

    fig = build_figure()
    outputs = {
        "svg": out_dir / "ckg_rl_framework_structure_preserved.svg",
        "pdf": out_dir / "ckg_rl_framework_structure_preserved.pdf",
        "png": out_dir / "ckg_rl_framework_structure_preserved.png",
    }
    fig.savefig(outputs["svg"], format="svg", pad_inches=0)
    fig.savefig(outputs["pdf"], format="pdf", pad_inches=0)
    fig.savefig(outputs["png"], format="png", dpi=300, pad_inches=0)
    plt.close(fig)

    for path in [spec_path, *outputs.values()]:
        print(path)


if __name__ == "__main__":
    main()
