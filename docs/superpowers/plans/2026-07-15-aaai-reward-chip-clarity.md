# AAAI Reward-Chip Clarity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enlarge the four course-reward chips and replace the prerequisite and difficulty icons with clearer, conventional symbols.

**Architecture:** Expand and recenter the existing 2×2 chip grid without changing its parent panel. Redraw only the `prereq` and `difficulty` branches of the reusable reward icon function, preserving semantic colors and the previously enlarged icon size.

**Tech Stack:** Python, Matplotlib, LaTeX/latexmk

---

### Task 1: Expand and recenter the reward chips

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:1129-1138`

- [x] **Step 1: Run the failing chip-geometry assertion**

Monkeypatch `box` and `reward_term_icon`, render `course_info_reward_panel(ax, 0, 0, 310, 120)`, and assert that the four chip boxes are `(80,32,104,34)`, `(196,32,104,34)`, `(80,75,104,34)`, and `(196,75,104,34)`. Assert the icon anchors are at x offsets 18 and y offsets 17, and the label anchors use x offset 42.

Expected: FAIL because the current chips use 94×30 geometry at x offsets 88/196 and y offsets 35/78.

- [x] **Step 2: Apply the new grid geometry**

Use `chip_x=x+80`, `chip_y=y+32`, `chip_w=104`, `chip_h=34`, `chip_step=116`, and `chip_row_step=43`. Use `cx+18, cy+17` for icons and `cx+42, cy+17` for labels.

- [x] **Step 3: Re-run the chip-geometry assertion**

Expected: exit code 0.

### Task 2: Redraw prerequisite and difficulty icons

**Files:**
- Modify: `paper_aaai27/figures/plot_ckg_rl_framework_topconf.py:865-877`

- [x] **Step 1: Run the failing icon-structure assertion**

Render a `prereq` icon and assert it contains one bold stepped line with at least eight vertices and one filled arrowhead polygon, with no block rectangles. Render a `difficulty` icon and assert it contains one arc with line width at least 2.0, exactly three tick lines plus one needle line, and one center circle.

Expected: FAIL because the current prerequisite icon uses three block rectangles and the current gauge arc is thinner with four ticks.

- [x] **Step 2: Implement the clear prerequisite icon**

Draw a lower-left to upper-right three-level staircase using a single bold line, followed by a filled upward arrowhead at the top step.

- [x] **Step 3: Implement the clear difficulty gauge**

Draw a bold semicircular arc, three major ticks, a high-pointing needle, and a visible center hub.

- [x] **Step 4: Re-run the icon-structure assertion**

Expected: exit code 0.

### Task 3: Regenerate and verify the paper

**Files:**
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.svg`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.pdf`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.png`
- Regenerate: `paper_aaai27/figures/ckg_rl_framework_topconf.tiff`

- [x] **Step 1: Regenerate all outputs**

```powershell
cd paper_aaai27
python figures/plot_ckg_rl_framework_topconf.py
```

Expected: four `saved:` lines.

- [x] **Step 2: Inspect boundaries and readability**

Confirm every label has positive clearance from its chip border and icon region, all icons remain inside their chips, and the prerequisite/difficulty meanings are visually recognizable.

- [x] **Step 3: Compile the paper**

```powershell
cd paper_aaai27
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build_state_transition_symmetry_verify main.tex
```

Expected: 11 pages, zero fatal errors, zero overfull boxes, and zero undefined references.

Keep the implementation in the current branch and workspace without creating an implementation commit.
