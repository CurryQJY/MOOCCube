# AAAI Reward-Chip Clarity Design

## Goal

Give the four `Course-info Reward` items more internal space and make the prerequisite and difficulty symbols immediately recognizable.

## Chip Layout

- Increase each reward chip from 94×30 to 104×34.
- Move the chip grid origin from `(x+88, y+35)` to `(x+80, y+32)`.
- Use a horizontal step of 116 and a vertical step of 43, keeping the 2×2 grid centered inside the 310×120 panel.
- Place each icon at `cx+18, cy+17` and each label at `cx+42, cy+17`.
- Keep the enlarged icon size and label size from the previous revision.

## Icon Redesign

- `prereq`: draw a bold three-level staircase rising from lower left to upper right, with a clear arrowhead at the top.
- `difficulty`: draw a bold semicircular gauge with three major ticks, a high-pointing needle, and a visible center hub.
- Preserve the existing orange and gold semantic colors.

## Verification

- Assert all four chips use 104×34 geometry and remain inside the course-reward panel.
- Assert every label has positive clearance from all chip borders and from its icon region.
- Assert the prerequisite icon contains a stepped path and arrowhead.
- Assert the difficulty icon contains a gauge arc, three major ticks, a needle, and a center hub.
- Regenerate SVG, PDF, PNG, and TIFF outputs and compile the paper without fatal errors, overfull boxes, or undefined references.
