# AAAI Reward Icon Redesign

## Goal

Replace the visually dense `prereq` and `difficulty` glyphs in the Course-info Reward panel with symbols that remain recognizable after the full framework figure is reduced to paper size.

## Scope

- Modify only the `prereq` and `difficulty` branches of `reward_term_icon`.
- Preserve the current reward-chip dimensions, positions, labels, semantic colors, and icon anchor points.
- Regenerate the existing SVG, PDF, PNG, and TIFF outputs; do not add a new figure asset or external icon dependency.

## Reference Patterns

- Prerequisite: use the directed-node convention common to dependency and graph icons, rather than a filled staircase.
- Difficulty: use the sparse outline convention common to speedometer and gauge icons, rather than a thick filled arc.
- Online references consulted: Font Awesome `stairs` and `gauge-high`, plus Bootstrap Icons `diagram-3` and `speedometer2`.

## Prerequisite Icon

- Draw three small hollow circular nodes from lower-left to upper-right.
- Connect consecutive nodes with two thin directed segments.
- Give each segment a compact arrowhead that remains separated from the destination node.
- Keep the interior of every node light so the icon retains visible negative space at the production size.
- Use the existing prerequisite orange for strokes and the existing pale-orange accent for node fills.

## Difficulty Icon

- Draw one thin open semicircular gauge outline.
- Add only three short major ticks with clear gaps from one another.
- Draw one high-position needle aimed toward the upper-right/high-difficulty region.
- Use a small, distinct center hub instead of a large filled dot.
- Use the existing gold palette and preserve visible white space between the outline, ticks, needle, and hub.

## Readability Constraints

- All geometry must remain inside the existing 20-point icon allocation.
- No solid region may visually occupy most of the icon bounding box.
- Strokes must remain individually distinguishable in a crop taken from the final full-resolution PNG.
- Neither icon may collide with its label or chip border.

## Verification

1. Run a failing structural assertion against the current icons at production size.
2. Verify the prerequisite icon contains three nodes and directed connectors, with no staircase path.
3. Verify the gauge uses a thin arc, three ticks, one needle, and one small hub.
4. Regenerate all figure formats and inspect an enlarged crop of the Course-info Reward panel.
5. Compile the AAAI paper and require 11 pages, zero fatal errors, zero overfull boxes, and zero undefined references.
