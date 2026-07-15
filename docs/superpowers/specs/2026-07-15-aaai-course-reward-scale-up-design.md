# AAAI Course-Reward Scale-Up Design

## Goal

Enlarge the complete Course-info Reward presentation so its four chips, labels, and icons remain readable after the full framework figure is reduced to paper size.

## Root Cause

The current Course-info Reward panel is only 310 by 120 coordinate units. Each chip is 104 by 34, while its rendered icon receives about 22 units. Replacing the glyph geometry within that unchanged allocation cannot provide enough negative space; the entire local layout must grow before the icons are redrawn.

## Layout Strategy

- Preserve the current two-by-two chip arrangement.
- Expand the Reward Function container leftward and upward while keeping its right and bottom boundaries unchanged.
- Keep the Embedding Alignment Reward panel at its current horizontal position and width.
- Move the plus sign and Course-info Reward panel left into the newly created space.
- Keep the Course-info Reward right edge at its current x-coordinate so the feedback arrow at the right remains clear.

## Target Geometry

- Reward Function container: move from `(545, 468, 660, 178)` to `(525, 458, 680, 188)`.
- Embedding Alignment Reward panel: retain x-position 545 and width 272; increase its height from 120 to 130 with the enlarged parent.
- Course-info Reward panel: move from x 875 to x 835 and grow from 310 by 120 to 350 by 130.
- Gap between the alignment and course panels: 18 units, with the plus sign centered in that gap.
- Course-info Reward title: increase the component base size from 10.0 to 11.0.
- Course book icon: increase the icon base size from 40 to 46.
- Course symbol: increase the symbol base size from 9.0 to 10.0.
- Reward chips: grow from 104 by 34 to 116 by 40.
- Chip grid: start at offset `(88, 34)`, use horizontal step 126 and vertical step 48.
- Chip icon anchor: use offset `(22, 20)` and increase the icon base size from 20 to 28.
- Chip label anchor: use offset `(48, 20)` and increase the symbol base size from 8.0 to 9.2.

## Icon Redrawing

- Recompute both `prereq` and `difficulty` geometry using the enlarged production icon size rather than merely scaling the former small glyph.
- Prerequisite nodes and arrowheads must retain visible gaps at the final PNG scale.
- Difficulty arc, ticks, needle, and hub must remain visually separate at the final PNG scale.
- Preserve the existing orange and gold semantic palettes.

## Scope

- Modify only the Reward Function geometry, the Course-info Reward panel, and the two affected icon branches.
- Do not alter the ranking panel, state-transition panel, feedback-arrow route, or global figure dimensions.
- Regenerate the existing SVG, PDF, PNG, and TIFF outputs without adding new assets.
- Keep the figure implementation uncommitted in the current branch and workspace.

## Verification

1. Run failing geometry assertions for the enlarged outer container, Course-info panel, chips, labels, and icon anchors.
2. Run failing production-size structure and spacing assertions for the two enlarged icons.
3. Implement the confirmed geometry and redraw both icons at the new size.
4. Regenerate all figure formats and inspect a full-resolution crop of the complete Reward Function region.
5. Confirm no chip, text, or icon crosses its boundary and the right-side feedback arrow remains unobstructed.
6. Compile the paper and require 11 pages, zero fatal errors, zero overfull boxes, and zero undefined references.
