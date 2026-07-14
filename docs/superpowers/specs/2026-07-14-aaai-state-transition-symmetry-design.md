# AAAI State-Transition Symmetry Adjustment

## Scope

Adjust the two curved action arrows and the paired state-representation groups inside the `State Transition Function` box in panel (b). Preserve the box, title, center user node, horizontal update arrow, explanatory text, colors, and line styles.

## Approved geometry

- Use the box center `flow_cx` as the horizontal symmetry axis.
- Keep the existing long curved-arrow endpoints at `flow_cx - 69` and `flow_cx + 69`.
- Keep the left curved arrow directed from the right side of the dark head of the `h_t` representation to the left edge of the center user node.
- Make the right curved arrow the exact horizontal mirror of the left arrow: it starts at the right edge of the center user node and ends before the `h_{t+1}` representation.
- Keep both curved arrows at the same vertical coordinates, curvature, line width, arrowhead size, and opacity.
- Keep the original long right arrow; do not shorten it to repair the left-side attachment.
- Treat each tilted representation and its math label as one movable group.
- Reduce both tilted representations from width `63` to width `50` so they can move outward without touching the dashed box after stroke rendering.
- Place the representation centers at `flow_cx - 82` and `flow_cx + 82`.
- At the curved-arrow vertical coordinate, align the right boundary of the tilted `h_t` head with the left arrow's outer endpoint, so the arrow visibly starts from the head rather than to its left.
- Center `h_t` and `h_{t+1}` beneath their respective tilted representations after the symmetric outward move.
- Keep the complete transition composition structurally centered at the box center `flow_cx`.

## Verification

- Assert the right arrow endpoints equal the horizontal mirror of the left arrow endpoints around `flow_cx`.
- Assert the long curved-arrow endpoints remain at `flow_cx - 69` and `flow_cx + 69`.
- Assert the two representation centers and labels are at `flow_cx - 82` and `flow_cx + 82`.
- Assert the tilted-vector calls use width `50` and remain equidistant from `flow_cx`.
- Assert the projected right boundary of the left tilted-vector head meets the arrow start to rendering tolerance.
- Assert the projected tilted-vector bounds keep at least four coordinate units of horizontal clearance from both dashed box edges.
- Regenerate SVG, PDF, PNG, and TIFF.
- Inspect both the standalone PNG and the compiled AAAI page for visual symmetry, clipping, and alignment.
