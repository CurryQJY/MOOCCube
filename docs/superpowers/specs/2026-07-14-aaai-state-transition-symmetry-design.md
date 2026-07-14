# AAAI State-Transition Symmetry Adjustment

## Scope

Adjust the two curved action arrows and the paired state-representation groups inside the `State Transition Function` box in panel (b). Preserve the box, title, center user node, horizontal update arrow, explanatory text, colors, and line styles.

## Approved geometry

- Use the box center `flow_cx` as the horizontal symmetry axis.
- Keep the left curved arrow from the inner side of the left representation to the left edge of the center user node.
- Make the right curved arrow the exact horizontal mirror of the left arrow: it starts at the right edge of the center user node and ends at the inner side of the right representation.
- Keep both curved arrows at the same vertical coordinates, curvature, line width, arrowhead size, and opacity.
- Keep the original long right arrow and extend the left arrow to its exact horizontal mirror.
- Treat each tilted representation and its math label as one movable group.
- Align the left representation group with the outer endpoint of the left curved arrow and align the right representation group with the outer endpoint of the right curved arrow.
- Center `h_t` and `h_{t+1}` beneath their respective tilted representations after the move.
- Keep the complete transition composition structurally centered at the box center `flow_cx`.

## Verification

- Assert the right arrow endpoints equal the horizontal mirror of the left arrow endpoints around `flow_cx`.
- Assert each state label uses its moved representation-center x coordinate.
- Assert the two representation-group centers remain equidistant from `flow_cx`.
- Regenerate SVG, PDF, PNG, and TIFF.
- Inspect both the standalone PNG and the compiled AAAI page for visual symmetry, clipping, and alignment.
