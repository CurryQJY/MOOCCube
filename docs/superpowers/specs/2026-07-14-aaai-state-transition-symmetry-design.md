# AAAI State-Transition Symmetry Adjustment

## Scope

Adjust only the two curved action arrows and the two state-representation labels inside the `State Transition Function` box in panel (b). Preserve the box, title, tilted vectors, center user node, horizontal update arrow, explanatory text, colors, and line styles.

## Approved geometry

- Use the box center `flow_cx` as the horizontal symmetry axis.
- Keep the left curved arrow from the inner side of the left representation to the left edge of the center user node.
- Make the right curved arrow the exact horizontal mirror of the left arrow: it starts at the right edge of the center user node and ends at the inner side of the right representation.
- Keep both curved arrows at the same vertical coordinates, curvature, line width, arrowhead size, and opacity.
- Center `h_t` at `left_c[0]` and `h_{t+1}` at `right_c[0]` so each complete math label is horizontally centered below its tilted representation.

## Verification

- Assert the right arrow endpoints equal the horizontal mirror of the left arrow endpoints around `flow_cx`.
- Assert both state labels use their representation-center x coordinates.
- Regenerate SVG, PDF, PNG, and TIFF.
- Inspect both the standalone PNG and the compiled AAAI page for visual symmetry, clipping, and alignment.
