# AAAI State-Transition Arrow Attachment Adjustment

## Scope

Adjust only the left curved action arrow inside the `State Transition Function` box in panel (b). Preserve the original state-representation sizes and positions, the right curved arrow, box, title, center user node, horizontal update arrow, explanatory text, colors, and line styles.

## Approved geometry

- Restore both tilted representations to their original width `63`, height `16`, and angle `-48` degrees.
- Restore the representation centers to `flow_cx - 69` and `flow_cx + 69`.
- Keep the `h_t` and `h_{t+1}` labels centered at those original representation-center positions.
- Keep the left curved arrow directed from the right side of the dark head of the `h_t` representation to the left edge of the center user node.
- Compute the left-arrow start from the rotated right edge of the original `h_t` vector at the arrow's vertical coordinate; do not move or resize the vector to meet the arrow.
- Keep the left arrow's end at the left edge of the center user node and preserve its curvature, line width, arrowhead size, and opacity.
- Keep the right curved arrow unchanged, from `(flow_cx + 13, y + 50)` to `(flow_cx + 69, y + 52)`.
- Accept that the two curved arrows have different horizontal spans: preserving the original representations and making the left arrow originate at the `h_t` head takes priority over exact arrow-length symmetry.
- Keep the paired representation centers and the complete state-representation composition centered at `flow_cx`.

## Verification

- Assert the two representation centers and labels are restored to `flow_cx - 69` and `flow_cx + 69`.
- Assert both tilted-vector calls use width `63`, height `16`, and angle `-48`.
- Assert the projected right boundary of the left tilted-vector head equals the left-arrow start to numerical tolerance.
- Assert the right curved arrow remains exactly `(flow_cx + 13, y + 50)` to `(flow_cx + 69, y + 52)`.
- Regenerate SVG, PDF, PNG, and TIFF.
- Inspect both the standalone PNG and the compiled AAAI page for clean left-arrow attachment, clipping, and label alignment.
