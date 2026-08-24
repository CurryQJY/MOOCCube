# AAAI State-Transition Upper-Flow Rebalancing

## Scope

Rearrange the upper brown action-flow arrows and `u_j` node inside the `State Transition Function` box to eliminate excessive whitespace before the right representation. Preserve the centered representation pair, centered red update arrow, representation sizes, labels, box, title, explanatory text, colors, and line styles.

## Approved geometry

- Keep the state-representation centers at approximately `982.321` and `1097.679`, with midpoint `1040`.
- Keep the red update arrow at approximately `1016.321 -> 1063.679`, with midpoint `1040`.
- Keep the left brown curved-arrow start attached to the moved `h_t` head at approximately `x=1004.964`.
- Extend the right brown curved-arrow endpoint to the right representation center at approximately `x=1097.679`.
- Place `u_j` at the midpoint of those two outer brown-arrow connection points: `action_cx = (left_arrow_start_x + right_c.x) / 2`, approximately `1051.321`.
- Keep the left arrow ending 13 units before `action_cx` and the right arrow starting 13 units after `action_cx`.
- The resulting brown arrows have equal horizontal spans of approximately `33.357` and are mirrored around `action_cx`.
- Leave the red update arrow and the representation pair centered around the box center `flow_cx`.

## Verification

- Assert `u_j.x` equals `(left_arrow_start_x + right_c.x) / 2`.
- Assert the right brown curved-arrow endpoint equals the right representation center.
- Assert the left brown curved-arrow start remains attached to the `h_t` head boundary.
- Assert the two brown arrows have equal horizontal spans and matching vertical endpoints.
- Assert the representation-pair midpoint and red-arrow midpoint remain `flow_cx`.
- Assert representation sizes, angles, and label alignment remain unchanged.
- Regenerate SVG, PDF, PNG, and TIFF.
- Inspect both the standalone PNG and the compiled AAAI page for balanced upper-flow spacing, clipping, and readability.
