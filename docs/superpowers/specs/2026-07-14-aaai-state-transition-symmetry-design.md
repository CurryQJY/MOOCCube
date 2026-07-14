# AAAI State-Transition Alignment Adjustment

## Scope

Move the `h_{t+1}` representation group left and shorten the central red update arrow while keeping that red arrow horizontally centered inside the `State Transition Function` box in panel (b). Preserve the left representation, both upper curved arrows, box, title, center user node, explanatory text, colors, and line styles.

## Approved geometry

- Keep the left `h_t` representation center at `flow_cx - 69`.
- Keep both tilted representations at width `63`, height `16`, and angle `-48` degrees.
- Keep the upper curved arrows as exact horizontal mirrors around `flow_cx`.
- Use the shortened right curved-arrow endpoint as the moved `h_{t+1}` representation center: `right_c.x = right_arrow_end_x`, approximately `1086.357` for the production box.
- Move the `h_{t+1}` label together with its representation by deriving both from `right_c.x`.
- Derive the shortened red update-arrow span from the moved representation spacing while retaining the existing 34-unit endpoint margins: `update_span = right_c.x - left_c.x - 68`.
- Place the red update-arrow endpoints symmetrically around `flow_cx`: `update_start_x = flow_cx - update_span / 2` and `update_end_x = flow_cx + update_span / 2`.
- For the production box, the red arrow is approximately `1016.321 -> 1063.679`, has span `47.357`, and midpoint `1040`.

## Verification

- Assert the right representation center equals the right curved-arrow endpoint.
- Assert the `h_{t+1}` label uses the moved right representation center.
- Assert both representation sizes and angles remain unchanged.
- Assert the upper curved arrows remain exact horizontal mirrors.
- Assert the red update-arrow span equals `right_c.x - left_c.x - 68`.
- Assert the red update-arrow midpoint equals `flow_cx` to numerical tolerance.
- Regenerate SVG, PDF, PNG, and TIFF.
- Inspect both the standalone PNG and the compiled AAAI page for arrow centering, clipping, and label alignment.
