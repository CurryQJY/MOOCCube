# AAAI State-Transition Whole-Group Centering

## Scope

Center the complete visual transition group inside the `State Transition Function` box in panel (b), while keeping the shortened central red arrow and the `u_j` node individually centered. Preserve representation size, representation angle, internal state spacing, box, title, explanatory text, colors, and line styles.

## Approved geometry

- Keep `u_j` and the central red update arrow centered at the box center `flow_cx`.
- Preserve the current left-to-right representation spacing of approximately `115.357` coordinate units.
- Translate both state representations and their labels together by the same horizontal offset so their midpoint equals `flow_cx`.
- Derive `group_shift = flow_cx - (base_left_c.x + base_right_c.x) / 2`.
- For the production box, `group_shift` is approximately `11.321`; the moved centers are `h_t: x=982.321` and `h_{t+1}: x=1097.679`.
- Keep both tilted representations at width `63`, height `16`, and angle `-48` degrees.
- Recompute the left curved-arrow start from the moved `h_t` head and mirror the right curved arrow around `flow_cx`.
- Keep the red update-arrow span at approximately `47.357`, with endpoints approximately `1016.321 -> 1063.679` and midpoint exactly `1040`.
- Leave the centered title and explanatory text unchanged.

## Verification

- Assert the two representation centers have midpoint `flow_cx`.
- Assert both labels use their moved representation-center x coordinates.
- Assert the representation spacing remains unchanged.
- Assert both representation sizes and angles remain unchanged.
- Assert the upper curved arrows are exact horizontal mirrors around `flow_cx`.
- Assert the left curved arrow starts at the moved `h_t` head boundary.
- Assert the red update-arrow midpoint equals `flow_cx` and its shortened span remains unchanged.
- Regenerate SVG, PDF, PNG, and TIFF.
- Inspect both the standalone PNG and the compiled AAAI page for whole-group centering, clipping, and label alignment.
