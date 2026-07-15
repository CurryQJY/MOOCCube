# AAAI Update-Arrow Alignment Design

## Goal

Horizontally align the red update arrow inside `State Transition Function` with the black main-flow arrow immediately to its right.

## Change

- Move only the red update arrow from `y + 74` to `y + 82`.
- With the current state-transition panel origin `y = 190`, the red arrow moves from absolute y=264 to y=272.
- Preserve the arrow's x endpoints, style, width, color, and z-order.
- Preserve both tilted state representations, the orange action curves and node, all labels, the right black arrow, and the next-state capsule.

## Verification

- Assert the red `update` arrow is drawn at y=272.
- Assert the black `data` arrow from x=1150 to 1192 remains at y=272.
- Assert all non-update geometry in the state-transition panel retains its previous coordinates.
- Regenerate SVG, PDF, PNG, and TIFF outputs and compile the paper without fatal errors, overfull boxes, or undefined references.
