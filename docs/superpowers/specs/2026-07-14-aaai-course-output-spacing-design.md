# AAAI Course Output Spacing Design

## Goal

Increase the vertical separation between the final course representation and the `Learner Encoder` heading in panel a.

## Change

- Move the final course representation vector and its $\mathbf{e}_c$ label upward by 12 canvas units.
- Move the endpoint of the incoming vertical arrow to the same new vertical coordinate.
- Keep the arrow source attached to the fusion gate.
- Preserve all sizes, horizontal positions, colors, and learner-encoder elements.

## Verification

- Assert that `q_vec_y` changes from 552 to 540.
- Regenerate SVG, PDF, PNG, and TIFF outputs.
- Visually confirm that the final course representation remains centered under the fusion gate and has more clearance above `Learner Encoder`.
- Compile the paper and confirm no fatal errors, overfull boxes, or undefined references.
