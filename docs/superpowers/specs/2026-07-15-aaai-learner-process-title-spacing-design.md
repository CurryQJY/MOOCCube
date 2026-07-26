# AAAI Learner Process and Title Spacing Design

## Goal

Increase the vertical separation between the `Learner Encoder` heading and its representation-initialization process without moving the heading or course-output group.

## Change

- Introduce an independent `learner_title_y = 554` anchor for the heading.
- Move the process anchor from `learner_cy = 584` to `learner_cy = 592`.
- Keep the user icon, $\mathbf{e}_u$ vector, MLP, $\mathbf{z}_u$ vector, connecting arrows, and lower labels tied to `learner_cy` so they move downward together by 8 units.
- Keep `q_vec_y = 510`, all horizontal positions, sizes, colors, and intra-process spacing unchanged.

## Verification

- Assert the heading remains at y=554.
- Assert the process center moves to y=592 and the lower learner labels move to y=622.
- Assert the final course-output label remains at y=519.
- Confirm panel-a bottom clearance remains above 35 rendered pixels.
- Regenerate SVG, PDF, PNG, and TIFF outputs and compile the paper without fatal errors, overfull boxes, or undefined references.
