# AAAI Exploration Pipeline Layout Design

## Goal

Make the exploration-set construction sequence read as a clear three-stage pipeline without text-border collisions or inconsistent label sizes.

## Change

- Rename the three stages to `Retrieve M`, `Sample N`, and `Fit Rerank`.
- Use a common box y-coordinate of 252 and height of 34.
- Place the boxes at x=640/730/824 with widths 70/74/68, respectively.
- Use the same `role_scale(9.0, "symbol")` font size and bold weight for all three labels.
- Add right-pointing micro-flow arrows from x=712 to 728 and from x=806 to 822 at y=269.
- Preserve the surrounding `Exploration Set Construction` container, title, candidate-user row, and all downstream elements.

## Verification

- Assert the three rendered labels have identical font sizes.
- Assert every label bounding box stays inside its corresponding stage box with positive horizontal clearance.
- Assert both arrow paths are present between the boxes and do not overlap the borders.
- Assert the full three-box pipeline remains horizontally centered inside the exploration container.
- Regenerate SVG, PDF, PNG, and TIFF outputs and compile the paper without fatal errors, overfull boxes, or undefined references.
