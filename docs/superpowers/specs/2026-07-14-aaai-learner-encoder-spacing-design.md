# AAAI Learner Encoder Spacing Design

## Goal

Increase the clearance between the learner-encoder components and the bottom border of panel a without changing their size or horizontal layout.

## Change

- Move the complete learner-encoder group upward by 10 canvas units.
- Change `learner_cy` from 624 to 614.
- Position the `Learner Encoder` heading at `learner_cy - 30`.
- Position the $\mathbf{e}_u$ and $\mathbf{z}_u$ labels at `learner_cy + 30`.
- Keep all existing horizontal coordinates, dimensions, colors, and intra-group spacing.
- Keep the final course representation and its arrow unchanged at `q_vec_y = 540`.

## Verification

- Assert `learner_cy == 614` and `q_vec_y == 540`.
- Assert that the title and lower labels retain offsets of -30 and +30 from `learner_cy`.
- Regenerate SVG, PDF, PNG, and TIFF outputs and visually inspect panel a.
- Compile the paper and confirm no fatal errors, overfull boxes, or undefined references.
