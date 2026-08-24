# AAAI Lower-Stack Whitespace Balance Design

## Goal

Balance the bottom whitespace of panel a against the whitespace above `Course Encoder` while preserving the internal spacing of the lower course-output and learner-encoder stack.

## Change

- Move the final course representation group upward by 30 canvas units by changing `q_vec_y` from 540 to 510.
- Move the complete learner-encoder group upward by the same 30 units by changing `learner_cy` from 614 to 584.
- Keep the `Learner Encoder` heading at `learner_cy - 30` and the $\mathbf{e}_u$/$\mathbf{z}_u$ labels at `learner_cy + 30`.
- Preserve all sizes, horizontal positions, colors, and the vertical separation between the course output and learner encoder.
- Keep the course-output arrow source attached to the fusion gate; only its endpoint follows `q_vec_y`.

## Target Geometry

- Current top whitespace above `Course Encoder`: approximately 54.2 rendered pixels.
- Expected bottom whitespace after the shift: approximately 51.6 rendered pixels.
- The difference between top and bottom whitespace should be less than 4 rendered pixels.

## Verification

- Assert `q_vec_y == 510` and `learner_cy == 584`.
- Assert the rendered learner heading is at y=554 and the lower learner labels are at y=614.
- Assert the course-output-to-learner-heading clearance remains unchanged.
- Regenerate SVG, PDF, PNG, and TIFF outputs and visually inspect panel a.
- Compile the paper and confirm no fatal errors, overfull boxes, or undefined references.
