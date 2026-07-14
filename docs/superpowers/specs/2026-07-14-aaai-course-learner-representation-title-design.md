# AAAI Figure A Title Adjustment

## Scope

Adjust only the textual hierarchy at the top of panel (a) in the AAAI CKG-RL model figure. Internal encoder geometry, arrows, icons, equations, colors, and the other panels remain unchanged.

## Approved wording

- Panel title: `Course and Learner Representation`
- Upper subsection title: `Course Encoder`
- Lower subsection title: `Learner Encoder` (unchanged)

## Layout rules

- Keep the panel title in its existing header position and style.
- Keep `Course Encoder` at the existing upper subsection-title anchor.
- Use the same font size, weight, and visual hierarchy for `Course Encoder` and `Learner Encoder`.
- Do not move the course icon or encoder components unless rendering reveals an actual overlap.

## Verification

- Regenerate SVG, PDF, PNG, and TIFF from the Python source.
- Inspect the PNG for clipping, overlap, and alignment.
- Compile `paper_aaai27/main.tex` in an isolated output directory and inspect the page containing the model figure.
