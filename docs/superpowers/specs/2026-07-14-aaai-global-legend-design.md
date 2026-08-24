# AAAI Framework Figure Global Legend Design

## Goal

Add a compact legend for the entire CKG-RL framework figure without making it look like part of the ranking computation in panel c.

## Approved Layout

- Keep the panel-c title and subtitle unchanged.
- Move the complete ranking computation group upward as one unit: the user input, course input, two elbow arrows, dot-product node, output arrow, and Top-K list.
- Keep the top of the dashed `Ranking Logit` region fixed and raise its bottom edge so it tightly contains the moved computation.
- Use the freed lower band of panel c for a separate solid-outline box titled `Legend`.
- Give the legend a neutral paper-colored fill and panel-colored border so it reads as figure-level metadata rather than another model component.

## Legend Content

Use a two-column, two-row layout with four recurring visual conventions:

1. Snowflake icon: `Cold Course`
2. Segmented vector swatch: `Embedding`
3. Solid arrow: `Main Flow`
4. Dashed arrow: `Feedback / Auxiliary Flow`

Person and course-book icons are excluded because their meanings are already self-evident and adding them would reduce print readability.

## Geometry and Hierarchy

- Preserve the existing horizontal coordinates of the ranking computation.
- Apply one shared vertical offset to all computation elements so arrows remain connected.
- Leave a clear gap between the shortened dashed ranking region and the new legend box.
- Keep all legend labels smaller than component labels but large enough to remain legible in the two-column AAAI rendering.
- Keep the legend entirely inside panel c and below the dashed ranking region.

## Verification

- Assert that every ranking element receives the same vertical offset and that all arrow endpoints remain aligned with the dot-product node.
- Confirm that the dashed region and legend box do not overlap and both remain inside panel c.
- Regenerate SVG, PDF, PNG, and TIFF outputs.
- Inspect the standalone figure and page 4 of the compiled AAAI manuscript.
- Require zero fatal LaTeX errors and zero overfull boxes.
