# AAAI Candidate-User Node Consistency Design

## Goal

Make the candidate-user nodes $u_1$, $u_j$, and $u_N$ identical in size and shape while preserving the selected-user meaning of $u_j$ through color only.

## Approved Visual Rules

- Draw all three nodes as circles with radius `7.6`.
- Use the same border width, `1.0`, for every node.
- Keep the existing centers, spacing, label positions, and `6.4` label font size.
- Draw unselected nodes with `COL["orange_soft"]` fill, `COL["orange"]` border, and `COL["orange"]` text.
- Draw the selected $u_j$ node with `COL["orange"]` fill, the same `COL["orange"]` border, and `COL["paper"]` text.
- Do not use radius or border-width differences to encode selection.

## Scope

Only `learner_action_set` changes. The selected-action badge used elsewhere, the state-transition $u_j$ node, arrows, spacing, and all other figure elements remain unchanged.

## Verification

- Assert that the three candidate-user circle patches have equal width and height.
- Assert that all three borders have the same line width.
- Assert that only the selected node has the dark fill and light text.
- Regenerate SVG, PDF, PNG, and TIFF outputs.
- Inspect the standalone figure and page 4 of the compiled AAAI manuscript.
