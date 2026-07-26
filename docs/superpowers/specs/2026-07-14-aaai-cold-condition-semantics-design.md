# AAAI Cold-Condition Semantics Design

## Goal

Separate the generic course entity from the strict item-cold condition so the figure does not imply that warm and cold courses use different initialization paths.

## Approved Semantics

- Keep the input label as the neutral `course c`.
- Remove the snowflake from the generic course input icon.
- Keep the content and behavior branches, fusion gate, and output embedding path unchanged for both warm and cold courses.
- Mark the cold-specific intervention at the `ID mask` operation by placing the snowflake beside that mask.
- Remove the snowflake from the shared `Course-info Reward` course icon because that reward path is not a separate cold-only initialization path.
- Keep snowflakes on explicitly cold Top-K course rows in panel c.
- Keep the snowflake in the global legend and rename its label from `Cold Course` to `Cold / ID-Masked`.

## Layout

- Place the new mask snowflake to the right of the `ID mask` chip, inside the behavior-embedding card and away from the downstream vector and variable label.
- Preserve the current card sizes, arrows, title sizes, and all embedding positions.
- Rebalance the legend label only if the longer wording approaches the next legend item.

## Verification

- Assert that no snowflake remains on the generic course input or the shared course-info reward icon.
- Assert that one snowflake appears beside the ID mask.
- Assert that the two explicitly cold Top-K rows retain their snowflakes.
- Assert that the legend contains `Cold / ID-Masked` and no longer contains `Cold Course`.
- Regenerate SVG, PDF, PNG, and TIFF outputs and inspect the complete figure.
- Compile the AAAI manuscript and require zero fatal errors, zero overfull boxes, and no undefined references.
