# AAAI Framework Readability Scaling Design

## Goal

Increase the readability of titles and recurring visual symbols across the complete three-panel CKG-RL framework figure while preserving all panel boundaries, internal connections, and semantic groupings.

## Scaling Strategy

Use explicit semantic roles instead of one global font multiplier:

- Panel headers and panel titles: increase by approximately 8%–12%.
- Internal box and component titles: increase by approximately 12%–18%.
- Mathematical state, embedding, action, reward, and candidate-user symbols: increase by approximately 10%–15%.
- Recurrent pictograms and vector glyphs, including learner, course, snowflake, clock, embedding bars, state vectors, reward icons, and action badges: increase by approximately 10%–15% where their parent boxes allow it.
- Legend title, legend symbols, and legend labels: increase by approximately 15%–20% because they are currently the smallest figure-level elements.
- Arrowheads: increase by up to 10%; line widths may increase by up to 5% only when separation from nearby text remains clear.

Explanatory sentences, optimization notes, formulas, and dense secondary annotations do not receive the title multiplier. They may be adjusted only when necessary to preserve hierarchy after neighboring elements grow.

## Title Scope

The title pass covers:

- Panel titles for a, b, and c.
- Panel-a headings: `Course Encoder`, `Content Emb.`, `Behavior Emb.`, and `Learner Encoder`.
- Panel-b headings: `Exploration Set Construction`, `State Transition Function`, `Actor-Critic Agent`, `Reward Function`, `Emb. Alignment Reward`, and `Course-info Reward`.
- Panel-c headings: `Ranking Logit`, `User Embedding`, `Course Embedding`, `Top-K`, and `Legend`.
- Compact row headings such as `Candidate users:` when they function as container labels.

## Symbol Scope

The symbol pass covers visible variables and reusable glyphs that carry model meaning, including course and learner embeddings, state/time symbols, action/reward symbols, candidate-user nodes, ranking symbols, cold-course marks, and the four legend conventions. Plain prose is excluded.

## Collision and Boundary Policy

For every enlarged element:

1. Keep it inside its current semantic parent box.
2. Preserve a visible padding from borders and dashed frames.
3. Prevent overlap with text, icons, arrows, and neighboring cards.
4. Preserve arrow endpoints and data-flow meaning.
5. If the target scale does not fit, first use available whitespace or make a small local position adjustment; otherwise cap that element's scale instead of enlarging its parent panel.

The three outer panel sizes and the full figure canvas remain unchanged.

## Implementation Shape

- Add role-specific scaling constants for panel titles, component titles, symbols, icons, legend content, and arrowheads.
- Apply the constants explicitly at relevant call sites rather than increasing `TEXT_SCALE` globally.
- Keep the existing deterministic Matplotlib source as the single source of truth and regenerate SVG, PDF, PNG, and TIFF from it.

## Verification

- Render the complete figure and inspect each panel at standalone resolution.
- Measure selected title and symbol bounding boxes against their parent regions.
- Check representative dense regions: panel-a branch cards, panel-b candidate row and state-transition card, reward subpanels, panel-c ranking card, and the global legend.
- Compile the AAAI manuscript and inspect page 4 at print scale.
- Require zero fatal LaTeX errors, zero overfull boxes, and no new undefined references.
