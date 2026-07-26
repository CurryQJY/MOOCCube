# Figure 1 Motivation Redesign

## Objective

Rebuild Figure 1 as a motivation diagnosis rather than a second result-comparison figure. It must show why existing methods are insufficient and map each observed limitation to the corresponding CKG-RL design, while leaving numerical improvement claims and trade-off auditing to Figure 3.

## Evidence Roles

### Panel (a): Course-model limitation

- Reference model: PCGNN.
- Panel title: Course model: structural mismatch.
- Show PCGNN held-out MOOCCube test diagnostics for prerequisite gap and difficulty gap.
- Use PCGNN course-level distributions for both diagnostics, summarized with compact horizontal boxplots and directly labeled means.
- Mark both metrics as lower-is-better.
- Do not plot CKG-RL numerical results in this panel.

### Panel (b): Generic cold-start limitation

- Reference model: CGRC.
- Panel title: Cold-start model: weak exposure.
- Show the course-level held-out test NDCG@10 distribution.
- Highlight the supported statement that 46% of audited cold-course cases have NDCG@10 no greater than 0.10.
- Optionally report the descriptive Top-10 cold-course share of 24.6%.
- Do not mix these repaired-export diagnostics with the main-table CGRC checkpoint values.

## Solution Mapping

Add a compact full-width strip below the diagnostic panels:

- Label: CKG-RL response.
- Structural mismatch maps to course-knowledge sampling, educational rewards, and prerequisite supervision.
- Weak exposure maps to content anchoring, cold-ID masking, and learner simulation.
- This strip describes design intent only. It must not claim that every diagnostic is improved.

## Visual Design

- Figure-level title: Why existing methods fall short.
- Remove the standalone legend. Identify PCGNN, CGRC, and CKG-RL directly in panel titles and the response strip.
- Use the existing color roles: orange for PCGNN, gray for CGRC, blue for the CKG-RL response.
- Do not rely on color. PCGNN marks use diagonal hatching, CGRC histogram bars use dotted or cross hatching, and the CKG-RL response strip uses a solid border plus a light fill. Direct labels remain visible in grayscale.
- Preserve a single-column AAAI layout with readable labels after paper-scale rendering.
- Avoid a composite pedagogical-risk score and avoid arbitrary quality thresholds beyond the already reported NDCG@10 cutoff.

## Manuscript Division of Labor

- Figure 1: existing-method problems and targeted method design.
- Figure 3: numerical CKG-RL comparisons, supported improvements, adverse results, inconclusive results, and exposure trade-offs.
- Introduction and Figure 1 caption must not state that CKG-RL has already solved every diagnosed issue.

## Caption

Use a concise caption based on:

> Existing methods leave complementary gaps. Course-specific PCGNN retains prerequisite and difficulty mismatch under zero interaction, while generic cold-start CGRC gives weak Top-10 exposure to many new courses. CKG-RL targets these limitations through course-knowledge constraints and cold-course representation learning; Figure 3 evaluates the resulting improvements and trade-offs. All diagnostics use held-out MOOCCube test data.

## Verification

- Unit tests must confirm that Figure 1 contains no CKG-RL-versus-baseline numerical comparison.
- The generated PDF, SVG, and PNG must be nonempty.
- Render the final main.pdf page containing Figure 1 and check for clipping, overlap, and unreadable labels.
- Render a grayscale copy of Figure 1 and verify that PCGNN, CGRC, and the CKG-RL response remain distinguishable without color.
- Compile the paper without undefined references, undefined citations, undefined control sequences, or overfull boxes.
