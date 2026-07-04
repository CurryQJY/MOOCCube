# Strict Item-Cold Paper Redesign

Date: 2026-05-26
Target: English ACM/WSDM-style submission
Source draft: `paper_wsdm/main.tex`

## Goal

Rewrite the paper around strict item-cold MOOC recommendation. The paper should no longer present cold/hot user reward ablation as the main narrative. Instead, it should argue that new-course recommendation requires an item-level cold-start protocol, full-ranking evaluation, and item-macro metrics that avoid popularity and interaction-volume bias.

## Core Claim

FAST3-ContentDelta improves strict item-cold MOOC recommendation by combining content-adaptive course representation, course-knowledge guided simulation, and reinforcement-learning-based ranking optimization. Under the final protocol, test cold courses have no training interactions, and the main metric is full-ranking item-macro Recall/NDCG.

The headline result should use the final three-seed item-macro table:

- FAST3-ContentDelta full cold item-macro R@10 / N@10: 0.2667 / 0.1962.
- CGRC-paper full cold item-macro R@10 / N@10: 0.2589 / 0.1845.
- FAST3-ContentDelta should be positioned as stronger under the strict item-cold primary metric, while hot item-macro metrics are auxiliary stability checks.

## Narrative Changes

The current draft already contains useful method material, but its story is dated. The rewrite should make these changes:

- Title: emphasize strict item-cold or new-course MOOC recommendation.
- Abstract: lead with new-course cold-start, strict item-cold evaluation, and the FAST3-ContentDelta result.
- Introduction: motivate item cold-start in MOOC platforms, distinguish it from user cold-start, and explain why sampled ranking is insufficient.
- Problem formulation: define strict item-cold split, full-ranking evaluation, and item-macro aggregation.
- Method: keep the core FAST3 method, but describe it as serving new-course generalization.
- Experiments: make the main comparison table the central evidence, with course reward ablation as supporting analysis.
- Discussion: explain why course knowledge helps, why user/item aggregation differs, and why hot metrics are not the primary win condition.

## Paper Structure

1. Introduction
   - State the new-course cold-start problem in MOOC recommendation.
   - Explain why collaborative signals and sampled ranking can overstate performance.
   - Introduce strict item-cold full-ranking item-macro evaluation.
   - Summarize FAST3-ContentDelta and headline improvements over strong baselines.

2. Related Work
   - MOOC and educational recommendation.
   - Cold-start recommendation, with emphasis on item cold-start.
   - Reinforcement learning for recommendation.
   - Knowledge-aware and content-aware recommendation.

3. Problem Formulation
   - Define users, courses, interactions, and course content/relations.
   - Define strict item-cold courses as courses unseen during training.
   - Define full-ranking evaluation.
   - Define item-macro aggregation and contrast it with interaction-weighted metrics.

4. Method
   - Content-adaptive course representation with content delta.
   - ID masking or forced-cold handling to reduce reliance on course IDs.
   - Retrieval-based user simulation.
   - Course-knowledge reward terms: concept match, prerequisite safety, difficulty adaptation, and redundancy penalty.
   - PPO-style optimization and final ranking objective.

5. Experiments
   - Dataset and preprocessing: MOOCCube/MOOCCubeX, about 199K users and 698 courses, 768-dimensional content embeddings.
   - Evaluation protocol: strict item-cold balanced split, three seeds, full-ranking item-macro Recall/NDCG.
   - Baselines: Popularity, BPR, LightGCN, DropoutNet, ContentProfile, CCFCRec, ALDI, CGRC-paper.
   - Main results: use the final item-macro table as the primary table.
   - Ablation study: use the existing course-side ablation report to explain individual knowledge signals.
   - Additional analysis: cold/hot item-macro behavior and stability.

6. Discussion
   - Explain why item-macro is appropriate for new-course recommendation.
   - Explain the cold/hot trade-off from course rewards.
   - Clarify limitations: small number of cold courses, dataset-specific course relations, and dependence on content embeddings.

7. Conclusion
   - Restate that strict item-cold MOOC recommendation needs course-side knowledge and evaluation that treats new courses fairly.

## Evidence Sources

Primary result files:

- `outputs/content_delta_pop5/course_ablation_e60_3seed/full/fast3_static_multiseed_summary.csv`
- `outputs/content_delta_pop5/static_item_cold_balanced/main_table_item_macro_final_audit_with_dropoutnet_official_teacher80_student120_cgrc_paper/main_table_item_macro_summary.csv`

Supporting analysis:

- `outputs/usim_feedback_fast3_course_ablation/ablation_report.md`
- `docs/PROJECT_STRUCTURE.md`
- `README.md`

## Main Table Plan

The main table should report full cold item-macro R@5/R@10/R@20 and N@5/N@10/N@20 for all baselines. Hot item-macro metrics can be included in a secondary table or compact appendix-style table if space is tight.

The paper should avoid presenting sampled metrics as the main result. Sampled metrics can appear only as supporting or historical analysis if needed.

## WSDM 2026 Best-Paper Style Calibration

Style reference: WSDM 2026 Best Full Research Paper, `Diversification as Risk Minimization`.

Useful narrative pattern:

- Start from a familiar evaluation practice that looks reasonable on average.
- Show the hidden failure mode: average-oriented evaluation can leave a minority target group poorly served.
- Reframe the problem as risk/robustness, not only aggregate accuracy.
- Introduce the proposed metric or objective as a direct way to optimize the real user/platform need.
- Keep the empirical claim sober: report the main gain, then state the trade-off instead of hiding it.

How to adapt this paper:

- Treat new courses as the under-served target group, analogous to minority intents in diversified search.
- Present interaction-weighted and sampled metrics as useful but insufficient because they can hide poor service to individual cold courses.
- Explain item-macro full-ranking as a risk-sensitive diagnostic: each cold course receives equal weight, so a method must serve the cold-course tail rather than only high-volume courses.
- Emphasize that FAST3-ContentDelta reduces new-course exposure risk under strict item-cold evaluation, rather than claiming universal superiority on all warm/hot metrics.
- Use explicit trade-off language in results and discussion: cold-course robustness improves, while hot-item performance remains an auxiliary stability concern.

## Rewrite Boundaries

Keep:

- Existing ACM template and bibliography setup.
- Method equations that describe content representation, simulator, rewards, PPO, and ranking loss.
- Course-side ablation results as a supporting analysis.

Replace or heavily revise:

- Abstract.
- Introduction.
- Problem formulation.
- Experiment protocol.
- Main result table and result interpretation.
- Conclusion.

Avoid:

- Claiming broad generalization beyond MOOCCube/MOOCCubeX.
- Claiming all hot-start metrics improve.
- Making sampled ranking the main evidence.
- Overstating LLM semantics if the final strict item-cold result depends primarily on content embeddings and course knowledge.

## Acceptance Criteria

The rewritten draft is successful when:

- A reviewer can identify the paper's main problem as strict item-cold/new-course MOOC recommendation within the first page.
- The primary metric is clearly full-ranking item-macro Recall/NDCG.
- The main results table supports the headline claim against CGRC-paper and other baselines.
- The method section still explains FAST3-ContentDelta without becoming a code-level description.
- The ablation section explains course knowledge effects without competing with the main story.
- The conclusion does not claim more than the experiments show.
