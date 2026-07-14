# TOME Adaptability Audit

Status: NO-GO for the strict three-dataset item-cold main table.

This audit concerns TOME, *Multi-Type MOOCs Recommendation: Leveraging Deep
Multi-Relational Representation and Hierarchical Reasoning* (AAAI 2025,
https://doi.org/10.1609/aaai.v39i12.33453).

## Previous Work

Before this audit, the workspace contained only paper and code-availability
screening in `A_TIER_COURSE_BASELINE_SCREENING_2026.md`. There was no TOME
repository clone, data adapter, feasibility run, or result report.

## Original Task

TOME is a sequential multi-task model over courses, knowledge concepts, and
videos. It constructs course-relational and concept-relational graphs over
video nodes, builds a time-evolving video graph from learner watch histories,
and trains three classification heads with equal-weight cross-entropy losses.
Hierarchical beam search starts from course predictions, constrains concept
predictions by selected courses, and then constrains video predictions by the
selected concepts.

The paper evaluates course and video prediction with HR@5, NDCG@5, and MRR;
it is not a strict item-cold course-ranking protocol.

## Data Compatibility

MOOCCube has the required raw modalities:

- `additional_information/user_video_act.json` contains per-user video watch
  records with `course_id`, `video_id`, and `local_start_time`.
- `relations/course-video.json` and `relations/video-concept.json` support
  the two static video graphs.

This makes a MOOCCube-only sequence construction technically possible, provided
that each user-video record is filtered to the user's strict training horizon.

Junyi and COCO cannot support the original task: their processed relation
directories contain `course-concept.json` (and, for Junyi, prerequisite
relations), but no user-video history, course-video relation, or video-concept
relation. Dropping the video task and the hierarchical video graph would change
the published method rather than merely adapting its loader.

## Strict Item-Cold Blocker

TOME's course head is a softmax classifier over a fixed course vocabulary. In
the strict split, a cold course has no course-level training label. Its output
row therefore receives no positive supervision and is updated only through the
softmax negative-class term from warm-course examples. The original objective
does not use a course-side content encoder to construct a score for an unseen
course label.

Replacing the classifier with a course-video/content-tied scorer could address
this problem, but would materially change TOME's model and no longer be a
defensible reproduction. An external full-catalog evaluator cannot repair the
untrained cold-course output rows.

## Reproducibility Gate

No official public implementation was found in the local workspace or through
an exact-title/acronym GitHub repository search on 2026-07-13. A from-paper
implementation would additionally require reproducing DGAT, dynamic video
graph construction, adaptive LSTM, three heads, and hierarchical beam search.

## Decision

Do not start a TOME GPU feasibility run or add TOME to the strict three-dataset
main table. Keep it as the closest recent AAAI course-recommendation related
work. Reconsider only if official code becomes available and the authors provide
a cold-course scoring mechanism that avoids training-label leakage.
