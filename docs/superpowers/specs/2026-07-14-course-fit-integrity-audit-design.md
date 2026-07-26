# Course-Fit Integrity Audit Design

## Goal

Audit the validation-selected `course_fit` test inference path on the three frozen recovered main-table checkpoints without retraining, modifying checkpoints, or selecting any new hyperparameter from test results.

## Integrity questions

1. Does course-fit read only train-history interactions when scoring candidate users?
2. Does any candidate user's history contain the strict-cold target item?
3. Are all refined catalog items absent from training, and how do the 102 refined items decompose across validation and test cold items?
4. Does the rollout ever receive a behavior/ground-truth target embedding?
5. Does enabling `feedback_course_match_exclude_target` change any per-item or aggregate test result?

## Instrumentation

Keep the recovered model unchanged. Extend `main_checkpoint_actor_inference_ab.py` with evaluation-only monkeypatches that:

- wrap the selected split function and fingerprint the train-only and train-plus-validation history mappings;
- wrap `set_user_seen_index` and classify every installed dense history as `train_only`, `train_plus_validation`, or `unknown`;
- wrap `_compute_candidate_course_fit` and count candidate-user/target pairs for which the target occurs in the candidate user's installed history;
- record all refined item IDs and classify them against train, validation, and test item sets;
- record that `target_emb=None` is supplied to every inference rollout;
- optionally override only `feedback_course_match_exclude_target` during evaluation and restore the checkpoint configuration afterward.

The existing checkpoint write blocker remains active. Audit instrumentation must not affect action scores, item states, evaluator inputs, or checkpoint contents.

## Experiment

Reuse the completed three-seed test `course_fit` outputs as the `exclude_target=False` reference. Replay the same frozen checkpoints once with `exclude_target=True` into a new output root. Compare both aggregate full-ranking CSVs and per-item cold item-macro CSVs exactly.

## Pass criteria

- all installed evaluation histories are classified as `train_only`;
- target-seen candidate pairs and target rows with any seen candidate are both zero;
- no refined item occurs in train;
- the 102 refined items decompose into 34 validation-only and 68 test-only items for each seed;
- every inference rollout has `target_emb=None`;
- aggregate cold item-macro metrics have maximum absolute difference zero;
- per-item cold CSVs are byte-identical with `exclude_target=False`;
- all three source checkpoints predate the audit run and remain unchanged.

Any failure blocks the current course-fit effectiveness claim until its cause is resolved.
