# Memory Index

- [CKG-RL cold-start diagnosis](ckg-rl-cold-start-diagnosis.md) — why ablations barely move & the train/inference mismatch that breaks the RL claim
- [Original USIM paper RL design](usim-original-paper-rl-design.md) — Liu et al. sequence-imagination MDP (state/action/transition/reward/deploy)
- [USIM V2/V3 experiment arc](usim-v2-v3-experiment-arc.md) — what V2, V3, V3.1 each did and their seed-2025 numbers
- [V3.2-Clean three-stage design](v32-clean-three-stage-design.md) — the T->G->V3 clean rebuild plan (implemented) + do-not-disturb GPU constraint
- [USIM V3.3-V3.6 experiment arc](usim-v33-v36-experiment-arc.md) — clean V3.2/V3.3/V3.4/V3.5 results and current cold-vs-overall Pareto gap
- [Old strong route audit](old-strong-route-audit.md) — why the historical 0.2863/0.2098 cold numbers cannot go back into the main table
- [Main-table evaluator crack](main-table-evaluator-crack.md) — cold main table uses legacy CKG-RL numbers while overall uses clean re-eval; the two contradict; the real cold/overall target lines
- [Old .2667 status open](old-2667-status-open.md) — .2667's cold is legacy-inflated but its hot/overall are clean & strong; NOT ruled out, needs clean-evaluator re-eval
- [V3.6 result & USIM stop-loss](v36-result-and-usim-stoploss.md) — V3.6 ran, viability gate failed on action_agreement; no test needed; USIM add-mechanism line exhausted
