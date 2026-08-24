---
name: usim-original-paper-rl-design
description: How the original USIM paper (Liu et al.) designs its RL user-sequence-imagination process
metadata:
  type: reference
---

Reference paper: Liu et al., "Fine-tuning out-of-vocabulary item recommendation with user sequence imagination" (PDF at `D:/DeskTop/组会/Liu 等 - Fine-tuning out-of-vocabulary item recommendation with user sequence imagination.pdf`).

The original USIM RL is NOT an online click-simulation environment. It builds a deterministic MDP where "pick a user as an imagined positive feedback and update the item vector accordingly." RL only decides: which user to imagine next, and when to stop.

For a training-phase warm/IV item `i` with known behavior vector `e_i` and interacted user set `U_i`:
- **State**: `s_t=[h_t, l_t]`, `h_0=G(c_i)` (content-generated init), `l_t=N-t` remaining steps. Final `h_T` is the deployed OOV vector.
- **Action**: pick a user `a_t ∈ U ∪ {a_end}` (explicit terminate action).
- **Transition**: `h_{t+1}=h_t+λ∇_{h_t} ŷ_{a_t,i}`; with dot-product scoring reduces to `h_{t+1}=h_t+λ e_{a_t}`.
- **Reward**: `r_t=R_emb+R_rec−p` — embedding alignment to `e_i`, recommendation improvement over full `U_i`, and per-step penalty `p`.
- **Training candidate set**: `TopK(cos(e_i−h_t, e_u)) ∪ U_i ∪ U_rand ∪ {a_end}`. `e_i, U_i` are training-only oracles.
- **Training**: replay buffer + RecPPO; critic supervised so `V([e_i, l])=0` (terminate at target).
- **Deployment**: for new item start from `G(c_i)`, run imagination once → `h_T`, use as static embedding in normal retrieval/ranking. NOT re-run per request user.

Key insight: the original paper's train/inference are legitimately different (training uses `e_i`/`U_i` oracles, deployment does not). That asymmetry is fine. The paper's clean training boundary: history → train IV backbone (`E_u, E_iv`, frozen) → train generator `G` with `||e_i−G(c_i)||²` (frozen) → train USIM policy/value only. Refs: §2 Eq.(1), §3.1–3.5 Eq.(3)–(13), Appendix B Eq.(14), Appendix C Algorithm 1, Appendix D.1.

Data split: 20% items OOV (val/test 1:1); 80% IV items' interactions 8:1:1 train/val/test; `G` early-stops on validation NDCG.

Related: [[ckg-rl-cold-start-diagnosis]], [[ckg-rl-v2-v3-experiment-history]].
