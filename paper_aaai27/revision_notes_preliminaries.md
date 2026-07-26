# Preliminaries Revision Notes

Working file: `main_revised.tex` (original `main.tex` / `main.pdf` untouched).

## Source-status note

| Artifact | Course set | Interaction form | Status |
|---|---|---|---|
| `main.pdf` (user-quoted) | \(\mathcal{I}\) | \((u,c,t)\) | **stale render** |
| `main.tex` | \(\mathcal{C}\) | \((u,c)\) static pairs | newer than PDF |
| `main_revised.tex` | \(\mathcal{C}\) | training set of pairs | this revision |

Edits were made on top of `main.tex`, not the outdated PDF text.

## What changed in Preliminaries

1. **Course catalog** kept as \(\mathcal{C}\) (not \(\mathcal{I}\)); named as *course catalog* for full-ranking language.
2. **Training set** \(\mathcal{D}_{\mathrm{train}}\): wording changed from “static training interactions” to **training set** of observed learner–course pairs.
3. **Dropped timestamps** \((u,c,t)\): protocol depends on membership in \(\mathcal{D}_{\mathrm{train}}\), not click order. Avoids later overload of \(t\) as simulator step.
4. **course-side information → course-related information**, defined as non-interaction evidence attached to courses.
5. **Content feature source made explicit**: frozen text encoder over title + descriptive fields (syllabus / short description), optionally knowledge concepts → \(\mathbf{x}_c\). Matches Implementation Details (BERT [CLS]) without over-specifying encoder here.
6. **Educational signals** reframed as four proxies (topical continuity, prerequisite readiness, relative difficulty, content redundancy), available under zero interactions, **not** full learner-state measurements.
7. Same-section cascade: Main Objective “course-side evidence” → “course-related evidence”.

## Full-text symbol audit (conflicts / risks)

### High

| Issue | Location | Problem | Recommended fix |
|---|---|---|---|
| PDF / source desync | `main.pdf` vs `main.tex` | PDF still uses \(\mathcal{I}\), \((u,c,t)\), \(\mathcal{C}_u\), micro-notation from an older draft | Recompile from `main_revised.tex` before further review |
| Reward \(C(u,c)\) vs set \(\mathcal{C}\) | Method / rewards | Same letter family; oral/reading collision | Keep \(\mathcal{C}\) for catalog; consider renaming reward to \(\mathrm{Con}(u,c)\) or \(B_{\mathrm{con}}(u,c)\) later if needed |
| Projected content \(\mathbf{c}_c\) vs catalog \(\mathcal{C}\) | Content-Anchored | \(\mathbf{c}_c=f_c(\mathbf{x}_c)\) reads as “course of course” | Optional rename to \(\tilde{\mathbf{x}}_c\) or \(\mathbf{x}^{\mathrm{proj}}_c\) in Method pass |

### Medium

| Issue | Location | Problem | Recommended fix |
|---|---|---|---|
| Complexity uses \(I\) for catalog size | Theoretical Complexity | \(I\) was the old course-set letter in PDF; now catalog is \(\mathcal{C}\) | Prefer \(\lvert\mathcal{C}\rvert\) or \(N_c\) instead of bare \(I\) |
| “course-side” remains in prose | Related Work, elsewhere | Inconsistent with new “course-related” | Global replace in next writing pass: course-side → course-related / course knowledge |
| \(p_c\) popularity vs \(\mathbf{p}_u\) learner ID emb. | Masking / Content | Different objects, similar glyph | Acceptable; or rename learner ID emb. to \(\mathbf{e}_u^{\mathrm{id}}\) later |
| Simulator step \(t\) | Method | Fine after dropping interaction time | Keep \(t\) only for simulator steps |
| \(\pi_t\) policy vs \(\Pi_u\) ranking | Method / Preliminaries | Different case/script | OK; do not redefine \(\pi_u\) as ranking |
| Micro/Macro naming | Metrics | `Micro(K)` / `Macro(K)` are operators, not sets | OK; keep roman operator style |

### Low / acceptable

| Issue | Notes |
|---|---|
| \(\mathbf{z}_u\) learner emb. vs \(\mathbf{z}_{c,T}\) course emb. | Same letter, different subscript — standard |
| \(\mathcal{A}_u\) candidates vs action \(a_t\) | Different scripts — OK |
| \(\xi_c\) cold mask vs \(\epsilon\) exploration | OK |
| Indicator \(\mathbb{I}[\cdot]\) | No longer collides with course set \(\mathcal{I}\) after C-migration |
| “training interactions” in cold-start prose | Natural English for zero-interaction courses; only the **set name** should be “training set” |

## Notation foundation after this pass

| Symbol | Meaning |
|---|---|
| \(\mathcal{U}\) | learner set |
| \(\mathcal{C}\) | course catalog |
| \(\mathcal{D}_{\mathrm{train}}\) | training set of learner–course pairs |
| \((u,c)\) | observed interaction pair |
| \(\Pi_u\) | ranked list for learner \(u\) |
| \(\mathbf{x}_c\) | content feature from course text profile |
| educational signals | concept / prerequisite / popularity / history-similarity proxies |

Deferred to later subsections (unchanged here): \(\mathcal{C}_{\mathrm{warm}},\mathcal{C}_{\mathrm{cold}},\mathcal{A}_u,\mathcal{H}_u^{\mathrm{train}}\), metrics, Method embeddings/rewards.

## Cascade checklist (next passes)

- [ ] Related Work: “course-side signals/evidence” wording
- [ ] Theoretical Complexity: replace catalog size \(I\) with \(\lvert\mathcal{C}\rvert\)
- [ ] Method: optional rename \(\mathbf{c}_c\) and reward \(C(\cdot)\) if reviewers find collision
- [ ] Recompile PDF from `main_revised.tex`
- [ ] Align Implementation Details content-source sentence with Preliminaries (already compatible)

## Residual risks left for Method

1. Reward tetrad \(C,P,D,R\) is dense and partially collides with set \(\mathcal{C}\) and difficulty \(d_c\) / gap \(D\).
2. \(\mathbf{c}_c\) projected content is the weakest Method-local name.
3. Difficulty from inverse popularity is a proxy — already caveated in Preliminaries; keep that caveat when \(d_c\) is defined.

## Independent symbol audit (workflow, 2026-07-13)

Additional high-severity collisions confirmed for later Method/Experiments passes:

| Collision | Fix suggestion |
|---|---|
| \(C(u,c)\) concept bonus vs \(\mathcal{C}\) | \(R_{\mathrm{con}}\) / \(B_{\mathrm{con}}\) |
| \(D(u,c)\) difficulty gap vs \(\mathcal{D}_{*}\) | \(G_{\mathrm{diff}}\) |
| \(R(u,c)\) redundancy vs table R@K | \(\mathrm{Pen}_{\mathrm{red}}\) |
| \(\mathbf{c}_c=f_c(\mathbf{x}_c)\) | \(\tilde{\mathbf{x}}_c\) |
| \(r_u\) readiness vs \(r_t\) step reward | readiness \(\rho_u\) |
| \(\gamma_R\) vs discount \(\gamma\) | \(\omega_{\mathrm{red}}\) |
| GAE \(\lambda\) vs \(\lambda_{\mathrm{aux}},\lambda_{\mathrm{pre}}\) | \(\lambda_{\mathrm{GAE}}\) |
| \(\mathcal{H}_u^{\mathrm{train}}\) vs entropy \(H(\pi_t)\) vs complexity bare \(H\) | entropy \(\mathrm{Ent}(\pi_t)\); define or drop complexity \(H\) |
| \(\varepsilon\) exploration vs \(\varepsilon_p\) PPO | \(\varepsilon_{\mathrm{exp}},\varepsilon_{\mathrm{clip}}\) |
| \(\tau_s\) vs ranking \(\tau\) | \(\tau_{\mathrm{samp}},\tau_{\mathrm{rank}}\) |
| sample size \(N\) vs N@K / \(\mathcal{N}_M\) | \(N_{\mathrm{cand}}\) |
| \(d_c\) difficulty vs dim \(d\) | \(\mathrm{diff}_c\) |
| popularity \(p_c\) vs \(\mathbf{p}_u\) | \(n_c^{\mathrm{train}}\) or \(\mathrm{pop}_c\) |
| catalog size bare \(I\) in complexity | \(\lvert\mathcal{C}\rvert\) |

**Design principles for later passes:** calligraphic letters for sets only; never reuse bare roman capitals as both metric abbreviations and reward functions; unique base letter or fixed subscript family per RL hyperparameter; disclose content text source before encoding.
