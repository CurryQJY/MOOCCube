# Abstract Reference — Top-Venue Cold-Start Recommendation

Reference paper for our abstract's structure and phrasing.

## Primary reference

**GoRec: A Generative Cold-Start Recommendation Framework**
Haoyue Bai, Min Hou, Le Wu, Yonghui Yang, Kun Zhang, Richang Hong, Meng Wang.
ACM MM 2023, pp. 1004–1012.
PDF: https://le-wu.com/files/Publications/CONFERENCES/MM23-GoRec-bai.pdf

Why chosen: closest match to our setting — item cold-start where new items
have **zero interactions** and must be represented **solely from content**.

### GoRec abstract skeleton (5 moves)

| Move | GoRec's sentence (reconstructed) | Function |
|------|----------------------------------|----------|
| ① Background | "The cold-start problem has been a long-standing issue... embedding-based models perform badly for cold items which haven't emerged in the training set." | one-line problem |
| ② Prior gap | "The most common solutions generate the cold embedding from content. **However**, the cold embeddings... have different distribution as the warm embeddings." | state the gap (distribution mismatch) |
| ③ Name | "We innovatively **break** the alignment-function schema and **propose** a Generative cold Recommendation (GoRec) framework." | announce method, with a strong contrast verb |
| ④ Mechanism | "GoRec directly models the conditional distribution of warm embeddings solely based on content..." | 1–2 sentences of how |
| ⑤ Results | "...extensive experiments demonstrate the effectiveness..." | close |

**Generic top-venue cold-start recipe:**
① Background → ② However + prior limitation → ③ propose + method name → ④ mechanism → ⑤ results.

## Secondary reference (evaluation-protocol critique angle)

**Take a Fresh Look at Recommender Systems from an Evaluation Standpoint**
Aixin Sun. SIGIR 2023, pp. 2629–2638.
https://dl.acm.org/doi/10.1145/3539618.3591931

Why chosen: supports our Move-② framing that standard offline protocols
(sampled candidates, interaction-weighted averages) can overstate progress.

## How our abstract maps to the recipe (current version)

Narrative arc we adopted: **background → education-specific motivation → why
existing methods fall short → what our method does better → evaluation
conditions → results**. The protocol critique is demoted to a subordinate
clause; `strict course-cold` is framed as the *measurement condition*, not the
headline contribution. This avoids the "construct-your-own-problem-then-win"
reading.

| Move | Our sentence | Note |
|------|--------------|------|
| ① Background | "MOOC platforms continually publish new courses that must be recommended before they accumulate any interaction history..." | real, external problem |
| ② Motivation | "...not only a sparsity problem: a useful new course must also be pedagogically appropriate---prerequisites, difficulty, non-redundant." | education-specific |
| ③ Why prior fails | "Existing cold-start recommenders reconstruct item embeddings from content but still inherit warm-item ranking objectives... leave a large fraction of genuinely new courses with near-zero ranking quality... failures that candidate-sampled, interaction-weighted evaluation tends to hide." | protocol critique demoted to a clause here |
| ④ Method | "We propose CKG-RL, which casts cold-start course ranking as a sequential candidate-selection policy trained to maximize an educational reward rather than to fit past clicks..." | contrast: reward vs click-fitting |
| ⑤ Eval conditions | "We evaluate under strict course-cold conditions---zero-interaction targets, full catalog, course-macro Recall/NDCG..." | measurement, not contribution |
| ⑥ Results | "...highest mean in every @5/@10 column, +3.5% to +33.1%; Holm-corrected 11 of 12; the only exception the descriptive +3.5% COCO Recall@10." | honest significance disclosure |

**Consistency rule:** the Introduction must follow the same order as the
abstract — background → motivation → existing methods fail (with concrete
CGRC evidence) → method → strict-cold evaluation conditions → results.
Do NOT let the protocol critique or the benchmark definition lead either one.

Takeaway: same 5–6 move structure as GoRec, but anchored in a real problem
(new courses need exposure + pedagogical fit) rather than a self-defined one,
and stronger on significance testing.

## Source list
- GoRec (MM'23): https://le-wu.com/files/Publications/CONFERENCES/MM23-GoRec-bai.pdf
- Aixin Sun (SIGIR'23): https://dl.acm.org/doi/10.1145/3539618.3591931
- Awesome-Cold-Start-Recommendation: https://github.com/YuanchenBei/Awesome-Cold-Start-Recommendation
