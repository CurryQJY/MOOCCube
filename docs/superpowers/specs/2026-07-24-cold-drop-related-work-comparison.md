# Cold-Drop Related Work Comparison (2025–2026)

**Status:** analysis note only; no implementation or experiment is authorized by this document.
**Purpose:** situate the observed cold-drop after graph propagation (see [[graph-propagation-result]]) against two recent (2025–2026) cold-start papers, and assess whether the magnitude-scaling guardrail hypothesis has external support.
**Scope:** 2025–2026 venue papers only. The 2024 CGRC-adjacent work is out of scope for this note.

## Evidence boundary (read first)

Sources actually consulted for this note, on 2026-07-24, via `curl` against DBLP / arXiv / Semantic Scholar (the WebSearch/WebFetch tools were non-functional this session; direct API access worked):

- **Inherited Popularity Bias**: DBLP metadata + arXiv full abstract (`arXiv:2510.11402`, open access). Mechanism claims below are quoted/paraphrased from the abstract. Full-text method and experiment details were **not** read.
- **G-TRAC**: DBLP metadata + Semantic Scholar abstract (via DOI). This is a **5-page WSDM short paper, closed access**; no arXiv preprint exists. Only the abstract-level framing is confirmed. Cold/hot trade-off behavior is **not** described in any source I could reach — do not attribute one to it.

Anything not covered by the above is marked `[TODO: verify against source]`.

## Paper 1 — On Inherited Popularity Bias in Cold-Start Item Recommendation

- **Authors:** Gregor Meehan, Johan Pauwels
- **Venue:** RecSys 2025, pp. 649–654. DOI `10.1145/3705328.3748035`. Open preprint: `arXiv:2510.11402`.

### Mechanism (from abstract)

- Cold-start models trained with supervision from warm CF models learn to replicate CF behavior — and therefore also **inherit CF predictive biases**, specifically popularity bias.
- Cold-start recommenders are affected **more severely** than the warm models they imitate: lacking interaction data, they estimate popularity **solely from content features**.
- Consequence: **systematic over-prediction of cold items whose content resembles popular warm items**, even when true popularity is very low.
- Evaluated on three multimedia datasets across three generative cold-start methods.
- **Mitigation:** a simple post-processing step using **embedding magnitude as a proxy for predicted popularity** to rebalance recommendations, with limited harm to user-oriented cold-start accuracy.

### Relevance to our cold-drop

This is the closest external result to our situation and it directly supports the magnitude-scaling guardrail hypothesis:

1. Our pipeline supervises the cold path against interaction-trained (warm) signal, so the "inherited bias" precondition applies in principle.
2. Their headline finding — cold scores track **content-induced popularity** rather than true cold relevance — is a candidate explanation for why adding LightGCN structure (a warm-derived signal) rebalanced score mass toward hot items and depressed cold (see [[graph-propagation-result]]).
3. Their mitigation (magnitude-as-popularity-proxy, post-hoc) is essentially the guardrail we were considering prototyping. This is independent RecSys'25 evidence that the premise is sound and that a post-processing form is viable without retraining.

### What this does NOT establish

- It does not confirm our cold-drop is magnitude-driven; it makes the hypothesis plausible, not verified. A direct check on our own embeddings is still required.
- Their setting is multimedia CF + generative cold-start, not graph-propagated course ranking. Transfer is by analogy, not by identical setup.
- `[TODO: verify against source]` whether their magnitude proxy is computed pre- or post-graph, and how it interacts with a structural (GNN) score term — the abstract does not say, and this matters for where we would insert a guardrail relative to LightGCN.

## Paper 2 — G-TRAC: Graph-textual Representations Alignment for Cold-start Recommendations

- **Authors:** Li-Yang Chang, Yuan Fang, Ming-Feng Tsai, Chuan-Ju Wang
- **Venue:** WSDM 2026, pp. 1089–1093 (short paper, closed access). DOI `10.1145/3773966.3779358`.

### Framing (from abstract)

- Integrates **transformer-based textual modeling with graph neural networks** to align textual and structural representations for cold-start.
- Claims to leverage both textual and structural information more effectively than GNN-only methods, with experiments reporting improved recommendation quality and generalization.

### Relevance to our cold-drop

- Topically adjacent: it is the graph+text alignment regime we entered when adding LightGCN on top of content embeddings. It is a reasonable **related-work citation** for the "combine structural and content signal for cold items" framing.
- However, the abstract gives **no mechanism for managing the cold↔hot trade-off**, which is the specific failure we are diagnosing. As a 5-page short paper with no reachable full text, it cannot currently be used to justify or design a guardrail.

### What this does NOT establish

- `[TODO: verify against source]` its alignment objective, whether it reports a cold/hot/overall breakdown, and whether it observes or addresses any cold regression. None of this is in the abstract.
- Do not cite it as support for the magnitude hypothesis — different mechanism, no relevant evidence.

## Bottom line

- **Inherited Popularity Bias (RecSys'25)** is the load-bearing reference: it independently motivates the magnitude-scaling guardrail and matches the direction of our cold-drop. Next concrete step remains the empirical check on our own scorer (option (b) from prior discussion) — this note strengthens the case for it but is not a substitute.
- **G-TRAC (WSDM'26)** is a relevant related-work citation for the graph+text cold-start regime, but carries no evidence about the cold/hot trade-off and cannot inform guardrail design at the abstract level.
- Before either is cited in the paper: obtain and read full text. Inherited Popularity Bias full text is freely available (`arXiv:2510.11402`); G-TRAC requires ACM DL access.
