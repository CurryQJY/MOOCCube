# Abstract Rewrite Notes (AAAI-26 Outstanding Paper Style)

References (AAAI-26 award papers, abstracts via Crossref / AAAI proceedings):

- [LLM2CLIP](https://doi.org/10.1609/aaai.v40i7.37427) — Outstanding Paper
- [ReconVLA](https://doi.org/10.1609/aaai.v40i22.38921) — Outstanding Paper
- [High-Pass Matters](https://doi.org/10.1609/aaai.v40i27.39469) — Outstanding Paper
- [Global Human Opinion alignment](https://doi.org/10.1609/aaai.v40i44.41102) — Best Paper, AI Alignment Track
- Award list: [AAAI conference paper awards](https://aaai.org/about-aaai/aaai-awards/aaai-conference-paper-awards-and-recognition/)

## Shared abstract skeleton in award papers

1. **Paradigm / task in one clean sentence**  
   LLM2CLIP: CLIP is a seminal multimodal model ...  
   ReconVLA: Recent advances in VLA models have enabled ...  
   High-Pass: HGNNs have shown great potential ...

2. **Bottleneck under a harder or neglected condition**  
   LLM2CLIP: long/complex captions  
   ReconVLA: visual attention is dispersed  
   High-Pass: existing HGNNs neglect high-frequency information

3. **Insight or guiding principle** (sometimes theory, sometimes empirical)  
   High-Pass: prove low-pass + high-pass is more expressive  
   ReconVLA: reconstruct gaze region to ground attention

4. **Method as one coherent upgrade**, not a component inventory  
   Name + 1–2 mechanism sentences; avoid four-item laundry lists.

5. **Results that validate the insight**, not only “SOTA everywhere”  
   LLM2CLIP: gains without large-scale retraining + broad transfer  
   High-Pass: superiority + validates high-frequency importance  
   ReconVLA: sim + real-world; precise manipulation and generalization

## What we borrowed for CKG-RL

| Award pattern | Our abstract choice |
|---|---|
| Open with paradigm | New MOOC courses must be ranked before interactions |
| Harder condition | Sampled / micro averages understate difficulty |
| Named setting | *strict course-cold* + full catalog + course-macro |
| Prior limit | Content-transfer leaves educational constraints passive |
| Explicit insight | Knowledge should shape candidate construction and rewards |
| Coherent method | CKG-RL: content anchor + ID mask + knowledge-guided simulation |
| Evidence | Best @5/@10 means, +3.5%–+33.1%, 11/12 Holm, ablation drivers |

## What we deliberately did *not* copy

- Over-long downstream laundry lists (LLM2CLIP can afford them; we have one main protocol).
- “Extensive experiments demonstrate superiority” with zero numbers (we keep the key numbers because recsys comparisons need them).
- Theory-first opening (our paper is empirical protocol + method; High-Pass style would overclaim).

## Claim–evidence map (revised abstract)

| Claim | Evidence | Status |
|---|---|---|
| Sampled/micro can hide cold failure | Protocol section + motivation figure | supported |
| Content-transfer leaves educational constraints passive | Related work + method motivation | supported (positioning) |
| Knowledge should enter sampling/rewards | Method design + RQ3 ablation | supported |
| Best early-rank course-macro @5/@10 on 3 datasets | Main table | supported |
| +3.5% to +33.1% vs strongest non-CKG | Imp. row | supported |
| 11/12 Holm-corrected significance | Table note + RQ1 text | supported |
| Gains from rewards / sampling / pre-aux | RQ3 | supported |
