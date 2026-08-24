# Strict Stage-B Component Campaign

## Objective

Screen previously implemented but inactive behavior without changing the main-table
code or the frozen-Hot validation protocol. A component may enter the incumbent
only after it improves strict-Cold validation while preserving the immutable Hot
and Overall retention floors.

## Fixed Contract

Every run reads only `meta.json`, `content_emb.pt`, `static_train.pkl`, and
`static_val.pkl` for a registered strict seed. It binds the completed
seed-specific frozen-Hot checkpoint contract, never loads a test or stream file,
and writes fresh output/checkpoint/log roots. The Hot expert remains frozen.

The immutable guard is the associated Hot checkpoint's epoch-zero validation
row: Hot R/N@10 and Overall R/N@10 must each remain at least `baseline - .003`.
At selection, eligible rows maximize Cold N@10, then Cold R@10, then epoch.

## Campaign Semantics

The first screen uses strict seed 2027. A candidate is provisionally accepted
only when its selected row passes the immutable guard, raises the incumbent Cold
N@10 by at least `.003`, and does not lower incumbent Cold R@10. A provisionally
accepted candidate is then replicated on strict seed 2026. Final acceptance
requires both strict seeds to pass the immutable guard, a mean Cold N@10 gain
of at least `.003` versus the same incumbent, and no seed-level Cold R@10
regression. Otherwise it is rejected and the incumbent remains unchanged.

The canonical ledger is append-only JSONL and CSV. A separately named JSON
snapshot is regenerated for convenient inspection. Every run record includes
the parent incumbent path/configuration, strict seed, selected epoch, all
Cold/Hot/Overall R@10/N@10 values and deltas, guard state, source/checkpoint
hash bindings, elapsed time, and decision. A final-incumbent artifact records
whether the screened component was promoted.

## Candidate Inventory

The baseline already includes shared content projection, initial-content
anchoring, hard spherical trust projection, frozen Hot routing, and deterministic
pseudo-cold training. These are recorded as already-enabled rather than screened
again.

The first runnable addition is the existing content-delta L2 regularizer,
ported as a final-space soft content-anchor loss for selected pseudo-cold items.
It has `weight=.10`, normalizes by the fixed `tau=.24929234`, and is additive to
the ranking objective. It does not introduce an item-ID target.

The old CBI cone, Hot ID/content gate, legacy simulator, PPO, and course reward
are not legal one-component additions to this protocol: they are respectively
redundant, mutate the frozen Hot bank, or couple to the legacy Fast3 model and
its item-ID/reward machinery. They are ledgered as protocol-rejected rather
than executed.

## Failure Handling

No per-component output or checkpoint root is created before input and
Hot-contract checks succeed. A failed job writes a terminal ledger row when an
output result is available; a launch or contract failure is recorded as `failed`
and does not promote a candidate. The campaign never evaluates test data and
never mutates the baseline artifacts.
