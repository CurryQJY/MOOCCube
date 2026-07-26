"""Isolated CBI simulator that uses the initial CBI vector as its soft target."""

from __future__ import annotations

from usim_feedback_fast3_content_delta import Fast3FeedbackUSIM


class CBIAnchorFast3FeedbackUSIM(Fast3FeedbackUSIM):
    """Preserve the parent rollout while removing its ID-embedding target."""

    def run_usim_episode(
        self,
        init_item_emb,
        target_emb=None,
        user_bank_raw=None,
        item_idx=None,
        target_pop=None,
        user_seen_items=None,
        deterministic=False,
    ):
        del target_emb
        return super().run_usim_episode(
            init_item_emb,
            target_emb=init_item_emb.detach(),
            user_bank_raw=user_bank_raw,
            item_idx=item_idx,
            target_pop=target_pop,
            user_seen_items=user_seen_items,
            deterministic=deterministic,
        )
