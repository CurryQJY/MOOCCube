"""Isolated entrypoint for the CBI soft-anchor simulator experiment."""

from __future__ import annotations

import fast3_delta.eval as eval_mod
import usim_feedback_fast3_content_delta as protocol

from cbi_anchor_sim import CBIAnchorFast3FeedbackUSIM
from cbi_trust_sim import install_trust_eval_adapter


USIM_STATIC_DELEGATE_ENTRYPOINT = True


def install_protocol(protocol_module=protocol, eval_module=eval_mod):
    """Patch only the current process with soft-anchor training and all-item eval."""
    base_resume_decision = protocol_module.checkpoint_resume_decision

    def checkpoint_resume_decision_with_reason(*args, **kwargs):
        decision = base_resume_decision(*args, **kwargs)
        protocol_module.cfg_reason = decision.reason
        return decision

    protocol_module.Fast3FeedbackUSIM = CBIAnchorFast3FeedbackUSIM
    protocol_module.checkpoint_resume_decision = checkpoint_resume_decision_with_reason
    install_trust_eval_adapter(protocol_module, eval_module)


def main():
    install_protocol()
    protocol.main()


if __name__ == "__main__":
    main()
