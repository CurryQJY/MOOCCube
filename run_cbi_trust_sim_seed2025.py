"""Isolated entrypoint for the CBI-constrained simulator experiment."""

from __future__ import annotations

import os

import fast3_delta.eval as eval_mod
import usim_feedback_fast3_content_delta as protocol

from cbi_trust_sim import CBITrustFast3FeedbackUSIM, install_trust_eval_adapter


USIM_STATIC_DELEGATE_ENTRYPOINT = True


def install_protocol(protocol_module=protocol, eval_module=eval_mod):
    """Patch only the current process with trust-sim model/config/evaluation."""
    base_config = protocol_module.Fast3Config
    base_resume_decision = protocol_module.checkpoint_resume_decision

    def checkpoint_resume_decision_with_reason(*args, **kwargs):
        decision = base_resume_decision(*args, **kwargs)
        protocol_module.cfg_reason = decision.reason
        return decision

    class CBITrustFast3Config(base_config):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cbi_trust_cosine_floor = float(
                os.environ.get("USIM_CBI_TRUST_COSINE_FLOOR", str((0.75) ** 0.5))
            )

    CBITrustFast3Config.__name__ = "CBITrustFast3Config"
    protocol_module.Fast3Config = CBITrustFast3Config
    protocol_module.Fast3FeedbackUSIM = CBITrustFast3FeedbackUSIM
    protocol_module.checkpoint_resume_decision = checkpoint_resume_decision_with_reason
    install_trust_eval_adapter(protocol_module, eval_module)
    return CBITrustFast3Config


def main():
    install_protocol()
    protocol.main()


if __name__ == "__main__":
    main()
