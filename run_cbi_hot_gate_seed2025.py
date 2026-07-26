"""Isolated entrypoint for the TDInit Hot-only gate screening experiment."""

from __future__ import annotations

import fast3_delta.eval as eval_mod
import usim_feedback_fast3_content_delta as protocol

from cbi_hot_gate import CBIHotGateFast3FeedbackUSIM
from cbi_trust_sim import install_trust_eval_adapter


USIM_STATIC_DELEGATE_ENTRYPOINT = True


def install_protocol(protocol_module=protocol, eval_module=eval_mod):
    protocol_module.Fast3FeedbackUSIM = CBIHotGateFast3FeedbackUSIM
    install_trust_eval_adapter(protocol_module, eval_module)


def main():
    install_protocol()
    protocol.main()


if __name__ == "__main__":
    main()
