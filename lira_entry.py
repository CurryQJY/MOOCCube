"""Shared-protocol entry point for the standalone LIRA model."""

import os

import usim_feedback_fast3_content_delta_recovered_51ea_candidate as shared_protocol

from lira.protocol_adapter import LIRAProtocolAdapter


USIM_STATIC_DELEGATE_ENTRYPOINT = True


def main() -> None:
    shared_protocol.Fast3FeedbackUSIM = LIRAProtocolAdapter
    shared_protocol.setup_seed(
        int(os.environ.get("USIM_STATIC_SEED", os.environ.get("USIM_SEED", "2025")))
    )
    shared_protocol.main()


if __name__ == "__main__":
    main()
