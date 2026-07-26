"""Shared-content projector with pseudo-cold training for the Hot-gate audit."""

from cbi_hot_gate_audit_seed2025 import install_protocol


USIM_STATIC_DELEGATE_ENTRYPOINT = True


def main():
    install_protocol()
    import usim_feedback_fast3_content_delta as protocol

    protocol.main()


if __name__ == "__main__":
    main()
