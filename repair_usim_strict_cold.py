"""Validate the repaired strict-cold USIM entrypoints.

The repaired method is intentionally implemented in new entrypoint scripts:
- usim_feedback_fast3_content_delta_repaired.py
- run_usim_feedback_fast3_content_delta_repaired_static.ps1

This helper is read-only. It does not patch the legacy model or runner.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED = [
    ROOT / "usim_feedback_fast3_content_delta_repaired.py",
    ROOT / "run_usim_feedback_fast3_content_delta_repaired_static.ps1",
]


def main() -> int:
    missing = [path.relative_to(ROOT).as_posix() for path in REQUIRED if not path.exists()]
    if missing:
        for item in missing:
            print(f"Missing: {item}")
        return 1

    print("Repaired strict-cold USIM entrypoints are present.")
    print("Use run_usim_feedback_fast3_content_delta_repaired_static.ps1 for repaired static runs.")
    print("Legacy runner/model files are not modified by this helper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
