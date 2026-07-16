# Course-fit pseudo-cold repair snapshot

- Branch: `codex/recppo-research-repair`
- Base commit: `10e0e9aabdc4e408d289d532160caa285bf47a13`
- Original protected snapshot: `backups/main_code_pre_cleanup_20260714_105010/backup_manifest.json`
- Interrupted run checkpoint: `checkpoints/recppo_research_repair/coursefit_pseudocold_minimal_seed2025`
- Interrupted run log: `background_logs/coursefit_pseudocold_minimal_seed2025/train_20260715_213338.stdout.log`

## Current file hashes before second-stage repair

| File | SHA-256 |
|---|---|
| `usim_feedback_fast3_content_delta_recovered_51ea_candidate.py` | `BC978EED6F6CB60927B1D43AE98690DCDFE950480FEBEF3CCEFF667D683470FD` |
| `fast3_delta/config.py` | `689112E234D3C48AEF320B4ECD8041603A12C27FC3CABA50FFF10C28CDBC930A` |
| `run_usim_feedback_fast3_content_delta_static.ps1` | `00DBF3C455DDA8B56711638197BD9061D8F0000CD97B8988086C16DD6A2E867E` |
| `run_coursefit_pseudocold_minimal_seed2025.ps1` | `1DA4E24161F0A62CBFE183A6B8E3CCD9F7C0172BF7CB8D79374A102F9E799665` |

The original main-table outputs and checkpoints are immutable inputs. This snapshot records the exact repaired working state before the second-stage model fixes, so either the original backup or this intermediate state can be reconstructed without overwriting result artifacts.
