# LightGCL Source Snapshot

- Repository: https://github.com/HKUDS/LightGCL
- Paper: LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation (ICLR 2023)
- Source commit: 5590453ad86782f58017e58d0b698d7f32175be3
- Pulled on: 2026-05-14

Only the official core source files are mirrored locally because full `git clone`
timed out in this environment:

- `README.md`
- `main.py`
- `model.py`
- `parser.py`
- `utils.py`

The local static item-cold protocol is implemented outside this source tree in
`lightgcl_static_hin_fair.py`. The official files are kept as a reference
snapshot and are not modified by the adapter.
