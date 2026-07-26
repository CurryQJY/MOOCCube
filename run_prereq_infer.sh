set -e
cd /d/DeskTop/MOOCCube
SEED="${1:-2025}"; PREREQ="${2:-0}"
PY="D:/Anaconda3/envs/zw/python.exe"
W="D:/DeskTop/MOOCCube"
MANIFEST="$W/outputs/significance_per_item_exports/mooccube/ckg_rl_full/strict_item_cold_balanced_thr1_seed_${SEED}/static_protocol_manifest.json"
CKPT="$W/checkpoints/recovery_validation/main_table_51ea12fc_candidate/strict_item_cold_balanced_thr1_seed_${SEED}/finished.pt"
OUTDIR="$W/outputs/prereq_infer_compare/seed${SEED}_prereq${PREREQ}"
export USIM_PREREQ_TARGET="$PREREQ"
export USIM_PREREQ_TARGET_PATH="$W/outputs/prereq_target/prereq_index_topk10.pt"
echo "=== seed${SEED} prereq=${PREREQ} | ckpt=candidate ==="
"$PY" -u evaluate_cbi_all_refined_seed2025.py --manifest "$MANIFEST" --checkpoint "$CKPT" --output-dir "$OUTDIR" 2>&1 | tail -50
