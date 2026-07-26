#!/usr/bin/env bash
# 路A纯推理开关对比：加载主表 ckg_rl_full checkpoint（真实路径，绕过符号链接），
# USIM_PREREQ_TARGET=0/1 各跑一次 eval，唯一变量=推理期先修target。
# 用法: bash run_prereq_infer_compare.sh <seed> <prereq 0|1>
set -e
REPO="/d/DeskTop/MOOCCube"
REPO_WIN="D:/DeskTop/MOOCCube"
cd "$REPO"
SEED="${1:-2025}"
PREREQ="${2:-0}"
PY="D:/Anaconda3/envs/zw/python.exe"

MANIFEST="$REPO_WIN/outputs/significance_per_item_exports/mooccube/ckg_rl_full/strict_item_cold_balanced_thr1_seed_${SEED}/static_protocol_manifest.json"
# 真实checkpoint（ckg_rl_full下是符号链接，Python读不了/d/前缀，直接用真实目录）
CKPT="$REPO_WIN/checkpoints/usim_feedback_fast3_content_delta/strict_item_cold_balanced_thr1_seed_${SEED}/finished.pt"
OUTDIR="$REPO_WIN/outputs/prereq_infer_compare/seed${SEED}_prereq${PREREQ}"

export USIM_PREREQ_TARGET="$PREREQ"
export USIM_PREREQ_TARGET_PATH="$REPO_WIN/outputs/prereq_target/prereq_index_topk10.pt"

echo "=== 纯推理 seed${SEED} prereq=${PREREQ} $(date) ==="
echo "ckpt=$CKPT"
"$PY" -u evaluate_cbi_all_refined_seed2025.py \
    --manifest "$MANIFEST" \
    --checkpoint "$CKPT" \
    --output-dir "$OUTDIR" 2>&1 | tail -60
echo "=== done seed${SEED} prereq=${PREREQ} $(date) ==="
