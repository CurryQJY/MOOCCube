#!/usr/bin/env bash
# 路A正式实验: 论文主表配置(ckg_rl_full) + 先修target开关(eval注入), 60 epochs.
# 用法: bash run_prereq_target.sh <seed>
# 对照: ckg_rl_full 同seed (seed2025=0.2732, 2026=0.2868, 2027=0.2988; 3seed均值0.2863)。
# 注入点: infer_refined_item_vectors (eval期冷课表示构造处), 仅作用于冷课, 不污染Hot。
set -e
REPO="/d/DeskTop/MOOCCube"
WREPO="D:/DeskTop/MOOCCube"
cd "$REPO"
SEED="${1:-2025}"
PY="D:/Anaconda3/envs/zw/python.exe"
MANIFEST="$WREPO/outputs/significance_per_item_exports/mooccube/ckg_rl_full/strict_item_cold_balanced_thr1_seed_${SEED}/static_protocol_manifest.json"

# 1. 从论文主表 ckg_rl_full manifest 导出全部 USIM env
"$PY" -c "
import json
m=json.load(open('$MANIFEST',encoding='utf-8'))
env=m.get('env',{})
with open('.prereq_target_env_${SEED}.sh','w',encoding='utf-8') as f:
    for k,v in env.items():
        if k.startswith('USIM'):
            f.write(f'export {k}=\"{v}\"\n')
"
source ".prereq_target_env_${SEED}.sh"

# 2. 覆盖: 独立输出目录 + 全新训练(不resume主表checkpoint)
EXPROOT="outputs/prereq_target_exp_v2/seed${SEED}"
export USIM_N_EPOCHS=60
export USIM_EARLY_STOP_PATIENCE=60
export USIM_FB_OUTPUT_DIR="$(pwd)/$EXPROOT/out"
export USIM_FB_CKPT_DIR="$(pwd)/$EXPROOT/ckpt"
export USIM_FB_FORCE_FRESH=1
export USIM_FB_AUTO_RESUME=0
export USIM_STATIC_SEED="$SEED"
export USIM_SEED="$SEED"

# 3. 开先修target开关 (唯一相对主表的改动; 先修索引checkpoint无关,三seed通用)
export USIM_PREREQ_TARGET=1
export USIM_PREREQ_TARGET_PATH="outputs/prereq_target/prereq_index_topk10.pt"

mkdir -p "$EXPROOT"
echo "=== 路A正式实验 seed${SEED} (60ep, PREREQ_TARGET=1, base=ckg_rl_full) 启动 $(date) ==="
"$PY" -u usim_feedback_fast3_content_delta.py > "$EXPROOT/train.log" 2>&1
echo "=== 完成 seed${SEED} $(date) EXIT=$? ==="
