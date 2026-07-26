#!/usr/bin/env bash
# 路A正式实验(单seed验证): 主表配置 + 先修target开关, 60 epochs, seed 2025.
# 对照: 主表 full_reference seed2025 (Cold R@10=0.2863 im / 0.2978 um)。
# 输出到独立目录, 不污染主表。
set -e
cd "D:/DeskTop/MOOCCube"
PY="D:/Anaconda3/envs/zw/python.exe"
MANIFEST="outputs/significance_per_item_exports/mooccube/aaai27_rl_component_gate_v1/full_reference/strict_item_cold_balanced_thr1_seed_2025/static_protocol_manifest.json"

# 1. 从主表 manifest 导出全部 USIM env (完整复制主表配置)
"$PY" -c "
import json
m=json.load(open('$MANIFEST',encoding='utf-8'))
env=m.get('env',{})
with open('.prereq_target_env.sh','w',encoding='utf-8') as f:
    for k,v in env.items():
        if k.startswith('USIM'):
            f.write(f'export {k}=\"{v}\"\n')
"
source .prereq_target_env.sh

# 2. 覆盖: 输出到独立实验目录 + 全新训练(不resume主表checkpoint)
EXPROOT="outputs/prereq_target_exp/seed2025"
export USIM_N_EPOCHS=60
export USIM_EARLY_STOP_PATIENCE=60
export USIM_FB_OUTPUT_DIR="$(pwd)/$EXPROOT/out"
export USIM_FB_CKPT_DIR="$(pwd)/$EXPROOT/ckpt"
export USIM_FB_FORCE_FRESH=1
export USIM_FB_AUTO_RESUME=0
export USIM_STATIC_SEED=2025
export USIM_SEED=2025

# 3. 开先修target开关 (唯一相对主表的改动)
export USIM_PREREQ_TARGET=1
export USIM_PREREQ_TARGET_PATH="outputs/prereq_target/prereq_index_topk10.pt"

mkdir -p "$EXPROOT"
echo "=== 路A正式实验 seed2025 (60ep, PREREQ_TARGET=1) 启动 $(date) ==="
"$PY" -u usim_feedback_fast3_content_delta.py > "$EXPROOT/train.log" 2>&1
echo "=== 完成 $(date) EXIT=$? ==="
