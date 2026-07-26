#!/usr/bin/env bash
# 路A烟雾测试：主表配置 + 先修target开关，epochs=1，输出到tmp，验证端到端不崩。
set -e
PY="D:/Anaconda3/envs/zw/python.exe"
MANIFEST="outputs/significance_per_item_exports/mooccube/aaai27_rl_component_gate_v1/full_reference/strict_item_cold_balanced_thr1_seed_2025/static_protocol_manifest.json"

# 1. 从 manifest 导出全部 env
"$PY" -c "
import json,os
m=json.load(open('$MANIFEST',encoding='utf-8'))
env=m.get('env',{})
with open('.prereq_smoke_env.sh','w',encoding='utf-8') as f:
    for k,v in env.items():
        if k.startswith('USIM'):
            f.write(f'export {k}=\"{v}\"\n')
"
source .prereq_smoke_env.sh

# 2. 覆盖：烟雾配置
export USIM_N_EPOCHS=1
export USIM_EARLY_STOP_PATIENCE=1
export TMPOUT="outputs/_prereq_smoke_tmp"
export USIM_FB_OUTPUT_DIR="$(pwd)/$TMPOUT/out"
export USIM_FB_CKPT_DIR="$(pwd)/$TMPOUT/ckpt"
export USIM_FB_FORCE_FRESH=1
export USIM_FB_AUTO_RESUME=0
export USIM_STATIC_SEED=2025
export USIM_SEED=2025
# 3. 开先修开关
export USIM_PREREQ_TARGET=1
export USIM_PREREQ_TARGET_PATH="outputs/prereq_target/prereq_index_topk10.pt"

mkdir -p "$TMPOUT"
echo "=== 烟雾测试启动 (epochs=1, PREREQ_TARGET=1) ==="
"$PY" -u usim_feedback_fast3_content_delta.py 2>&1 | tail -40
