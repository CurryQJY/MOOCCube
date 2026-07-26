#!/usr/bin/env bash
# 纯BERT+gate实验：主表配置基础上关掉所有RL/课程信号，只留 content_proj+gate+ID dropout。
# 目的：验证去掉RL后 Cold 能否达到/接近主表0.2863，决定论文骨架能否简化。
# 用法: bash run_bert_gate_only.sh <seed>
set -e
REPO="/d/DeskTop/MOOCCube"; WREPO="D:/DeskTop/MOOCCube"
cd "$REPO"
SEED="${1:-2025}"
PY="D:/Anaconda3/envs/zw/python.exe"
MANIFEST="$WREPO/outputs/significance_per_item_exports/mooccube/ckg_rl_full/strict_item_cold_balanced_thr1_seed_${SEED}/static_protocol_manifest.json"

# 从主表manifest导出全部env
"$PY" -c "
import json
m=json.load(open(r'$MANIFEST',encoding='utf-8-sig'))
env=m.get('env',{})
with open('.bert_gate_env_${SEED}.sh','w',encoding='utf-8') as f:
    for k,v in env.items():
        if k.startswith('USIM'):
            f.write(f'export {k}=\"{v}\"\n')
"
source ".bert_gate_env_${SEED}.sh"

# === 关掉所有 RL / 课程信号 (纯BERT+gate) ===
export USIM_STEPS=0                    # 关模拟器rollout
export USIM_PPO_LOSS_WEIGHT=0          # 关PPO
export USIM_USE_COURSE_REWARD=0        # 关课程奖励
export USIM_FB_COURSE_SAMPLE_BETA=0    # 关知识引导采样
export USIM_USE_COURSE_FEEDBACK=0
export USIM_USE_COURSE_SAMPLE=0
export USIM_USE_PREREQ_AUX=0
# ID dropout 保留(主表默认0.35), content_proj+gate 保留

EXPROOT="outputs/bert_gate_only/seed${SEED}"
export USIM_N_EPOCHS=60
export USIM_EARLY_STOP_PATIENCE=60
export USIM_FB_OUTPUT_DIR="$WREPO/$EXPROOT/out"
export USIM_FB_CKPT_DIR="$WREPO/$EXPROOT/ckpt"
export USIM_FB_FORCE_FRESH=1
export USIM_FB_AUTO_RESUME=0
export USIM_STATIC_SEED="$SEED"
export USIM_SEED="$SEED"
export USIM_PREREQ_TARGET=0

mkdir -p "$EXPROOT"
echo "=== 纯BERT+gate seed${SEED} (steps0,ppo0,无课程信号) 启动 $(date) ==="
"$PY" -u usim_feedback_fast3_content_delta.py > "$EXPROOT/train.log" 2>&1
echo "=== done seed${SEED} $(date) EXIT=$? ==="
tail -20 "$EXPROOT/train.log"
