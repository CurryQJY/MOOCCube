#!/usr/bin/env bash
# Registered replication of the one config that passed criterion 1.
# See prereg "Block E result, and the replication it triggers": E3_noend_only is
# re-run at 5 FRESH seeds because its sd is 2.0x its mean, the same shape that
# collapsed the earlier ppo-vs-random claim from 5/5 to 2/5.
# Confirmed only at >= 4/5 fresh seeds. No third seed set.
set -u
PY="D:/anaconda3/envs/req_py312/python.exe"
mkdir -p outputs/_ctrl_logs
BATCH="outputs/xds_mooccube_ppo_component_ablation"
GRID_WIDE="0.0 0.0025 0.005 0.0075 0.01 0.015 0.02 0.03 0.04 0.05 0.075 0.1 0.15 0.2 0.25"
NULLS="--with-random-policy-arm --with-centroid-step-arm --with-global-shift-arm --with-norm-only-arm"

for seed in 3030 3031 3032 3033 3034; do
  out="${BATCH}/E3_noend_only_replicate/seed${seed}"
  log="outputs/_ctrl_logs/ppoabl_E3rep_mooccube_${seed}.log"
  if [ -f "$out/pilot_results.json" ]; then echo "[skip] seed$seed done"; continue; fi
  free_kb=$(df -k . | awk 'NR==2{print $4}')
  if [ "$free_kb" -lt 2000000 ]; then echo "[STOP] only ${free_kb}KB free"; exit 2; fi
  echo "[run ] E3rep seed$seed at $(date +%H:%M:%S) free=${free_kb}KB"
  "$PY" ridge_course_reward_rl_pilot.py \
    --seed "$seed" \
    --data-dir processed_data_hin_clean_pop5 \
    --split-root outputs/content_delta_pop5/static_item_cold_balanced \
    --ckpt-root outputs/graph_knp_confirmatory_source \
    --course-relation-dir MOOCCube/relations \
    --ppo-arms ridge_ppo_core \
    --reward-geometry cosine \
    --ridge-alpha 1.0 --retention-reference ridge \
    --candidate-count 20 \
    --course-bias-scale 0.2 \
    --course-concept-weight 0.04 --course-prereq-weight 0.08 \
    --course-difficulty-weight 0.03 --course-redundant-weight 0.0 \
    --course-reward-scale 0.5 --course-reward-mode absolute \
    --policy-epochs 5 --policy-batch-size 8 --policy-lr 0.0003 \
    --eval-batch-users 512 --retrieval-chunk 8192 \
    --replay-capacity 8192 --replay-batch-size 512 \
    --max-pseudo-pop 25.0 --pseudo-val-fraction 0.2 \
    --hot-tolerance 0.003 \
    $NULLS \
    --delta-grid $GRID_WIDE --max-steps 5 --step-size 0.05 \
    --step-penalty 0.0 --no-end-action \
    --out "$out" > "$log" 2>&1
  rc=$?
  if [ -f "$out/pilot_results.json" ]; then
    rm -f "$out"/*_selected_eval.pt
    echo "[done] E3rep seed$seed exit=$rc OK at $(date +%H:%M:%S)"
  else
    echo "[FAIL] E3rep seed$seed exit=$rc NO OUTPUT -- see $log"
  fi
done
echo "E3 REPLICATE DONE at $(date +%H:%M:%S)"
