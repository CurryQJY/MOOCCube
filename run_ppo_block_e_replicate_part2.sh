#!/usr/bin/env bash
# Part 2 of the registered E3 replication: seeds 3033/3034, chained behind their
# backbone training. Completes the 5 fresh seeds the prereg requires.
set -u
PY="D:/anaconda3/envs/req_py312/python.exe"
BATCH="outputs/xds_mooccube_ppo_component_ablation"
GRID_WIDE="0.0 0.0025 0.005 0.0075 0.01 0.015 0.02 0.03 0.04 0.05 0.075 0.1 0.15 0.2 0.25"
NULLS="--with-random-policy-arm --with-centroid-step-arm --with-global-shift-arm --with-norm-only-arm"

for i in $(seq 1 180); do
  grep -q "BACKBONES DONE" outputs/_ctrl_logs/backbone_fresh_driver.log 2>/dev/null && break
  sleep 20
done
if ! grep -q "BACKBONES DONE" outputs/_ctrl_logs/backbone_fresh_driver.log 2>/dev/null; then
  echo "[STOP] backbones never reported DONE; refusing to run"; exit 1
fi
for seed in 3033 3034; do
  if [ ! -f "outputs/graph_knp_confirmatory_source/seed${seed}/best.pt" ]; then
    echo "[STOP] seed$seed backbone missing; part 2 incomplete"; exit 1
  fi
done
echo "[chain] backbones ready, starting E3 part 2 at $(date +%H:%M:%S)"

for seed in 3033 3034; do
  out="${BATCH}/E3_noend_only_replicate/seed${seed}"
  log="outputs/_ctrl_logs/ppoabl_E3rep_mooccube_${seed}.log"
  if [ -f "$out/pilot_results.json" ]; then echo "[skip] seed$seed done"; continue; fi
  echo "[run ] E3rep seed$seed at $(date +%H:%M:%S)"
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
    echo "[FAIL] E3rep seed$seed exit=$rc -- see $log"
  fi
done
echo "E3 REPLICATE PART2 DONE at $(date +%H:%M:%S)"
