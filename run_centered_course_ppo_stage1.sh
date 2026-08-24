#!/usr/bin/env bash
# Stage 1 of the centered-course-reward PPO matrix.
# Pre-registration: docs/superpowers/specs/2026-08-24-prereg-centered-course-ppo-main-model.md
#
# Two objective defects measured today, never varied in any prior batch:
#   D1 the course reward is a near-constant penalty (-1.75e-2, ~15x the two
#      positive terms combined, only 9% epoch-to-epoch variation) because the
#      largest weight sits on prereq_gap and a strict-cold course's candidate
#      users have seen ~none of its prerequisites. Fix: --course-reward-mode
#      centered, which subtracts the candidate-set mean so only the
#      "which candidate is better" part survives. Audited: 153/153 manifests
#      record absolute, zero record centered. This is its first run.
#   D2 entropy_weight 0.01 x ln(20) = 0.030 dwarfs the ~1.1e-3 reward signal 27x,
#      so the objective's optimum is the uniform policy == ridge_random_policy.
#      Fix: --ppo-entropy-weight.
#
# Bar (registered): the PPO arm must beat ALL THREE zero-training nulls on
# >= 4/5 seeds. Beating random_policy alone does not qualify --
# greedy_course_fit consumes the same course signal with zero training.
#
# Stage 1 is seeds 2025-2029 only. Blind seeds 3030-3034 are RESERVED for
# stage 2 and must not be touched here.
set -u
PY="D:/anaconda3/envs/req_py312/python.exe"
mkdir -p outputs/_ctrl_logs
BATCH="outputs/xds_mooccube_centered_course_ppo"
GRID="0.0 0.0025 0.005 0.0075 0.01 0.015 0.02 0.03 0.04 0.05 0.075 0.1 0.15 0.2 0.25"
NULLS="--with-random-policy-arm --with-centroid-step-arm --with-global-shift-arm --with-norm-only-arm"

run_one() {
  local tag="$1"; shift
  local seed="$1"; shift
  local out="${BATCH}/${tag}/seed${seed}"
  local log="outputs/_ctrl_logs/ccppo_${tag}_mooccube_${seed}.log"
  if [ -f "$out/pilot_results.json" ]; then
    echo "[skip] $tag seed$seed already complete"; return 0
  fi
  local free_kb
  free_kb=$(df -k . | awk 'NR==2{print $4}')
  if [ "$free_kb" -lt 2000000 ]; then
    echo "[STOP] only ${free_kb} KB free; aborting before $tag seed$seed"; return 2
  fi
  echo "[run ] $tag seed$seed at $(date +%H:%M:%S) free=${free_kb}KB extra: $*"
  "$PY" ridge_course_reward_rl_pilot.py \
    --seed "$seed" \
    --data-dir processed_data_hin_clean_pop5 \
    --split-root outputs/content_delta_pop5/static_item_cold_balanced \
    --ckpt-root outputs/graph_knp_final \
    --course-relation-dir MOOCCube/relations \
    --ppo-arms ridge_ppo_course_reward_only \
    --reward-geometry cosine \
    --ridge-alpha 1.0 --retention-reference ridge \
    --candidate-count 20 \
    --course-bias-scale 0.2 \
    --course-concept-weight 0.04 --course-prereq-weight 0.08 \
    --course-difficulty-weight 0.03 --course-redundant-weight 0.0 \
    --course-reward-scale 0.5 \
    --policy-epochs 5 --policy-batch-size 8 --policy-lr 0.0003 \
    --eval-batch-users 512 --retrieval-chunk 8192 \
    --replay-capacity 8192 --replay-batch-size 512 \
    --max-pseudo-pop 25.0 --pseudo-val-fraction 0.2 \
    --hot-tolerance 0.003 \
    --delta-grid $GRID --max-steps 5 --step-size 0.05 \
    --step-penalty 0.0 --no-end-action \
    $NULLS \
    "$@" \
    --out "$out" > "$log" 2>&1
  local rc=$?
  if [ -f "$out/pilot_results.json" ]; then
    rm -f "$out"/*_selected_eval.pt
    echo "[done] $tag seed$seed exit=$rc OK at $(date +%H:%M:%S) (eval bundle pruned)"
  else
    echo "[FAIL] $tag seed$seed exit=$rc NO OUTPUT -- see $log"
  fi
  return 0
}

# F0/F1 seed2025 first: they are the mechanism gate. If centered does not move
# the logged training-time `course=` term away from -1.75e-2, the flag is inert
# and the rest of the matrix would measure nothing.
run_one F0_absolute_ref 2025 --course-reward-mode absolute --ppo-entropy-weight 0.01
run_one F1_centered     2025 --course-reward-mode centered --ppo-entropy-weight 0.01
echo "[gate] does centering move the training-time course term?"
for t in F0_absolute_ref F1_centered; do
  printf '  %-18s ' "$t"
  grep -oE "course=[-0-9.]+" "outputs/_ctrl_logs/ccppo_${t}_mooccube_2025.log" | tail -1
done

for seed in 2026 2027 2028 2029; do
  run_one F0_absolute_ref "$seed" --course-reward-mode absolute --ppo-entropy-weight 0.01 || break
done
for seed in 2026 2027 2028 2029; do
  run_one F1_centered "$seed" --course-reward-mode centered --ppo-entropy-weight 0.01 || break
done
for seed in 2025 2026 2027 2028 2029; do
  run_one F2_centered_ent1e3 "$seed" --course-reward-mode centered --ppo-entropy-weight 0.001 || break
done
for seed in 2025 2026 2027 2028 2029; do
  run_one F3_centered_ent1e4 "$seed" --course-reward-mode centered --ppo-entropy-weight 0.0001 || break
done
# F4 isolates D2 from D1: if F4 == F3, centering was not what mattered.
for seed in 2025 2026 2027 2028 2029; do
  run_one F4_absolute_ent1e4 "$seed" --course-reward-mode absolute --ppo-entropy-weight 0.0001 || break
done
echo "CENTERED STAGE1 DONE at $(date +%H:%M:%S)"
