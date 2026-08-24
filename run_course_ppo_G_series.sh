#!/usr/bin/env bash
# G series: the last mechanical levers for course PPO.
# Pre-registration: docs/superpowers/specs/2026-08-24-prereg-course-ppo-rescue-G-series.md
#
# Reference is the existing F2 (centered, entropy 1e-3, scale 0.5). Not re-run.
#
# Bar (unchanged from predecessors): the PPO arm must beat ALL THREE zero-training
# nulls -- random_policy, centroid_step, greedy_course_fit -- on >= 4/5 seeds.
# Beating centroid_step alone does not qualify.
#
# Seeds 3035-3039 are RESERVED for confirmation and must not appear here.
# Blind seeds 3030-3034 are spent on the E3 replication.
set -u
PY="D:/anaconda3/envs/req_py312/python.exe"
mkdir -p outputs/_ctrl_logs
BATCH="outputs/xds_mooccube_course_ppo_G"
GRID="0.0 0.0025 0.005 0.0075 0.01 0.015 0.02 0.03 0.04 0.05 0.075 0.1 0.15 0.2 0.25"
# G3 only: the rollout produces |d| ~ 0.02 once step_size is 0.004, so a grid
# reaching 0.25 would be dead weight. This does shrink the delta budget from 15
# to 7 candidates, which is inherent to budget alignment -- E1 had the same
# property -- and is recorded rather than corrected.
GRID_ALIGNED="0.0 0.0025 0.005 0.0075 0.01 0.015 0.02"
NULLS="--with-random-policy-arm --with-centroid-step-arm --with-global-shift-arm --with-norm-only-arm"

# --- G0 blocking gate: the two new flags must be bit-neutral on the default path.
NEUTRAL="outputs/_neutrality_check_F2_seed2025/pilot_results.json"
REF="outputs/xds_mooccube_centered_course_ppo/F2_centered_ent1e3/seed2025/pilot_results.json"
if [ ! -f "$NEUTRAL" ]; then
  echo "[STOP] G0 gate artifact missing: $NEUTRAL -- run the neutrality check first"; exit 1
fi
"$PY" - "$NEUTRAL" "$REF" <<'PYGATE'
import json,sys
# Gate on the primary metrics only, which is what the prereg registered
# ("max |diff| on cold_N@10"), widened here to all ten primaries.
# Three derived fields are excluded because they were measured to be
# nondeterministic run-to-run under UNCHANGED code: two identical re-runs
# differ by matched_hot_cold_vs_ridge_bias 8.95e-08, cold_N@20 6.21e-09,
# overall_N@20 6.57e-10. Gating on them is impossible, not lenient.
PRIMARY=("cold_R@5","cold_R@10","cold_R@20","cold_N@5","cold_N@10",
         "hot_R@5","hot_R@10","hot_N@5","hot_N@10","overall_R@10","overall_N@10")
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
mx=0.0;n=0;other=0.0
for split in ("validation","test"):
    for arm in b[split]:
        for k,v in b[split][arm].items():
            if not isinstance(v,(int,float)): continue
            if arm not in a[split] or k not in a[split][arm]: continue
            d=abs(float(a[split][arm][k])-float(v))
            if k in PRIMARY: mx=max(mx,d); n+=1
            else: other=max(other,d)
print(f"[G0] {n} primary-metric cells compared, max|diff|={mx:.3e}, threshold 1e-12")
print(f"[G0] non-primary max|diff|={other:.3e} (known nondeterministic, not gated)")
sys.exit(0 if mx<1e-12 else 3)
PYGATE
if [ $? -ne 0 ]; then
  echo "[STOP] G0 gate FAILED -- the new flags are not bit-neutral; do not interpret any G run"; exit 3
fi
echo "[G0] PASS -- proceeding"
echo "[sha] ridge_course_reward_rl_pilot.py $(sha256sum ridge_course_reward_rl_pilot.py | cut -c1-16)"
echo "[sha] ckg_rl_usim_v32_clean.py       $(sha256sum ckg_rl_usim_v32_clean.py | cut -c1-16)"

run_one() {
  local tag="$1"; shift
  local seed="$1"; shift
  local grid="$1"; shift
  local out="${BATCH}/${tag}/seed${seed}"
  local log="outputs/_ctrl_logs/gppo_${tag}_mooccube_${seed}.log"
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
    --course-reward-mode centered --ppo-entropy-weight 0.001 \
    --policy-epochs 5 --policy-batch-size 8 --policy-lr 0.0003 \
    --eval-batch-users 512 --retrieval-chunk 8192 \
    --replay-capacity 8192 --replay-batch-size 512 \
    --max-pseudo-pop 25.0 --pseudo-val-fraction 0.2 \
    --hot-tolerance 0.003 \
    --delta-grid $grid \
    $NULLS \
    "$@" \
    --out "$out" > "$log" 2>&1
  local rc=$?
  if [ -f "$out/pilot_results.json" ]; then
    rm -f "$out"/*_selected_eval.pt
    echo "[done] $tag seed$seed exit=$rc OK at $(date +%H:%M:%S)"
  else
    echo "[FAIL] $tag seed$seed exit=$rc NO OUTPUT -- see $log"
  fi
  return 0
}

# G1/G2 seed2025 first as a mechanism probe: raising the scale must move the
# logged training-time `course=` term roughly proportionally. If it does not,
# the scale flag is not reaching the reward and the rest would measure nothing.
run_one G1_scale2 2025 "$GRID" --course-reward-scale 2.0 --max-steps 5 --step-size 0.05 --step-penalty 0.0 --no-end-action
run_one G2_scale8 2025 "$GRID" --course-reward-scale 8.0 --max-steps 5 --step-size 0.05 --step-penalty 0.0 --no-end-action
echo "[probe] training-time course term vs scale (seed2025, last epoch):"
printf '  %-14s ' "F2 scale0.5"; grep -oE "course=[-0-9.]+" outputs/_ctrl_logs/ccppo_F2_centered_ent1e3_mooccube_2025.log | tail -1
for t in G1_scale2 G2_scale8; do
  printf '  %-14s ' "$t"; grep -oE "course=[-0-9.]+" "outputs/_ctrl_logs/gppo_${t}_mooccube_2025.log" | tail -1
done

for seed in 2026 2027 2028 2029; do
  run_one G1_scale2 "$seed" "$GRID" --course-reward-scale 2.0 --max-steps 5 --step-size 0.05 --step-penalty 0.0 --no-end-action || break
done
for seed in 2026 2027 2028 2029; do
  run_one G2_scale8 "$seed" "$GRID" --course-reward-scale 8.0 --max-steps 5 --step-size 0.05 --step-penalty 0.0 --no-end-action || break
done
for seed in 2025 2026 2027 2028 2029; do
  run_one G3_aligned "$seed" "$GRID_ALIGNED" --course-reward-scale 0.5 --max-steps 5 --step-size 0.004 --step-penalty 0.0 --no-end-action || break
done
for seed in 2025 2026 2027 2028 2029; do
  run_one G4_noepochsel "$seed" "$GRID" --course-reward-scale 0.5 --max-steps 5 --step-size 0.05 --step-penalty 0.0 --no-end-action --no-epoch-selection || break
done
echo "G SERIES DONE at $(date +%H:%M:%S)"
