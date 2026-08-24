#!/usr/bin/env bash
# Backbones for fresh seeds 3033/3034, to complete the registered 5-seed
# replication of E3_noend_only. Config copied verbatim from
# outputs/graph_knp_confirmatory_source/seed3030/run_manifest.json:
#   n_layers 2, prereq_aux_weight 2.0, epochs 60, batch 2048, emb 128, lr 0.001,
#   delta_ref 0.25, skip_test true, floor_tol 0.015.
# floor-tol MUST be passed explicitly: the script default is 0.003 (both-lines
# rule v1), while seeds 3030-3032 used 0.015 (cold-first rule v2). Because
# delta* selection sits inside the validation loop, the tolerance also moves
# best_epoch, i.e. the checkpoint weights -- omitting it silently produces a
# different-caliber backbone. First attempt on 3033/3034 did exactly that and
# was moved aside to seed<n>_floortol0.003_DISCARD.
# skip_test is kept so these stay blind, exactly like 3030-3032.
set -u
PY="D:/anaconda3/envs/req_py312/python.exe"
mkdir -p outputs/_ctrl_logs
for seed in 3033 3034; do
  out="outputs/graph_knp_confirmatory_source/seed${seed}"
  if [ -f "$out/best.pt" ]; then echo "[skip] seed$seed backbone exists"; continue; fi
  free_kb=$(df -k . | awk 'NR==2{print $4}')
  if [ "$free_kb" -lt 2000000 ]; then echo "[STOP] only ${free_kb}KB free"; exit 2; fi
  echo "[run ] backbone seed$seed at $(date +%H:%M:%S) free=${free_kb}KB"
  "$PY" graph_knp_consistent.py \
    --seed "$seed" \
    --data-dir processed_data_hin_clean_pop5 \
    --split-dir "outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_${seed}" \
    --output-dir "$out" \
    --epochs 60 --batch-size 2048 --n-layers 2 \
    --prereq-weight 2.0 --delta-ref 0.25 \
    --floor-tol 0.015 \
    --prereq-path outputs/prereq_target/prereq_index_topk10.pt \
    --skip-test \
    > "outputs/graph_knp_confirmatory_source/seed${seed}_run.log" 2>&1
  rc=$?
  if [ -f "$out/best.pt" ]; then
    echo "[done] backbone seed$seed exit=$rc OK at $(date +%H:%M:%S)"
  else
    echo "[FAIL] backbone seed$seed exit=$rc -- see seed${seed}_run.log"
  fi
done
echo "BACKBONES DONE at $(date +%H:%M:%S)"
