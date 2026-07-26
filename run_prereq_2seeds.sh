cd /d/DeskTop/MOOCCube
for S in 2026 2027; do
  for P in 0 1; do
    echo "=== START seed$S prereq=$P $(date) ==="
    bash run_prereq_infer.sh $S $P > outputs/prereq_infer_compare/log_${S}_${P}.txt 2>&1
    echo "=== DONE seed$S prereq=$P $(date) ==="
  done
done
echo "ALL DONE"
