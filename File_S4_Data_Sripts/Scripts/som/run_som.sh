#!/usr/bin/env bash
set -uo pipefail

SOM_LRS=(0.5 0.4 0.3 0.2 0.1)
INPUT="bron_concat_matrix_final.csv"
SEED=123

for LR in "${SOM_LRS[@]}"; do
  OUT_DIR="outputs/${LR}"
  LOG="${OUT_DIR}/run.log"
  mkdir -p "$OUT_DIR"

  echo "Running SOM with lr=${LR}"

  if python3 som_delimit.py \
      --input "$INPUT" \
      --outputs "$OUT_DIR" \
      --iters 500 \
      --sigma 1.0 \
      --lr "$LR" \
      --batch_size 8 \
      --mds_dim 5 \
      --k_min 2 \
      --k_max 9 \
      --seed "$SEED" \
      > "$LOG" 2>&1; then
    echo "Run with lr=${LR} succeeded."
  else
    echo "Run with lr=${LR} failed. See ${LOG}"
  fi
done

echo "All SOM runs completed."
