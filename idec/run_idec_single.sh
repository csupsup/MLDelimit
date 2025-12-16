#!/usr/bin/env bash
set -uo pipefail 

LEARNING_RATES=(0.1 0.01 0.001 0.0001 0.00001)
INPUT="bron_concat_matrix_final.csv"

for LR in "${LEARNING_RATES[@]}"; do
  OUT_DIR="outputs/${LR}"
  PLOT_DIR="plots/${LR}"
  LOG="${OUT_DIR}/run.log"
  mkdir -p "$OUT_DIR" "$PLOT_DIR"

  echo "Running iDEC-single with learning_rate=${LR}"

  if python3 idec_single_delimit.py \
      --input "$INPUT" \
      --plots_dir "$PLOT_DIR" \
      --outputs_dir "$OUT_DIR" \
      --latent 8 \
      --n_iter 50 \
      --update_interval 10 \
      --gamma 0.1 \
      --alpha 1.0 \
      --learning_rate "$LR" \
      --batch_size 16 \
      --k_min 2 \
      --k_max 5 \
      --wrap_width 30 \
      > "$LOG" 2>&1; then
    echo "Run with learning_rate=${LR} succeeded."
  else
    echo "Run with learning_rate=${LR} failed. See ${LOG}"
  fi
done

echo "All iDEC-single runs completed."
