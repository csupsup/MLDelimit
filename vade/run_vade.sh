#!/usr/bin/env bash
set -uo pipefail

LEARNING_RATES=(0.1 0.01 0.001 0.0001 0.00001)
INPUT="bron_concat_matrix_final.csv"
ID_COL="ID"

for LR in "${LEARNING_RATES[@]}"; do
  OUTPUT_DIR="bron_outputs/${LR}"
  LOG="${OUTPUT_DIR}/run.log"
  mkdir -p "$OUTPUT_DIR"

  echo "Running VaDE with learning_rate=${LR}"

  if python3 vade_delimit.py \
      --input "$INPUT" \
      --id_col "$ID_COL" \
      --latent_dim 7 \
      --hidden_dims 50 50 200 \
      --batch_size 16 \
      --pretrain_epochs 50 \
      --vade_epochs 100 \
      --pretrain_lr "$LR" \
      --vade_lr "$LR" \
      --beta_anneal_epochs 30 \
      --k_min 2 \
      --k_max 9 \
      --plots_dir "$OUTPUT_DIR" \
      --seed 1225 \
      > "$LOG" 2>&1; then
    echo "Run with learning_rate=${LR} succeeded."
  else
    echo "Run with learning_rate=${LR} failed. See ${LOG}"
  fi
done

echo "All VaDE runs completed."
