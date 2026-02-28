#!/bin/bash

LOG_DIR="${1:-token_accuracy_logs}"
mkdir -p "$LOG_DIR"

CONFIGS=(
  sft_nondistributed_1gpu
  sft_dtensor_tp2_8gpu
  sft_megatron_tp2_8gpu
  sft_megatron_cp2_8gpu
  sft_megatron_tp2_cp2_8gpu
  sft_megatron_tp2_seqpack_8gpu
  sft_megatron_tp1_openmathinstruct_8gpu
)

PID=""
trap 'echo "Interrupted. Killing current run..."; [ -n "$PID" ] && kill $PID 2>/dev/null; exit 1' INT TERM

for cfg in "${CONFIGS[@]}"; do
  echo "=== $(date): Starting $cfg ==="
  nohup docker exec nemo_rl_dev bash -c \
    "cd /workspace && NRL_FORCE_REBUILD_VENVS=true uv run python examples/run_sft.py \
     --config examples/configs/token_accuracy_experiments/${cfg}.yaml" \
    > "$LOG_DIR/${cfg}.log" 2>&1 &
  PID=$!
  wait $PID
  STATUS=$?
  echo "=== $(date): Finished $cfg (exit code: $STATUS) ==="
  echo ""
done

echo "All runs complete. Logs in $LOG_DIR/"
