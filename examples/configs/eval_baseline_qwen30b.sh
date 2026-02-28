#!/bin/bash
# Evaluate the original Qwen3-30B-A3B (instruct) model with thinking disabled
# Server starts once; all datasets evaluated sequentially

MODEL_NAME="Qwen/Qwen3-30B-A3B"

mkdir -p eval_logs

echo "========================================================"
echo "[$(date)] Starting evaluation across all datasets..."
echo "========================================================"

NRL_FORCE_REBUILD_VENVS=true uv run examples/run_eval.py \
    --config=examples/configs/evals/eval.yaml \
    --datasets=gpqa_diamond,math500,aime2024,aime2025 \
    generation.model_name=${MODEL_NAME} \
    generation.vllm_cfg.max_model_len=16384 \
    generation.max_new_tokens=8192 \
    tokenizer.chat_template_kwargs.enable_thinking=false \
    eval.save_path=eval_results/baseline_qwen3_30b_a3b \
    > eval_logs/eval_baseline.log 2>&1

echo "[$(date)] All evaluations done (exit code: $?)"
echo "  Results: eval_results/baseline_qwen3_30b_a3b/"
