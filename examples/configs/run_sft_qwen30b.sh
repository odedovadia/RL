#!/bin/bash
# nohup docker exec -it nemo_rl_dev bash

export STEP=1855

# Grid: freeze_moe_router:moe_router_topk
CONFIGS=("false:8" "false:16" "true:8")
LEARNING_RATES=(5e-06 8e-06 1e-05 3e-05 5e-05 8e-05)
EVAL_DATASETS=(math500 aime2024 aime2025 gpqa_diamond)

S3_BUCKET="s3://clm-research/models/moe_ablation"

TOTAL_EXPS=$(( ${#CONFIGS[@]} * ${#LEARNING_RATES[@]} ))
EXP_NUM=0

for CONFIG in "${CONFIGS[@]}"; do
    FREEZE_MOE_ROUTER="${CONFIG%%:*}"
    MOE_ROUTER_TOPK="${CONFIG##*:}"

    echo ""
    echo "########################################################"
    echo "# Config: freeze=${FREEZE_MOE_ROUTER} topk=${MOE_ROUTER_TOPK}"
    echo "########################################################"

    for LR in "${LEARNING_RATES[@]}"; do
        EXP_NUM=$((EXP_NUM + 1))
        set -a
        source .env
        set +a

        if [ "$FREEZE_MOE_ROUTER" = "true" ]; then
            export MODEL_NAME=sft_openmathinstruct2_megatron_30b_freeze_router
            export MOE_ROUTER_LOAD_BALANCING_TYPE="none"
            export MOE_ROUTER_BIAS_UPDATE_RATE=0.0
            export MOE_AUX_LOSS_COEFF=0.0
        else
            export MOE_ROUTER_LOAD_BALANCING_TYPE="aux_loss"
            export MOE_ROUTER_BIAS_UPDATE_RATE=0.001
            export MOE_AUX_LOSS_COEFF=0.0000001 # 1e-7
            export MODEL_NAME=sft_openmathinstruct2_megatron_30b_active_router
        fi

        EXPERIMENT_DIR="topk${MOE_ROUTER_TOPK}_lr_${LR}"
        export MODEL_NAME=${MODEL_NAME}_${EXPERIMENT_DIR}
        LOG_DIR="log_history_full/${EXPERIMENT_DIR}"
        mkdir -p "${LOG_DIR}"

        echo "========================================================"
        echo "Experiment ${EXP_NUM}/${TOTAL_EXPS}: ${MODEL_NAME}"
        echo "  FREEZE_MOE_ROUTER=${FREEZE_MOE_ROUTER}"
        echo "  MOE_ROUTER_TOPK=${MOE_ROUTER_TOPK}"
        echo "  LR=${LR}"
        echo "  MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF}"
        echo "  MOE_ROUTER_LOAD_BALANCING_TYPE=${MOE_ROUTER_LOAD_BALANCING_TYPE}"
        echo "  MOE_ROUTER_BIAS_UPDATE_RATE=${MOE_ROUTER_BIAS_UPDATE_RATE}"
        echo "  STEP=${STEP}"
        echo "  LOG_DIR=${LOG_DIR}"
        echo "========================================================"

        echo "[$(date)] Training ${MODEL_NAME}..."
        NRL_FORCE_REBUILD_VENVS=true uv run python examples/run_sft.py \
           --config examples/configs/sft_openmathinstruct2_megatron_30b.yaml \
           policy.megatron_cfg.freeze_moe_router=${FREEZE_MOE_ROUTER} \
           policy.megatron_cfg.moe_router_load_balancing_type=${MOE_ROUTER_LOAD_BALANCING_TYPE} \
           policy.megatron_cfg.moe_router_bias_update_rate=${MOE_ROUTER_BIAS_UPDATE_RATE} \
           policy.megatron_cfg.moe_aux_loss_coeff=${MOE_AUX_LOSS_COEFF} \
           policy.megatron_cfg.moe_router_topk=${MOE_ROUTER_TOPK} \
           policy.megatron_cfg.optimizer.lr=${LR} \
           logger.wandb.name=${MODEL_NAME} \
           checkpointing.checkpoint_dir=results/${MODEL_NAME} \
           sft.max_num_steps=${STEP} \
          > "${LOG_DIR}/train.log" 2>&1
        echo "[$(date)] Training done (exit code: $?)"

        echo "[$(date)] Converting checkpoint to HF format..."
        NRL_FORCE_REBUILD_VENVS=true uv run --extra mcore python examples/converters/convert_megatron_to_hf.py \
            --config=examples/configs/sft_openmathinstruct2_megatron_30b.yaml \
            --megatron-ckpt-path results/${MODEL_NAME}/step_${STEP}/policy/weights/iter_0000000 \
            --hf-ckpt-path results/${MODEL_NAME}/step_${STEP}/hf \
            > "${LOG_DIR}/convert.log" 2>&1
        echo "[$(date)] Conversion done (exit code: $?)"

        DATASETS_CSV=$(IFS=,; echo "${EVAL_DATASETS[*]}")
        echo "[$(date)] Running evaluation on ${DATASETS_CSV}..."
        NRL_FORCE_REBUILD_VENVS=true uv run examples/run_eval.py \
            --config=examples/configs/evals/eval.yaml \
            --datasets=${DATASETS_CSV} \
            generation.model_name=results/${MODEL_NAME}/step_${STEP}/hf \
            generation.vllm_cfg.max_model_len=16384 \
            generation.max_new_tokens=8192 \
            tokenizer.name=Qwen/Qwen3-30B-A3B-Base \
            eval.save_path=eval_results_full/${MODEL_NAME} \
            > "${LOG_DIR}/eval.log" 2>&1
        echo "[$(date)] Evaluation done (exit code: $?)"

        echo "[$(date)] Syncing HF checkpoint to S3..."
        aws s3 sync results/${MODEL_NAME}/step_${STEP}/hf \
            ${S3_BUCKET}/${MODEL_NAME} \
            > "${LOG_DIR}/s3_sync.log" 2>&1
        echo "[$(date)] S3 sync done (exit code: $?)"

        echo "[$(date)] Cleaning up checkpoints to save space..."
        rm -rf results/${MODEL_NAME}
        echo "[$(date)] Cleanup done"

        echo "[$(date)] Finished run: ${MODEL_NAME}"
        echo ""

    done
done

echo "========================================================"
echo "All runs complete."
echo "  Logs:    log_history_full/"
echo "  Results: eval_results_full/"
echo "  Models:  ${S3_BUCKET}/"
echo "========================================================"
