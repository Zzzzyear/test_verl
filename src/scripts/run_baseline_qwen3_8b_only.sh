#!/bin/bash
# 文件路径: src/scripts/run_dry_run.sh
set -e 

# 路径配置
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$SRC_DIR")"

# 强制设置 PYTHONPATH
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH}"

# 黄金环境配置
export VLLM_USE_V1=1
unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_VISIBLE_DEVICES=0

# 数据与模型
DATA_DIR="$PROJECT_ROOT/datasets/processed"
TRAIN_FILE="$DATA_DIR/mixed_debug.parquet"
TEST_FILE="$DATA_DIR/mixed_debug.parquet" 
MODEL_PATH="/data/zhaoqn/models/Qwen/Qwen3-1.7B"

# 实验配置
EXP_NAME="dry_run_egpo_hybrid"
OUTPUT_DIR="$PROJECT_ROOT/outputs/checkpoints/$(date +%Y%m%d)_$EXP_NAME"
mkdir -p "$OUTPUT_DIR"

echo "🚀 Running EGPO Dry Run (Hybrid Mode)..."

python3 -m egpo.main_egpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.adv_estimator=egpo_grpo \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=hybrid \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='EGPO_DryRun' \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_training_steps=5 \
    hydra.run.dir="$OUTPUT_DIR" \
    +egpo.entropy_mode="answer" \
    +egpo.lambda_min=0.5 \
    +egpo.lambda_max=2.0 \
    +egpo.entropy_epsilon=1e-6