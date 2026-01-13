#!/bin/bash
set -x

# ===============================================================
# 1. Environment & Path Configuration
# ===============================================================

# Define the absolute root path of the project
PROJECT_ROOT="/data/zhaoqn/workspace/EGPO"

# Input Paths: Model weight location and Preprocessed dataset location
MODEL_PATH="/data/zhaoqn/models/Qwen/Qwen3-1.7B"
DATA_DIR="$PROJECT_ROOT/verl/data/gsm8k"

# Output Paths: Directories for saving model checkpoints and execution logs
# Checkpoints will be saved here
CKPT_DIR="$PROJECT_ROOT/outputs/checkpoints"
# WandB local run files and console logs will be saved here
LOG_DIR="$PROJECT_ROOT/outputs/logs"

# Set WandB directory to store run metadata in the specified log directory
# ensuring the project root remains clean
export WANDB_DIR="$LOG_DIR"
export WANDB_PROJECT="EGPO_Test_Runs"
export WANDB_NAME="qwen3_1.7b_one_epoch_test"

# Create output directories if they do not exist
mkdir -p $CKPT_DIR $LOG_DIR

# Performance Optimization for A800 GPUs
# Use XFORMERS backend for vLLM to improve inference speed
export VLLM_ATTENTION_BACKEND=XFORMERS
# Reduce memory fragmentation for PyTorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===============================================================
# 2. Training Command Execution
# ===============================================================
# Explanation of key arguments:
# - adv_estimator=grpo: Enable Group Relative Policy Optimization
# - n_gpus_per_node=1: Force usage of a single GPU
# - save_freq=10: Save model checkpoint every 10 steps for verification
# - default_local_dir: Redirect model artifacts to the outputs/checkpoints folder
# - gpu_memory_utilization=0.4: Limit vLLM memory usage to 40% to prevent OOM during training

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='EGPO_Test' \
    trainer.experiment_name='qwen3_test_run_one_epoch' \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 
