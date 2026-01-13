#!/bin/bash
set -x

# ================= 配置区 =================
PROJECT_ROOT="/data/zhaoqn/workspace/EGPO"
MODEL_PATH="/data/zhaoqn/models/Qwen/Qwen3-1.7B"
DATA_DIR="$PROJECT_ROOT/verl/data/gsm8k"
CKPT_DIR="$PROJECT_ROOT/outputs/checkpoints"
LOG_DIR="$PROJECT_ROOT/outputs/logs"

export WANDB_DIR="$LOG_DIR"
export WANDB_PROJECT="EGPO_Test_Runs"
export WANDB_NAME="qwen3_1.7b_quick_test"
mkdir -p $CKPT_DIR $LOG_DIR

# ================= 关键环境配置 (Fix vLLM 0.11 Compatibility) =================
# 1. 强制使用 XFORMERS 后端 (vLLM 默认)
# export VLLM_ATTENTION_BACKEND=XFORMERS xformers已卸载

# 2. 显式开启 vLLM V1 引擎
# 原因: verl 的 vllm_async_server.py 使用了 AsyncLLM，这是 V1 引擎的 API。
# 如果不开启，会报 "ValueError: Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False"
export VLLM_USE_V1=1

# 3. 禁用 PyTorch Expandable Segments
# 原因: vLLM V1 引擎的 Memory Pool 与 PyTorch 的 expandable_segments 互斥。
# 如果开启，会报 "AssertionError: Expandable segments are not compatible with memory pool"
# 我们这里显式 unset 它，确保它为空。
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF=""

# ================= 启动命令 =================
# gpu_memory_utilization=0.25: 限制 vLLM 显存占用，留给训练
# total_training_steps=5: 测试模式，跑几步就停

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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='EGPO_Test' \
    trainer.experiment_name='qwen3_quick_test' \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=2 \
    trainer.test_freq=5 \
    trainer.total_training_steps=5