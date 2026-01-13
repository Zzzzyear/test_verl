#!/bin/bash
set -x

# ================= 路径配置 =================
PROJECT_ROOT="/data/zhaoqn/workspace/EGPO"
# 使用你的 mixed_debug 数据集
DATA_DIR="$PROJECT_ROOT/datasets/processed"
TRAIN_FILE="$DATA_DIR/mixed_debug.parquet"
TEST_FILE="$DATA_DIR/mixed_debug.parquet"
MODEL_PATH="/data/zhaoqn/models/Qwen/Qwen3-1.7B"

# 输出路径
EXP_NAME="dry_run_egpo_v2"
CKPT_DIR="$PROJECT_ROOT/outputs/checkpoints/$EXP_NAME"
LOG_DIR="$PROJECT_ROOT/outputs/logs/$EXP_NAME"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

# ================= 环境配置 (复用成功经验) =================
export VLLM_USE_V1=1
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF=""
export CUDA_VISIBLE_DEVICES=0

# ================= WandB 配置 =================
# 指定 WandB 的元数据和保存目录
export WANDB_PROJECT="EGPO_DryRun"       # 项目名称
export WANDB_NAME="$EXP_NAME"            # 本次实验的具体名称
export WANDB_DIR="$LOG_DIR"              # WandB 的本地日志保存路径
export WANDB_MODE="online"               # 确保是在线模式，如果是离线环境改用 "offline"
# export WANDB_API_KEY="your_key_here"   # 如果服务器没有登录过 wandb，需要解开注释填入 Key

# ================= 启动命令 =================
# 说明：
# 1. algorithm.adv_estimator=egpo : 启用我们在 Core Algos 注册的算法
# 2. reward_model.reward_manager=hybrid : 启用我们注册的混合奖励 Loop
# 3. use_kl_loss=True : 保持开启以走通标准 PPO 路径 (避免 Batch Size 丢失 bug)
# 4. kl_loss_coef=0.0 : 数学上禁用 KL 惩罚 
# 5. trainer.logger : 【关键修改】开启 wandb 支持

python3 -m verl.trainer.main_ppo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
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
    algorithm.adv_estimator=egpo \
    algorithm.use_kl_in_reward=False \
    algorithm.egpo.entropy_mode=answer \
    algorithm.egpo.lambda_min=0.5 \
    algorithm.egpo.lambda_max=2.0 \
    reward_model.reward_manager=hybrid \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='EGPO_DryRun' \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_training_steps=5