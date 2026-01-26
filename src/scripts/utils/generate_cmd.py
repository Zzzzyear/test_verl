import sys
import os
import argparse
from omegaconf import OmegaConf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, required=True) 
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--project_root", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    
    args = parser.parse_args()

    # 1. 加载配置
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    cfg = OmegaConf.load(args.config)
    
    if args.task not in cfg.tasks:
        raise ValueError(f"Task '{args.task}' not found.")
    if args.mode not in cfg.modes:
        raise ValueError(f"Mode '{args.mode}' not found.")

    defaults = cfg.defaults
    mode_cfg = cfg.modes[args.mode]
    task_cfg = cfg.tasks[args.task]
        
    # 2. 智能路径逻辑 (Robust Fallback)
    # 获取原始文件名 (例如 mixed_reasoning.parquet)
    raw_filename = task_cfg.filename
    base_name = raw_filename.replace(".parquet", "")
    
    # 定义 3 个关键路径
    raw_path = os.path.join(args.project_root, defaults.data_dir, raw_filename)
    split_train_path = os.path.join(args.project_root, defaults.data_dir, f"{base_name}_train_final.parquet")
    split_val_path = os.path.join(args.project_root, defaults.data_dir, f"{base_name}_val_fixed.parquet")
    
    # [修正 1 & 2] 完整的回退逻辑检查
    # 只要切分后的训练集 OR 验证集有一个不存在，就回退到原始文件
    if os.path.exists(split_train_path) and os.path.exists(split_val_path):
        train_path = split_train_path
        val_path = split_val_path
    else:
        print(f"[Warning] Split files not found completely. Fallback to original: {raw_filename}")
        # 如果没有切分文件，训练和验证都使用原始大文件 (注意: 此时需要在 config 中限制验证步数)
        train_path = raw_path
        val_path = raw_path
        
    # [修正 3] 提前创建 Checkpoint 目录，防止权限或路径错误
    ckpt_dir = os.path.join(args.project_root, "outputs/checkpoints", args.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 读取 Offload 配置
    use_offload = mode_cfg.get("offload", False)
    offload_str = "True" if use_offload else "False"

    # 3. 构建参数列表
    cmd_args = [
        f"data.train_files='{train_path}'",
        f"data.val_files='{val_path}'",
        
        f"data.train_batch_size={mode_cfg.mini_bs}",
        f"data.max_prompt_length={task_cfg.max_prompt_length}",
        f"data.max_response_length={task_cfg.max_response_length}",
        f"data.truncation=left",
        
        f"actor_rollout_ref.model.path='{args.model_path}'",
        f"actor_rollout_ref.actor.optim.lr={defaults.lr}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        
        f"actor_rollout_ref.actor.ppo_mini_batch_size={mode_cfg.mini_bs}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={mode_cfg.micro_bs}",
        
        # 动态 Offload
        f"actor_rollout_ref.actor.fsdp_config.param_offload={offload_str}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload_str}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=False", 
        
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={mode_cfg.micro_bs*4}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={mode_cfg.micro_bs*4}",

        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={mode_cfg.vllm_mem}",
        f"actor_rollout_ref.rollout.n={mode_cfg.rollout_n}",
        f"actor_rollout_ref.rollout.prompt_length={task_cfg.max_prompt_length}",
        f"actor_rollout_ref.rollout.response_length={task_cfg.max_response_length}",
        
        "algorithm.adv_estimator=egpo",
        "algorithm.use_kl_in_reward=False",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={defaults.kl_loss_coef}",
        f"actor_rollout_ref.actor.kl_loss_type={defaults.kl_loss_type}",
        f"algorithm.egpo.entropy_mode={defaults.entropy_mode}",
        f"algorithm.egpo.lambda_min={defaults.lambda_min}",
        f"algorithm.egpo.lambda_max={defaults.lambda_max}",
        f"algorithm.egpo.entropy_epsilon={defaults.entropy_epsilon}",
        f"algorithm.egpo.negative_weight_mode={defaults.negative_weight_mode}",

        "reward_model.reward_manager=hybrid",
        
        "trainer.critic_warmup=0",
        "trainer.logger=['console','wandb']",
        f"trainer.project_name='{defaults.wandb_project}'",
        f"trainer.experiment_name='{args.exp_name}'",
        f"trainer.default_local_dir='{ckpt_dir}'",
        f"trainer.n_gpus_per_node={mode_cfg.n_gpus}",
        f"trainer.nnodes={args.nnodes}",
        f"trainer.save_freq={mode_cfg.save_freq}",
        f"trainer.test_freq={mode_cfg.test_freq}",
        f"trainer.total_epochs={mode_cfg.total_epochs}",
        "trainer.val_before_train=False"
    ]

    print(" ".join(cmd_args))

if __name__ == "__main__":
    main()