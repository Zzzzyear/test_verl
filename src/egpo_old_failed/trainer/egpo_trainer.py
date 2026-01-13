# 文件路径: src/egpo/trainer/egpo_trainer.py
import torch
import numpy as np
import os
from typing import Optional
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo import core_algos
from egpo.entropy.mask_generator import EntropyMaskGenerator
from egpo.core_config import EGPOConfig

# =========================================================================
# 1. 注册 EGPO Advantage 计算函数 (核心算法逻辑)
# =========================================================================
def compute_egpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    input_ids: torch.Tensor,
    egpo_config: EGPOConfig,
    mask_generator: EntropyMaskGenerator,
    norm_adv_by_std_in_grpo: bool = True,
    **kwargs
):
    """
    EGPO 优势估计器：
    1. 计算标准 GRPO 优势 (A_grpo)
    2. 计算序列级熵 (H)
    3. 计算动态权重 Weight = Mean_H_Group / (H + epsilon)
    4. 返回 A_final = A_grpo * Weight
    """
    # --- Step 1: 标准 GRPO 计算 ---
    adv, returns = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo
    )
    
    # --- Step 2: 熵计算与加权 ---
    with torch.no_grad():
        # 生成掩码 (只计算 Answer 或 Thinking 部分)
        entropy_mask = mask_generator.generate_mask(input_ids, response_mask)
        
        # 计算 Token 级 NLL
        token_nll = -old_log_probs * entropy_mask
        
        # 计算平均熵 (Sum / Count)
        valid_counts = entropy_mask.sum(dim=-1) + 1e-8
        seq_entropy = token_nll.sum(dim=-1) / valid_counts # Shape: [Batch]
        
        # --- Step 3: 分组并计算 Scaling Factor ---
        unique_indices, inverse_indices = np.unique(index, return_inverse=True)
        device = adv.device
        group_indices = torch.tensor(inverse_indices, device=device)
        num_groups = len(unique_indices)
        
        # 计算组内平均熵
        group_sum_entropy = torch.zeros(num_groups, device=device).scatter_add_(0, group_indices, seq_entropy)
        group_counts = torch.zeros(num_groups, device=device).scatter_add_(0, group_indices, torch.ones_like(seq_entropy))
        group_mean_entropy = group_sum_entropy / (group_counts + 1e-8)
        
        # 广播回每个样本
        batch_group_mean = group_mean_entropy[group_indices]
        
        # 计算权重: Weight = Mean / (Self + epsilon)
        epsilon = egpo_config.entropy_epsilon
        ratio = batch_group_mean / (seq_entropy + epsilon)
        
        # 截断权重防止梯度爆炸
        weight = torch.clamp(ratio, min=egpo_config.lambda_min, max=egpo_config.lambda_max)
        weight_expanded = weight.unsqueeze(-1) # Shape: [Batch, 1]
        
    # --- Step 4: 应用加权 ---
    adv_final = adv * weight_expanded
    
    return adv_final, returns

# 注册到 Verl
core_algos.register_adv_est("egpo_grpo")(compute_egpo_advantage)


# =========================================================================
# 2. EGPOTrainer 实现
# =========================================================================

# 保存原始函数引用，用于 Monkey Patch
from verl.trainer.ppo import ray_trainer as original_ray_trainer_module
_original_compute_advantage = original_ray_trainer_module.compute_advantage

class EGPOTrainer(RayPPOTrainer):
    def __init__(self, config, tokenizer, **kwargs):
        # 打印调试信息 (使用正确的 Python 语法)
        print(f"[EGPO] Trainer initializing in PID: {os.getpid()}")
        
        # 检查 Config，打印 Batch Size 确认 (不修改它，避免 TypeError)
        actor_conf = config.actor_rollout_ref.actor
        ppo_micro_bsz = actor_conf.get("ppo_micro_batch_size_per_gpu")
        print(f"[EGPO] Check: actor.ppo_micro_batch_size_per_gpu = {ppo_micro_bsz}")

        # 调用父类初始化
        super().__init__(config=config, tokenizer=tokenizer, **kwargs)
        
        # 初始化 EGPO 配置
        self.egpo_config = getattr(config, 'egpo', None)
        if self.egpo_config is None:
            self.egpo_config = EGPOConfig()
        
        # 初始化熵掩码生成器
        self.mask_generator = EntropyMaskGenerator(tokenizer, self.egpo_config.entropy_mode)
        
        # 自动切换算法名称 (如果用户配置写的是 grpo)
        if config.algorithm.adv_estimator == 'grpo':
            print(f"[EGPO] Swapping adv_estimator from 'grpo' to 'egpo_grpo'")
            self.config.algorithm.adv_estimator = 'egpo_grpo'

    def fit(self):
        """
        重写 fit 方法的核心目的是在调用 compute_advantage 时注入额外数据。
        我们通过临时替换模块级函数 (Monkey Patch) 来实现这一点。
        """
        original_func = original_ray_trainer_module.compute_advantage
        
        # 定义闭包函数，它可以访问 self.egpo_config 和 self.mask_generator
        def egpo_compute_advantage_wrapper(data, adv_estimator, **kwargs):
            if adv_estimator == 'egpo_grpo':
                # 显式调用我们的逻辑，并传入 data.batch 中的额外字段
                adv, returns = compute_egpo_advantage(
                    token_level_rewards=data.batch['token_level_rewards'],
                    response_mask=data.batch['response_mask'],
                    index=data.non_tensor_batch['uid'],
                    norm_adv_by_std_in_grpo=kwargs.get('norm_adv_by_std_in_grpo', True),
                    # 【关键】注入标准接口没有透传的参数
                    old_log_probs=data.batch['old_log_probs'],
                    input_ids=data.batch['input_ids'],
                    egpo_config=self.egpo_config,
                    mask_generator=self.mask_generator
                )
                data.batch["advantages"] = adv
                data.batch["returns"] = returns
                return data
            else:
                # 如果不是我们的算法，回退到原始逻辑
                return original_func(data, adv_estimator, **kwargs)
        
        # 应用 Patch
        original_ray_trainer_module.compute_advantage = egpo_compute_advantage_wrapper
        
        try:
            # 执行训练循环
            super().fit()
        finally:
            # 还原 Patch (保证无副作用)
            original_ray_trainer_module.compute_advantage = original_func