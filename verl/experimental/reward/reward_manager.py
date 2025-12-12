# 文件路径: verl/verl/experimental/reward/reward_manager.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# ... (License header) ...

import logging
import os
import ray
from omegaconf import DictConfig

# --- 原有的 imports ---
from verl.experimental.reward.reward_loop import get_reward_loop_manager_cls
from verl.protocol import DataProto
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

@ray.remote
class RewardManagerWorker:
    def __init__(self, config: DictConfig, reward_router_address: str = None):
        self.config = config
        self.reward_router_address = reward_router_address
        self._init_reward_fn()

    def _init_reward_fn(self):
        # =========================================================
        # 【EGPO 终极注入补丁】
        # 直接操作当前进程内存中的 Registry，无视任何路径/版本冲突
        # =========================================================
        try:
            print(f"[EGPO/Patch] Injecting HybridRewardLoop into Registry (PID: {os.getpid()})...")
            
            # 1. 导入我们的类
            import egpo.signals.hybrid_reward_loop
            from egpo.signals.hybrid_reward_loop import HybridRewardLoop
            
            # 2. 导入当前上下文正在使用的注册表 (就在隔壁目录，绝对不会错)
            from verl.experimental.reward.reward_loop.registry import REWARD_LOOP_MANAGER_REGISTRY
            
            # 3. 暴力写入
            REWARD_LOOP_MANAGER_REGISTRY['hybrid'] = HybridRewardLoop
            
            print(f"[EGPO/Patch] Registry Keys Now: {list(REWARD_LOOP_MANAGER_REGISTRY.keys())}")
            print(f"[EGPO/Patch] Injection SUCCESS. 'hybrid' is ready.")
            
        except Exception as e:
            print(f"[EGPO/Patch] CRITICAL ERROR during injection: {e}")
            import traceback
            traceback.print_exc()
        # =========================================================

        input_tokenizer_local_path = copy_to_local(self.config.actor_rollout_ref.model.path)
        self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)
        self.reward_model_tokenizer = None
        if self.config.reward_model.enable:
            reward_model_tokenizer_local_path = copy_to_local(self.config.reward_model.model.path)
            self.reward_model_tokenizer = hf_tokenizer(reward_model_tokenizer_local_path, trust_remote_code=True)
        
        self.reward_fn = get_custom_reward_fn(self.config)
        
        # 此时 registry 里一定有 hybrid 了
        manager_name = self.config.reward_model.reward_manager
        print(f"[RewardManagerWorker] Requesting manager: {manager_name}")
        
        reward_loop_manager_cls = get_reward_loop_manager_cls(manager_name)
        
        self.reward_loop = reward_loop_manager_cls(
            self.config, self.input_tokenizer, self.reward_fn, self.reward_router_address, self.reward_model_tokenizer
        )

    async def compute_score(self, data: DataProto) -> DataProto:
        return await self.reward_loop.run_single(data)
