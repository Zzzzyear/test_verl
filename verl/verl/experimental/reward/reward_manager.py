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
        # 1. 准备 Tokenizer
        input_tokenizer_local_path = copy_to_local(self.config.actor_rollout_ref.model.path)
        self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)
        
        self.reward_model_tokenizer = None
        if self.config.reward_model.enable:
            reward_model_tokenizer_local_path = copy_to_local(self.config.reward_model.model.path)
            self.reward_model_tokenizer = hf_tokenizer(reward_model_tokenizer_local_path, trust_remote_code=True)
        
        # 2. 获取自定义评分函数 (如 naive loop 需要)
        self.reward_fn = get_custom_reward_fn(self.config)
        
        # 3. 加载 Loop Manager 类 (此时 hybrid 已经在 __init__.py 注册过了)
        manager_name = self.config.reward_model.reward_manager
        print(f"[RewardManagerWorker] Requesting manager: {manager_name}")
        
        reward_loop_manager_cls = get_reward_loop_manager_cls(manager_name)
        
        # 4. 实例化 Loop Manager
        # 注意: 这里的参数必须和 hybrid.py 的 __init__ 匹配
        # 我们之前修改了 hybrid.py 加上了 **kwargs，所以这里传多少参数都安全
        self.reward_loop = reward_loop_manager_cls(
            config=self.config, 
            tokenizer=self.input_tokenizer, 
            compute_score=self.reward_fn, 
            reward_router_address=self.reward_router_address, 
            reward_model_tokenizer=self.reward_model_tokenizer
        )

    async def compute_score(self, data: DataProto) -> DataProto:
        return await self.reward_loop.run_single(data)