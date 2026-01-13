# 文件路径: src/egpo/signals/hybrid_reward_loop.py
import asyncio
import logging
import os
import torch
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl import DataProto
from verl.utils.reward_score import gsm8k

# 打印加载日志
print(f"[EGPO] Loading hybrid_reward_loop module... (PID: {os.getpid()})")

@register("hybrid")
class HybridRewardLoop(RewardLoopManagerBase):
    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None, **kwargs):
        super().__init__(config, tokenizer)
        
    async def run_single(self, data: DataProto) -> dict:
        """
        处理单个数据样本 (AgentLoop Batch Size=1)
        """
        data_item = data[0]
        response_ids = data_item.batch['responses']
        
        # 解码
        response_str = await self.loop.run_in_executor(
            None, 
            lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
        )
        
        # 获取元数据
        extra_info = data_item.non_tensor_batch.get('extra_info', {})
        ability = extra_info.get('ability')
        # 兜底
        if not ability:
             ability = data_item.non_tensor_batch.get('ability', 'math')

        rm_data = data_item.non_tensor_batch.get('reward_model', {})
        ground_truth = rm_data.get('ground_truth', '')
        
        score = -1.0
        try:
            if ability == 'code':
                # Code Mock: 检查 def 和 markdown 块
                if "def " in response_str and "```" in response_str:
                        score = 1.0 
                else:
                        score = -1.0
            else:
                # [Math/Science 逻辑]
                # 使用 verl 官方的 extract_solution
                # 关键：使用 method='flexible' 以支持非 #### 格式的答案 (如 \boxed)
                ans = gsm8k.extract_solution(response_str, method='flexible')
                
                # 如果 gsm8k 提取失败，且答案包含 Boxed，尝试手动提取 Boxed 内容作为降级方案
                if ans is None and "\\boxed" in response_str:
                    import re
                    matches = re.findall(r"\\boxed\{(.*?)\}", response_str)
                    if matches:
                        ans = matches[-1]

                # 判分逻辑
                if ans and str(ground_truth) in str(ans):
                    score = 1.0
                elif str(ground_truth) in response_str: 
                    score = 1.0
                else:
                    score = -1.0
        except Exception as e:
            print(f"[HybridLoop] Scoring failed for ability {ability}: {e}")
            score = -1.0
            
        return {
            "reward_score": score,
            "reward_extra_info": {
                "ability": ability,
                "pred_len": len(response_ids)
            }
        }

print(f"[EGPO] HybridRewardLoop registered successfully as 'hybrid'")