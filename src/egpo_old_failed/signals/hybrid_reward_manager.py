# 文件路径: src/egpo/signals/hybrid_reward_manager.py
import torch
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.utils.reward_score import gsm8k, math_verify

@register(name='hybrid')
class HybridRewardManager(AbstractRewardManager):
    """
    EGPO 混合奖励管理器 (Standard API)
    用于适配 Driver 端的 load_reward_manager 调用。
    逻辑与 HybridRewardLoop 保持一致，但适配不同的接口。
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key 

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict:
        # 如果已经有分数，直接返回
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        # 准备输出容器 [Batch, SeqLen]
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        
        # 遍历 Batch
        for i in range(len(data)):
            data_item = data[i]
            
            # 1. 解码
            response_ids = data_item.batch["responses"]
            # 注意：在 Standard API 中，responses 可能包含 padding
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            # 2. 获取元数据
            ability = data_item.non_tensor_batch.get("ability", "math") 
            rm_data = data_item.non_tensor_batch.get("reward_model", {})
            ground_truth = rm_data.get("ground_truth", "")
            
            # 3. 判分逻辑 (与 Loop 保持一致)
            score = -1.0
            try:
                if ability == 'code':
                    if "def " in response_str and "```" in response_str:
                         score = 1.0
                    else:
                         score = -1.0
                else:
                    ans = gsm8k.extract_answer(response_str)
                    if not ans:
                        ans = math_verify.parse(response_str)
                    
                    if ans and str(ground_truth) in str(ans):
                        score = 1.0
                    elif str(ground_truth) in response_str:
                        score = 1.0
                    else:
                        score = -1.0
            except:
                score = -1.0

            # 4. 填充 (Sparse Reward, 填在最后一位)
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = score

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor

print("[EGPO] HybridRewardManager registered successfully (Standard API)")