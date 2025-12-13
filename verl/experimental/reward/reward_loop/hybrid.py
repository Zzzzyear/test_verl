import asyncio
import numpy as np
import torch
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.utils.reward_score import gsm8k
import re

@register("hybrid")
class HybridRewardLoop(RewardLoopManagerBase):
    def __init__(self, config, tokenizer, **kwargs):
        super().__init__(config, tokenizer)

    async def run_single(self, data_item) -> dict:
        """
        执行单个样本的奖励计算 (参考官方 Naive 架构优化)
        """
        # --- 1. 获取并解包数据 (修复 AttributeError) ---
        # DataProtoItem.non_tensor_batch 中的 dict 往往被封装在 numpy 数组中
        extra_info = data_item.non_tensor_batch.get('extra_info', {})
        if isinstance(extra_info, np.ndarray):
            extra_info = extra_info.item()
        
        # 防御性：确保解包后是 dict
        if not isinstance(extra_info, dict):
            extra_info = {}

        # 获取能力标签 (默认为 math)
        ability = extra_info.get('ability', 'math')
        
        # 获取 Ground Truth
        rm_data = data_item.non_tensor_batch.get('reward_model', {})
        if isinstance(rm_data, np.ndarray):
            rm_data = rm_data.item()
        ground_truth = rm_data.get('ground_truth', '')

        # --- 2. 准备 Response ID (修复 TypeError) ---
        response_ids_raw = data_item.batch['responses']
        
        # 定义强力展平函数
        def to_flat_int_list(data):
            if hasattr(data, 'tolist'): data = data.tolist()
            if hasattr(data, 'cpu'): data = data.cpu().tolist()
            if isinstance(data, (int, float, np.integer, np.floating)): return [int(data)]
            flat_list = []
            if isinstance(data, (list, tuple, np.ndarray)):
                for item in data:
                    flat_list.extend(to_flat_int_list(item))
            return flat_list

        # --- 3. 异步解码 (参考官方优化) ---
        # 使用 run_in_executor 避免阻塞 asyncio 循环
        def decode_fn():
            try:
                # 展平并解码
                ids = to_flat_int_list(response_ids_raw)
                return self.tokenizer.decode(ids, skip_special_tokens=True)
            except Exception as e:
                print(f"[HybridLoop] Decode Warning: {e}")
                return ""

        response_str = await self.loop.run_in_executor(None, decode_fn)

        # --- 4. 执行判分逻辑 ---
        # 封装为同步函数以便放入 executor
        def compute_score_fn(resp_str, gt, abil):
            score = -1.0
            try:
                if abil == 'code':
                    # [Code 逻辑] 简单检查 (适配 Debug 数据)
                    if "def " in resp_str: 
                        score = 1.0
                else:
                    # [Math 逻辑]
                    ans = gsm8k.extract_solution(resp_str)
                    if not ans and "\\boxed" in resp_str:
                        matches = re.findall(r"\\boxed\{(.*?)\}", resp_str)
                        if matches: ans = matches[-1]
                    
                    if ans and str(gt).strip() in str(ans).strip():
                        score = 1.0
                    elif str(gt).strip() in resp_str: 
                        score = 1.0
            except:
                pass
            return score

        # 异步执行判分
        score = await self.loop.run_in_executor(None, compute_score_fn, response_str, ground_truth, ability)

        return {
            "reward_score": score,
            "reward_extra_info": {"ability": ability}
        }
