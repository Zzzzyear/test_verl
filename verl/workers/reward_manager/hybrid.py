import torch
import numpy as np
import re
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.utils.reward_score import gsm8k

@register(name='hybrid')
class HybridRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, num_examine, compute_score=None, **kwargs):
        super().__init__(tokenizer, num_examine, compute_score, **kwargs)

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        for i in range(len(data)):
            data_item = data[i]
            
            # --- 1. 元数据解包 (修复 AttributeError) ---
            extra_info = data_item.non_tensor_batch.get('extra_info', {})
            if isinstance(extra_info, np.ndarray):
                extra_info = extra_info.item()
            if not isinstance(extra_info, dict):
                extra_info = {}
                
            ability = extra_info.get('ability', 'math')
            
            rm_data = data_item.non_tensor_batch.get('reward_model', {})
            if isinstance(rm_data, np.ndarray):
                rm_data = rm_data.item()
            ground_truth = rm_data.get('ground_truth', '')

            # --- 2. 强力解码 (修复 TypeError) ---
            response_ids_raw = data_item.batch['responses']
            
            def to_flat_int_list(data):
                if hasattr(data, 'tolist'): data = data.tolist()
                if hasattr(data, 'cpu'): data = data.cpu().tolist()
                if isinstance(data, (int, float, np.integer, np.floating)): return [int(data)]
                flat_list = []
                if isinstance(data, (list, tuple, np.ndarray)):
                    for item in data:
                        flat_list.extend(to_flat_int_list(item))
                return flat_list

            try:
                ids = to_flat_int_list(response_ids_raw)
                response_str = self.tokenizer.decode(ids, skip_special_tokens=True)
            except:
                response_str = ""

            # --- 3. 判分逻辑 ---
            score = -1.0
            try:
                if ability == 'code':
                    if "def " in response_str:
                        score = 1.0
                else:
                    ans = gsm8k.extract_solution(response_str)
                    if not ans and "\\boxed" in response_str:
                        matches = re.findall(r"\\boxed\{(.*?)\}", response_str)
                        if matches: ans = matches[-1]
                    
                    if ans and str(ground_truth).strip() in str(ans).strip():
                        score = 1.0
                    elif str(ground_truth).strip() in response_str:
                        score = 1.0
            except:
                pass
            
            # 填充
            valid_len = int(torch.sum(data_item.batch['attention_mask']).item())
            if 0 < valid_len <= reward_tensor.size(1):
                reward_tensor[i, valid_len - 1] = score

        if return_dict:
            return {'reward_tensor': reward_tensor}
        return reward_tensor
