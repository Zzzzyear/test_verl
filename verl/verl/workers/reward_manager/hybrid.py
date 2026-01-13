import torch
import numpy as np
import re
import ast
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.utils.reward_score import gsm8k

# ======================================================
#  🛠️ 辅助函数区 (提升至全局，避免循环内重复定义)
# ======================================================

def to_flat_int_list(data_obj):
    """将任意维度的 List/Tensor/Numpy 展平为 int list"""
    if hasattr(data_obj, 'tolist'): data_obj = data_obj.tolist()
    if hasattr(data_obj, 'cpu'): data_obj = data_obj.cpu().tolist()
    if isinstance(data_obj, (int, float, np.integer, np.floating)): return [int(data_obj)]
    flat_list = []
    if isinstance(data_obj, (list, tuple, np.ndarray)):
        for item in data_obj:
            flat_list.extend(to_flat_int_list(item))
    return flat_list

def normalize_math_str(s):
    """数学公式智能归一化"""
    if not s: return ""
    # 基础清洗
    s = str(s).replace(" ", "").replace("\n", "").replace("$", "")
    
    # [关键正则找回] 处理等价形式
    # \sqrt[3]{8} -> root(3, 8)
    s = re.sub(r"\\sqrt\[([^\]]+)\]\{([^}]+)\}", r"root(\1, \2)", s)
    # \sqrt{x}^{y} -> sqrt(x)^y
    s = re.sub(r"\\sqrt\{([^}]+)\}\^\{([^}]+)\}", r"sqrt(\1)^\2", s)
    # \sqrt{x} -> sqrt(x)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    # \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
    
    # 移除 Latex 修饰符
    s = s.replace(r"\mathrm", "").replace(r"\text", "")
    s = s.replace(r"\left", "").replace(r"\right", "")
    
    # 统一括号
    s = s.replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")
    
    return s.strip()

# ======================================================
#  🚀 Reward Manager 类定义
# ======================================================

@register(name='hybrid')
class HybridRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, num_examine, compute_score=None, **kwargs):
        super().__init__(tokenizer, num_examine, compute_score, **kwargs)
        self.tokenizer = tokenizer

    def __call__(self, data: DataProto, return_dict: bool = False):
        # 初始化 Tensor
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        for i in range(len(data)):
            data_item = data[i]
            
            # === 1. 元数据解包 ===
            extra_info = data_item.non_tensor_batch.get('extra_info', {})
            if isinstance(extra_info, np.ndarray): extra_info = extra_info.item()
            if not isinstance(extra_info, dict): extra_info = {}
            ability = extra_info.get('ability', 'math')
            
            rm_data = data_item.non_tensor_batch.get('reward_model', {})
            if isinstance(rm_data, np.ndarray): rm_data = rm_data.item()
            ground_truth = str(rm_data.get('ground_truth', '')).strip()

            # === 2. 强力解码 ===
            response_ids_raw = data_item.batch['responses']
            try:
                ids = to_flat_int_list(response_ids_raw)
                response_str = self.tokenizer.decode(ids, skip_special_tokens=True)
            except:
                response_str = ""

            # === 3. 判分逻辑 ===
            score = -1.0
            try:
                # [Code 任务]
                if ability == 'code':
                    code_match = re.search(r"```python\n(.*?)```", response_str, re.DOTALL)
                    clean_code = code_match.group(1) if code_match else response_str
                    try:
                        ast.parse(clean_code)
                        # 严格二元判定
                        if "def " in clean_code and ("return " in clean_code or "print" in clean_code):
                            score = 1.0
                    except SyntaxError:
                        pass # 保持 -1.0

                # [Math 任务]
                else:
                    matches = re.findall(r"\\boxed\{(.*?)\}", response_str)
                    extracted_ans = matches[-1] if matches else ""
                    if not extracted_ans:
                        extracted_ans = gsm8k.extract_solution(response_str)
                    
                    if extracted_ans:
                        # 直接调用外部定义的函数
                        clean_extracted = normalize_math_str(extracted_ans)
                        clean_gt = normalize_math_str(ground_truth)
                        
                        if clean_gt == clean_extracted:
                            score = 1.0
                        elif clean_gt and clean_gt in clean_extracted:
                            score = 1.0
                        elif clean_extracted and len(clean_extracted) > 3 and clean_extracted in clean_gt:
                            score = 1.0
            except:
                pass # 保持 -1.0
            
            # === 4. 安全填充 Reward (Tensor Safe) ===
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None: pad_token_id = self.tokenizer.eos_token_id

            if not isinstance(response_ids_raw, torch.Tensor):
                response_tensor = torch.tensor(to_flat_int_list(response_ids_raw))
            else:
                response_tensor = response_ids_raw.view(-1)

            if response_tensor.is_cuda: response_tensor = response_tensor.cpu()

            if pad_token_id is not None:
                response_len = (response_tensor != pad_token_id).sum().item()
            else:
                response_len = len(response_tensor)

            if response_len > 0:
                idx = min(response_len - 1, reward_tensor.size(1) - 1)
                reward_tensor[i, idx] = score
            else:
                if reward_tensor.size(1) > 0: reward_tensor[i, 0] = score

        if return_dict: return {'reward_tensor': reward_tensor}
        return reward_tensor