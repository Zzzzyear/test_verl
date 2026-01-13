import asyncio
import numpy as np
import re
import ast
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.utils.reward_score import gsm8k

# ======================================================
#  🛠️ 辅助函数区 (与 Driver 端保持 100% 一致)
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
    """数学公式智能归一化 (包含正则增强)"""
    if not s: return ""
    s = str(s).replace(" ", "").replace("\n", "").replace("$", "")
    
    # [关键] 必须与 Driver 端正则保持一致
    s = re.sub(r"\\sqrt\[([^\]]+)\]\{([^}]+)\}", r"root(\1, \2)", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}\^\{([^}]+)\}", r"sqrt(\1)^\2", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
    
    s = s.replace(r"\mathrm", "").replace(r"\text", "")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")
    return s.strip()

# ======================================================
#  🚀 Reward Loop 类定义
# ======================================================

@register("hybrid")
class HybridRewardLoop(RewardLoopManagerBase):
    def __init__(self, config, tokenizer, **kwargs):
        super().__init__(config, tokenizer)
        # [安全修复] 显式保存 tokenizer，防止基类没存导致 AttributeError
        self.tokenizer = tokenizer

    async def run_single(self, data_item) -> dict:
        """
        执行单个样本的奖励计算 (Worker Side)
        """
        # --- 1. 元数据解包 ---
        extra_info = data_item.non_tensor_batch.get('extra_info', {})
        if isinstance(extra_info, np.ndarray): extra_info = extra_info.item()
        if not isinstance(extra_info, dict): extra_info = {}
        ability = extra_info.get('ability', 'math')
        
        rm_data = data_item.non_tensor_batch.get('reward_model', {})
        if isinstance(rm_data, np.ndarray): rm_data = rm_data.item()
        ground_truth = str(rm_data.get('ground_truth', '')).strip()

        # --- 2. 解码准备 ---
        response_ids_raw = data_item.batch['responses']

        # --- 3. 核心判分 (同步函数) ---
        def compute_score_fn():
            try:
                ids = to_flat_int_list(response_ids_raw)
                # [安全修复] 如果 tokenizer 没存，这里会报错，现在我们在 init 里修了
                response_str = self.tokenizer.decode(ids, skip_special_tokens=True)
            except:
                return -1.0

            score = -1.0
            try:
                # [Code 任务]
                if ability == 'code':
                    code_match = re.search(r"```python\n(.*?)```", response_str, re.DOTALL)
                    clean_code = code_match.group(1) if code_match else response_str
                    try:
                        ast.parse(clean_code)
                        if "def " in clean_code and ("return " in clean_code or "print" in clean_code):
                            score = 1.0
                    except SyntaxError:
                        pass

                # [Math 任务]
                else:
                    matches = re.findall(r"\\boxed\{(.*?)\}", response_str)
                    extracted_ans = matches[-1] if matches else ""
                    if not extracted_ans:
                        extracted_ans = gsm8k.extract_solution(response_str)
                    
                    if extracted_ans:
                        # 调用全局清洗函数
                        clean_extracted = normalize_math_str(extracted_ans)
                        clean_gt = normalize_math_str(ground_truth)
                        
                        if clean_gt == clean_extracted:
                            score = 1.0
                        elif clean_gt and clean_gt in clean_extracted:
                            score = 1.0
                        elif clean_extracted and len(clean_extracted) > 3 and clean_extracted in clean_gt:
                            score = 1.0
            except:
                pass
            
            return score

        # 异步执行，防止阻塞 Ray Actor 的事件循环
        score = await self.loop.run_in_executor(None, compute_score_fn)

        return {
            "reward_score": score,
            "reward_extra_info": {"ability": ability}
        }