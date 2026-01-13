from dataclasses import dataclass, field

@dataclass
class EGPOConfig:
    """
    EGPO 算法核心超参数配置
    公式: Weight = Clip( H_group_mean / (H_i + epsilon), lambda_min, lambda_max )
    """
    
    # 实验开关：计算哪一部分的熵
    # "answer":   仅计算 </think> 之后的内容 (B1 实验推荐，更能反映自信度)
    # "thinking": 仅计算 <think>...</think> 内部 (用于对比实验)
    # "joint":    计算全序列
    entropy_mode: str = field(
        default="answer",
        metadata={"help": "Which part to calculate entropy: 'answer' or 'thinking'"}
    )
    
    # 权重下限 (Lambda Min)
    # 防止 "犹豫且错误" 的样本权重过低导致梯度消失
    lambda_min: float = field(
        default=0.5,
        metadata={"help": "Lower bound for entropy weight clipping."}
    )
    
    # 权重上限 (Lambda Max)
    # 限制 "自信且正确" (强奖励) 或 "自信且错误" (强惩罚) 的倍率，防止梯度爆炸
    lambda_max: float = field(
        default=2.0,
        metadata={"help": "Upper bound for entropy weight clipping."}
    )
    
    # 数值稳定性项
    entropy_epsilon: float = field(
        default=1e-6,
        metadata={"help": "Small value to prevent division by zero."}
    )