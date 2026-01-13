class SignalProvider:
    """
    负责计算 Ground Truth Reward。
    设计原则：提供对称的 +1/-1 信号，以便 Advantage 计算时能产生正负区分。
    """
    def compute_reward(self, response_str, ground_truth_str):
        # 这里应该接入您之前的 math_grader.py 逻辑
        # 简单示例：
        is_correct = self._check_correctness(response_str, ground_truth_str)
        
        if is_correct:
            return 1.0
        else:
            return -1.0 

    def _check_correctness(self, pred, gt):
        # 实际应复用 evaluate_benchmarks.py 中的 check_sample 逻辑
        # 提取 boxed -> normalize -> compare
        return str(gt) in str(pred) # 临时占位