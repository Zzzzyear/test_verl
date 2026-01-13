# 文件路径: src/tests/test_egpo_core.py
import unittest
import torch
from unittest.mock import MagicMock

# [关键修正]: 直接从 egpo 导入，因为 PYTHONPATH 已包含 src
from egpo.entropy.mask_generator import EntropyMaskGenerator
from egpo.config import EGPOConfig

class TestEGPOCore(unittest.TestCase):

    def setUp(self):
        """准备测试环境：Mock Tokenizer 和 Config"""
        # 1. Mock Tokenizer
        self.tokenizer = MagicMock()
        # 模拟 encode 方法：假设 <think> = 101, </think> = 102
        # 使用 side_effect 模拟不同输入的返回值
        def encode_side_effect(text, **kwargs):
            if text == "<think>": return [101]
            if text == "</think>": return [102]
            return [0]
        self.tokenizer.encode.side_effect = encode_side_effect
        
        # 2. Config
        self.config = MagicMock()
        # 默认测试 Answer 模式
        self.config.egpo.entropy_mode = "answer"
        self.config.egpo.lambda_min = 0.5
        self.config.egpo.lambda_max = 2.0
        self.config.egpo.entropy_epsilon = 1e-6

    def test_mask_generation_answer_mode(self):
        """测试：entropy_mode='answer' 时，只 Mask 思考结束后的部分"""
        # 初始化
        generator = EntropyMaskGenerator(self.tokenizer, entropy_mode="answer")
        # 强行注入 ID，防止 Mock 初始化时的查找失败
        generator.think_start_id = 101
        generator.think_end_id = 102
        
        # 构造数据: [Prompt(1,2), <think>(101), ThinkContent(5,6), </think>(102), Answer(8,9)]
        # 索引对应: 0, 1, 2, 3, 4, 5, 6, 7
        input_ids = torch.tensor([[1, 2, 101, 5, 6, 102, 8, 9]])
        attention_mask = torch.ones_like(input_ids)

        mask = generator.generate_mask(input_ids, attention_mask)
        
        # 预期逻辑：
        # generator 找到 </think> (102) 在 index 5
        # split_pos = 5 + 1 = 6
        # mask[6:] = 1.0 (即 index 6, 7 为 1)
        expected = torch.tensor([[0., 0., 0., 0., 0., 0., 1., 1.]])
        
        print(f"\n[Test Mask Answer] \nCalculated: {mask}\nExpected:   {expected}")
        self.assertTrue(torch.equal(mask, expected), "Answer Mode Mask 生成错误！")

    def test_mask_generation_thinking_mode(self):
        """测试：entropy_mode='thinking' 时，只 Mask 思考部分"""
        generator = EntropyMaskGenerator(self.tokenizer, entropy_mode="thinking")
        generator.think_start_id = 101
        generator.think_end_id = 102
        
        input_ids = torch.tensor([[1, 2, 101, 5, 6, 102, 8, 9]])
        attention_mask = torch.ones_like(input_ids)
        
        mask = generator.generate_mask(input_ids, attention_mask)
        
        # 预期逻辑：
        # split_pos = 6
        # mask[:6] = 1.0
        expected = torch.tensor([[1., 1., 1., 1., 1., 1., 0., 0.]])
        
        print(f"\n[Test Mask Thinking] \nCalculated: {mask}\nExpected:   {expected}")
        self.assertTrue(torch.equal(mask, expected), "Thinking Mode Mask 生成错误！")

    def test_advantage_scaling_math(self):
        """测试：核心公式 A_final = A_base * Clip(H_mean / H_i)"""
        
        h_i = torch.tensor([0.1, 1.0, 1.9])
        h_mean = h_i.mean() # 1.0
        epsilon = 1e-6
        
        # 原始比例: [10.0, 1.0, 0.52]
        raw_ratio = h_mean / (h_i + epsilon)
        
        # Clip [0.5, 2.0]
        lambda_min = 0.5
        lambda_max = 2.0
        
        expected_weight = torch.clamp(raw_ratio, min=lambda_min, max=lambda_max)
        # 预期: [2.0, 1.0, 0.526...]
        
        print(f"\n[Test Scaling Math]")
        print(f"H_i: {h_i}")
        print(f"Expected Weight: {expected_weight}")
        
        self.assertAlmostEqual(expected_weight[0].item(), 2.0, places=4)
        self.assertAlmostEqual(expected_weight[1].item(), 1.0, places=4)
        self.assertTrue(expected_weight[2].item() < 0.6)

if __name__ == '__main__':
    unittest.main()