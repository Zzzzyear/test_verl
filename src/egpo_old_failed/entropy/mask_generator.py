import torch

class EntropyMaskGenerator:
    def __init__(self, tokenizer, entropy_mode="answer"):
        self.tokenizer = tokenizer
        self.entropy_mode = entropy_mode
        
        # 尝试获取 token id
        try:
            # 某些 tokenizer 需要在这处理
            self.think_start_id = tokenizer.convert_tokens_to_ids("<think>")
            self.think_end_id = tokenizer.convert_tokens_to_ids("</think>")
        except:
            self.think_start_id = -1
            self.think_end_id = -1

    def generate_mask(self, input_ids, response_mask):
        """
        生成用于计算熵的 Mask。
        Args:
            input_ids: [B, L] 全序列
            response_mask: [B, L] 1.0 表示是 response 部分，0.0 是 prompt 或 padding
        Returns: 
            entropy_mask: [B, L]
        """
        batch_size, seq_len = input_ids.shape
        entropy_mask = torch.zeros_like(response_mask, dtype=torch.float32)

        # 获取 tensor 所在的 device，避免设备不一致报错
        device = input_ids.device

        for i in range(batch_size):
            # 仅在 response 区域内查找标签
            # 找到 </think> 的位置
            row_ids = input_ids[i]
            # 必须结合 response_mask，防止匹配到 prompt 里的 tag (虽然 prompt 里一般没有)
            valid_indices = torch.nonzero(response_mask[i]).squeeze(-1)
            if len(valid_indices) == 0:
                continue
                
            start_pos = valid_indices[0].item()
            
            # 在该行中寻找 </think>
            end_indices = (row_ids == self.think_end_id).nonzero(as_tuple=True)[0]
            
            # 筛选出位于 response 区域内的 </think>
            valid_end_indices = end_indices[end_indices >= start_pos]

            if len(valid_end_indices) > 0:
                split_pos = valid_end_indices[0].item() + 1 # 跳过 </think> 自身
            else:
                # 没找到 </think>，回退策略
                # 如果是 Answer 模式，且没找到结束符，说明可能整个都是思考，或者格式错误。
                # 此时为了安全，若 entropy_mode="answer"，则全不选（或全选，视策略而定）。
                # 这里采用 B1 实验观察：没找到通常意味着模型还在思考被截断，或者是纯回答。
                # 简单起见：如果没找到，假设整个 Response 都是 Answer (兼容普通模型)
                split_pos = start_pos 

            if self.entropy_mode == "answer":
                if split_pos < seq_len:
                    entropy_mask[i, split_pos:] = 1.0
            elif self.entropy_mode == "thinking":
                if split_pos > start_pos:
                    entropy_mask[i, start_pos:split_pos] = 1.0
            else: 
                # joint or default
                entropy_mask[i, start_pos:] = 1.0

        # 再次确保只保留 response 部分且非 padding
        entropy_mask = entropy_mask * response_mask
        return entropy_mask