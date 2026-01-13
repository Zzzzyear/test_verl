import torch

class EntropyMaskGenerator:
    def __init__(self, tokenizer, entropy_mode="answer"):
        self.tokenizer = tokenizer
        self.entropy_mode = entropy_mode
        
        # 获取特殊 Token ID
        try:
            self.think_start_id = tokenizer.convert_tokens_to_ids("<think>")
            self.think_end_id = tokenizer.convert_tokens_to_ids("</think>")
        except:
            # 兜底：如果 tokenizer 里没有这些 token，可能需要报错或 fallback
            self.think_start_id = -1
            self.think_end_id = -1

    def generate_mask(self, input_ids: torch.Tensor, response_mask: torch.Tensor):
        """
        生成用于计算熵的 Mask。
        """
        if self.entropy_mode == "joint":
            return response_mask

        batch_size, seq_len = input_ids.shape
        entropy_mask = torch.zeros_like(response_mask, dtype=torch.float32)
        device = input_ids.device

        for i in range(batch_size):
            # 找到 response 的有效区域
            valid_indices = torch.nonzero(response_mask[i]).squeeze(-1)
            if len(valid_indices) == 0:
                continue
            
            # Response 的起始位置
            start_pos = valid_indices[0].item()
            row_ids = input_ids[i]

            # 寻找 </think> 的位置
            end_indices = (row_ids == self.think_end_id).nonzero(as_tuple=True)[0]
            # 必须是在 start_pos 之后的 </think>
            valid_end_indices = end_indices[end_indices >= start_pos]

            if len(valid_end_indices) > 0:
                split_pos = valid_end_indices[0].item() + 1 # +1 包含标签本身或跳过
            else:
                # 没找到 </think>，如果是 answer 模式，假设全段都是 answer (fallback)
                split_pos = start_pos 

            # 根据模式打标
            if self.entropy_mode == "answer":
                if split_pos < seq_len:
                    entropy_mask[i, split_pos:] = 1.0
            elif self.entropy_mode == "thinking":
                if split_pos > start_pos:
                    entropy_mask[i, start_pos:split_pos] = 1.0
            else:
                entropy_mask[i, start_pos:] = 1.0

        # 再次与 response_mask 取交集，确保不溢出
        return entropy_mask * response_mask