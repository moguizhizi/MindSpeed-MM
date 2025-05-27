from typing import Tuple
import torch
import vllm.model_executor.layers.rotary_embedding



def MRotaryEmbedding_forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward().

        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        assert positions.ndim == 1 or positions.ndim == 2
        import torch_npu
        if positions.ndim == 1:
            ...
            query, key = torch_npu.npu_mrope(positions, query.contiguous(), key, self.cos_sin_cache.contiguous(), self.head_size,
                                             rotary_mode='half')
        else:
            query, key = torch_npu.npu_mrope(positions, query.contiguous(), key.contiguous(), self.cos_sin_cache.contiguous(), self.head_size,
                                             mrope_section=self.mrope_section, rotary_mode='half')
        return query, key

def MRotaryEmbedding_forward_patch():
    vllm.model_executor.layers.rotary_embedding.MRotaryEmbedding.forward=MRotaryEmbedding_forward