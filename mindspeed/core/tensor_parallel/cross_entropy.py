from typing import Tuple

import torch


def calculate_predicted_logits(
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        logits_max: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # subtraction the maximum value.
    # Use in-place to reduce memory pressure.
    vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)

    # Create a mask of valid vocab ids (1 means it needs to be masked).
    target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
    masked_target = target.clone() - vocab_start_index
    masked_target *= ~target_mask

    # Get predicted-logits = logits[target].
    # For Simplicity, we convert logits to a 2-D tensor with size
    # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
    partition_vocab_size = vocab_parallel_logits.size()[-1]
    logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
    masked_target_1d = masked_target.view(-1)
    arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
    predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
    predicted_logits_1d = predicted_logits_1d.clone().contiguous()
    predicted_logits = predicted_logits_1d.view_as(target)
    predicted_logits *= ~target_mask

    exp_logits = vocab_parallel_logits
    torch.exp(vocab_parallel_logits, out=exp_logits)
    sum_exp_logits = exp_logits.sum(dim=-1)

    return target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits
