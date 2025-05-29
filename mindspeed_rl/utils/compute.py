# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
from typing import Dict, List, Tuple, Union
import torch

_ParallelState = None
_VocabParallel = None


def set_parallel_state(obj):
    global _ParallelState
    _ParallelState = obj


def get_parallel_state():
    global _ParallelState
    return _ParallelState


def set_vocab_parallel(obj):
    global _VocabParallel
    _VocabParallel = obj


def get_vocab_parallel():
    global _VocabParallel
    return _VocabParallel


def compute_log_probs(
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the log probabilities of the given labels under the given logits.

    In the tensor parallelism case, it takes into account the vocab parallelism and
    performs the necessary adjustments to the labels and logits.

    Args:
        logits: The logits tensor.
        labels: The label tensor.

    Returns:
        Log probabilities.
    """
    vocab_parallel_cross_entropy = get_vocab_parallel()
    labels = labels.clone()
    log_probs = -vocab_parallel_cross_entropy(vocab_parallel_logits=logits, target=labels)
    return log_probs

#  verl实现
def logprobs_from_logits_v2(logits: torch.FloatTensor, labels):
    """
    A memory efficient implementation of logprobs_from_logits
    """
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(logit, dim=-1) for logit in logits])
        logprobs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        logprobs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_logprobs = torch.nn.functional.log_softmax(row_logits, dim=-1)
            row_logprobs_labels = row_logprobs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            logprobs_labels.append(row_logprobs_labels)
        logprobs_labels = torch.stack(logprobs_labels)
    return logprobs_labels