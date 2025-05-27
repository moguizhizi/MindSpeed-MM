# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contain small torch utilities
"""
from transformers import PreTrainedTokenizer
import torch
import torch.nn.functional as F

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return F.pad(tensors, pad_tuple, 'constant', pad_token_id)


def tokenize_and_postprocess_data(prompt: str,
                                  tokenizer: PreTrainedTokenizer,
                                  max_length: int,
                                  pad_token_id: int,
                                  left_pad=True,
                                  truncation='error'):
    """
    input_data is the output from tokenizer.
    """
    assert truncation in ['left', 'right', 'error']

    input_data = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)

    input_ids = input_data['input_ids']
    attention_mask = input_data['attention_mask']

    assert input_ids.ndim == 2

    sequence_length = input_ids.shape[-1]
    if sequence_length < max_length:
        input_ids = pad_sequence_to_length(input_ids,
                                           max_seq_len=max_length,
                                           pad_token_id=pad_token_id,
                                           left_pad=left_pad)
        attention_mask = pad_sequence_to_length(attention_mask,
                                                max_seq_len=max_length,
                                                pad_token_id=0,
                                                left_pad=left_pad)
    elif sequence_length > max_length:
        if truncation == 'left':
            # actually, left truncation may not be reasonable
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
        elif truncation == 'right':
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        elif truncation == 'error':
            raise NotImplementedError(f'{sequence_length=} is larger than {max_length=}')
        else:
            raise NotImplementedError(f'Unknown truncation method {truncation}')

    return input_ids, attention_mask


def compute_position_id_with_mask(mask):
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)
