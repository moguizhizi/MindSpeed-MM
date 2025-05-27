# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
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
import torch_npu
from megatron.training import get_args


def flash_attn_forward(qkvn, attn_mask=None, softmax_scale=1.0, sparse_mode=0, next_tokens=0):
    """FlashAttention forward"""
    args = get_args()
    q, k, v, n = qkvn
    output = torch_npu.npu_fusion_attention(
        q, k, v, n, "SBH",
        pse=None,
        padding_mask=None,
        atten_mask=attn_mask,
        scale=softmax_scale,
        pre_tockens=args.pre_tockens,
        next_tockens=next_tokens,
        keep_prob=1.0 - args.attention_dropout,
        inner_precise=0,
        sparse_mode=sparse_mode
    )
    return output


def flash_attn_backward(qkvn, dy, softmax_max, softmax_sum, attn_out,
                        attn_mask=None, softmax_scale=1.0, sparse_mode=0, next_tokens=0):
    """FlashAttention backward"""
    q, k, v, n = qkvn
    output = torch_npu.npu_fusion_attention_grad(
        q, k, v, dy, n,
        "SBH",
        pse=None,
        padding_mask=None,
        atten_mask=attn_mask,
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        attention_in=attn_out,
        scale_value=softmax_scale,
        pre_tockens=k.shape[0],
        next_tockens=next_tokens,
        sparse_mode=sparse_mode
    )
    return output
