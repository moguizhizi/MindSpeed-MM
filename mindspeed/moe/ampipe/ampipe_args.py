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
from dataclasses import dataclass, field
from typing import Union, Callable, Optional, List

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.legacy.model import LayerNorm, RMSNorm


@dataclass
class ForwardArgs:
    attention_dense: tensor_parallel.RowParallelLinear
    bias_dropout_add_func: Callable
    post_attention_norm: Union[LayerNorm, RMSNorm]
    moe: torch.nn.Module
    hidden_dropout: float


@dataclass
class FlashAttentionFwdArgs:
    flash_tensor_list: List[Tensor]
    attention_dense: tensor_parallel.RowParallelLinear
    head_num: int
    softmax_scale: float
    chunk_len: int
    q_token_start_idx: int = 0
    sparse_mode: int = 0
    cur_degree: int = 0
    kv_list: List[Tensor] = field(default_factory=list)
    o_max_sum_list: List[Tensor] = field(default_factory=list)


@dataclass
class FACpFwdArgs:
    q: Tensor
    k: Tensor
    v: Tensor


@dataclass
class FlashAttentionSaveForBwdArgs:
    n: int = 0
    rank: int = 0
    keep_prob: float = 0.0
    cp_size: int = 0
    prev_rank: int = 0
    next_rank: int = 0
    softmax_scale: float = 0.0
    next_tokens: int = 0
    cp_group: torch.distributed.ProcessGroup = None
    cp_group_for_send_recv_overlap: torch.distributed.ProcessGroup = None
    rng_states_qa_kva: List = field(default_factory=list)
    rng_states_qb_kva: List = field(default_factory=list)
    rng_states_qb_kvb: List = field(default_factory=list)


@dataclass
class FlashAttentionBwdArgs:
    grad_q: List
    grad_k: Optional[Tensor]
    grad_v: Optional[Tensor]
    flash_tensor_list: List[Tensor]
    dense_tensor_list: List[Tensor]
    attn_out_all: Tensor = None
    k: Tensor = None
    v: Tensor = None
    cur_degree: int = 0
    flash_tensor_list_len: int = 0
    dense_tensor_list_len: int = 0
    kv_list: List[Tensor] = field(default_factory=list)
    dkv_list: List[Tensor] = field(default_factory=list)
    dout_list: List[Tensor] = field(default_factory=list)


@dataclass
class BiasDropoutAddNormArgs:
    bias_dropout_add_func: Callable
    post_attention_norm: Union[LayerNorm, RMSNorm]
    residual: Tensor
    bias: Optional[Tensor]
    prob: float


@dataclass
class FwdCommArgs:
    cur_degree: int
    mlp_inputs: List[Tensor]
    a2a_inputs: List[Tensor]
    a2a_events: List
    ag_events: List


@dataclass
class BwdCommArgs:
    cur_degree: int
    grad_mlp_input_list: List[Tensor]
    grad_a2a_input_list: List[Tensor]
    a2a_events: List
    ag_events: List


@dataclass
class MLPFwdArgs:
    a2a_events: List = field(default_factory=list)
    ag_events: List = field(default_factory=list)


@dataclass
class MLPSaveForBwdArgs:
    ampipe_degree: int = 0
    num_local_experts: int = 0
    ep_size: int = 0
    hidden_size: int = 0
    sequence_parallel: bool = False
    multi_data: int = 0
    multi_stream: bool = False
    input_list_before_expert: List[Tensor] = field(default_factory=list)


@dataclass
class MLPBwdArgs:
    sequence_parallel: bool
    mlp_tensor_list_len: int
    a2a_events: List
    ag_events: List
    mlp_tensor_list: List[Tensor]
    second_a2a_events: List = field(default_factory=list)


@dataclass
class PostMLPArgs:
    seqlen: int = 0
    a2a_events: List = field(default_factory=list)
    moe_output_list: List[Tensor] = field(default_factory=list)
    weights_list: List[Tensor] = field(default_factory=list)
    token_ec_idx_list: List[Tensor] = field(default_factory=list)
