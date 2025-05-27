# coding=utf-8
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2022; the respective contributors; as shown by the AUTHORS file.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import math
from functools import wraps

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from torch import Tensor
from megatron.training import get_args
from megatron.core import mpu
from mindspeed.core.models.common.embeddings.rotary_pos_embedding import yarn_get_mscale
from mindspeed.core.context_parallel.ring_context_parallel import ringattn_context_parallel
from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
from mindspeed.model.alibi_mask import AlibiForFusionAttnSingleton
from mindspeed.core.parallel_state import (get_context_parallel_group_for_hybrid_ring,
                                           get_context_parallel_for_hybrid_ring_world_size,
                                           get_context_parallel_for_hybrid_ring_rank,
                                           get_context_parallel_for_hybrid_ring_global_ranks,
                                           get_ring_ranks_for_intra_window,
                                           get_ring_ranks_for_inter_window_kv,
                                           get_ring_ranks_for_inter_window_dkv,
                                           get_ring_group_for_intra_window,
                                           get_ring_group_for_intra_window_send_recv_overlap)
from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP
from mindspeed.model.transformer import get_attention_mask
from mindspeed.core.context_parallel.adaptive_context_parallel import adaptive_attn_context_parallel
from mindspeed.core.context_parallel.utils import get_scheduling_info
from mindspeed.utils import get_actual_seq_len

try:
    from einops import rearrange, repeat
except ImportError:
    rearrange = None


def dot_product_attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        config = args[1] if len(args) > 1 else kwargs['config']
        cp_size = config.context_parallel_size
        config.context_parallel_size = 1
        fn(self, *args, **kwargs)
        config.context_parallel_size = cp_size

        # add pse
        args = get_args()
        self.pse = None
        self.pse_type = args.alibi_fusion_attn_type

        if args.multi_head_latent_attention:
            self.scale_mask_softmax.scale = True
            self.hidden_size_per_partition = args.num_attention_heads * args.v_head_dim
            self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
            self.softmax_scale = self.q_head_dim ** (-0.5)

            if args.rope_scaling_type is not None:
                mscale_all_dim = args.rope_scaling_mscale_all_dim if args.rope_scaling_mscale_all_dim else 0
                scaling_factor = args.rope_scaling_factor

                if mscale_all_dim:
                    mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                    self.softmax_scale = self.softmax_scale * mscale * mscale

            self.norm_factor = 1.0 / self.softmax_scale

        if self.pse_type is None:
            self.pse_type = 1  # not use pse
        elif self.pse_type == 0:
            alibi = AlibiForFusionAttnSingleton.get_alibi_tensor_for_fusion_attn(args.seq_length,
                                                                                 args.num_attention_heads,
                                                                                 args.params_dtype,
                                                                                 args.alibi_diagonal_opposite,
                                                                                 1024)
            self.pse = alibi
        elif self.pse_type == 2 or self.pse_type == 3:
            self.pse = AlibiForFusionAttnSingleton.get_alibi_slopes_for_fusion_attn(args.num_attention_heads)

    return wrapper


def dot_product_attention_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, query, key, value, attention_mask, attn_mask_type, packed_seq_params):
        if attention_mask is None:
            attention_mask = get_attention_mask()
        if get_args().use_flash_attn:
            return dot_product_attention_forward(self, query, key, value, attention_mask, attn_mask_type,
                                                 packed_seq_params)
        return fn(self, query, key, value, attention_mask, attn_mask_type, packed_seq_params)

    return wrapper


def dot_product_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask,
        attn_mask_type,
        packed_seq_params,
):
    is_vit = (getattr(self.config, 'model_id', None) == 'MiniCPMViT')
    seq_length, bsz, n_head, head_dim = query.shape[0], query.shape[1], query.shape[2], query.shape[3]
    if not is_vit:
        if attention_mask is not None:
            attention_mask = None
    query = query.transpose(0, 1).contiguous()
    key = key.transpose(0, 1).contiguous()
    value = value.transpose(0, 1).contiguous()

    if attention_mask is not None:
        query, key, value, indices_q, cu_seq_lens, max_seq_lens = _unpad_input(
            query, key, value, attention_mask, seq_length
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        attn_output_upad = torch_npu.npu_fusion_attention(
            query, key, value, n_head,
            pse=None,
            atten_mask=None,
            scale=1.0 / math.sqrt(query.shape[-1]),
            keep_prob=1,
            input_layout="TND",
            actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
            actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()))[0]
        attn_output = _pad_input(attn_output_upad, indices_q, bsz, seq_length)
    else:
        atten_mask_npu = torch.from_numpy(np.triu(np.ones([2048, 2048]), k=1)).bool().to(query.device)
        attn_output = torch_npu.npu_fusion_attention(query, key, value, n_head, "BSND", keep_prob=1.0,
                                                     scale=1.0 / math.sqrt(query.shape[-1]), atten_mask=atten_mask_npu,
                                                     sparse_mode=3)[0]
    attn_output = attn_output.reshape(bsz, seq_length, n_head * head_dim)

    return attn_output


def _unpad_input(query_layer, key_layer, value_layer, attention_mask, query_length):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(
            query_layer.reshape(batch_size * kv_seq_len, query_layer.shape[2], head_dim), indices_k
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _pad_input(hidden_states, indices, batch, seqlen):
    dim = hidden_states.shape[-1]
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, indices):
        ctx.save_for_backward(indices)
        if input_tensor.ndim < 2:
            raise ValueError("Input tensor must have at least 2 dimensions")
        ctx.first_axis_dim, other_shape = input_tensor.shape[0], input_tensor.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            rearrange(input_tensor, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        if grad_output.ndim < 2:
            raise ValueError("Gradient output tensor must have at least 2 dimensions")
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        if indices.ndim != 1:
            raise ValueError("Indices must be 1 dimension")
        if values.ndim < 2:
            raise ValueError("Values must have at least 2 dimensions")
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


class IndexFirstAxisResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, indices):
        ctx.save_for_backward(indices)
        if input_tensor.ndim < 2:
            raise ValueError("Input tensor must have at least 2 dimensions")
        ctx.first_axis_dim, other_shape = input_tensor.shape[0], input_tensor.shape[1:]
        second_dim = other_shape.numel()
        output = input_tensor[indices]

        return output, input_tensor.detach()

    @staticmethod
    def backward(ctx, grad_output, grad_residual):
        (indices,) = ctx.saved_tensors
        if grad_output.ndim < 2:
            raise ValueError("Gradient output tensor must have at least 2 dimensions")
        other_shape = grad_output.shape[1:]
        if grad_residual.shape[1:] != other_shape:
            raise ValueError("Gradient residual tensor must have the same shape as the output")
        grad_input = grad_residual
        indices = indices.reshape(indices.shape[0], *((1,) * (grad_output.ndim - 1)))
        indices = indices.expand_as(grad_output)
        grad_input.scatter_add_(0, indices, grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis_residual = IndexFirstAxisResidual.apply


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def unpad_input_for_concatenated_sequences(hidden_states, attention_mask_in_length):
    length = attention_mask_in_length.sum(dim=-1)
    seqlen = attention_mask_in_length.size(-1)
    attention_mask_2d = torch.arange(seqlen, device=length.device, dtype=length.dtype).expand(len(length),
                                                                                              seqlen) < length.unsqueeze(
        1)
    real_indices_idx = torch.nonzero(attention_mask_in_length.flatten(), as_tuple=False).flatten()
    seqlens_in_batch = attention_mask_in_length.flatten()[real_indices_idx]
    indices = torch.nonzero(attention_mask_2d.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )