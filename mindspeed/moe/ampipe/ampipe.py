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
import itertools
from collections import namedtuple

import torch
from einops import rearrange

from megatron.training import get_args
from mindspeed.moe.ampipe.ampipe_args import (ForwardArgs, FlashAttentionFwdArgs, FwdCommArgs, BiasDropoutAddNormArgs,
                                              MLPFwdArgs, PostMLPArgs, BwdCommArgs, FlashAttentionBwdArgs, MLPBwdArgs)
from mindspeed.moe.ampipe.ampipe_async_communication import AsyncCommunication
from mindspeed.moe.ampipe.ampipe_bias_dropout_add_ln_computer import BiasDropoutAddNormComputer
from mindspeed.moe.ampipe.ampipe_fa_computer import FlashAttentionComputer
from mindspeed.moe.ampipe.ampipe_moe_gating_computer import MoEGatingComputer
from mindspeed.moe.ampipe.ampipe_moe_mlp_computer import MoEMLPComputer
from mindspeed.moe.ampipe.ampipe_post_mlp_computer import MoEPostMLPComputer
from mindspeed.moe.async_comm_utils import get_async_comm_utils_data_instance


class AttMoEPipe(torch.autograd.Function):
    """
    Ampipe autograd.Function Class

    Include FlashAttention & LayerNorm & MoE Layer
    Args:
        q: query
        k: key
        v: value
        hidden_states: hidden_states before transformer layer used as residual.
        attention_mask: global attention mask.
        attention_dense: post attention dense layer object.
        bias_dropout_add_func: bias dropout add function
        post_attention_norm: post attention norm object.
        moe: moe layer object.
        hidden_dropout: dropout prob.
    """
    @staticmethod
    def forward(ctx, q, k, v, hidden_states, attention_mask, ampipe_forward_args: ForwardArgs):
        attention_dense = ampipe_forward_args.attention_dense
        bias_dropout_add_func = ampipe_forward_args.bias_dropout_add_func
        post_attention_norm = ampipe_forward_args.post_attention_norm
        moe = ampipe_forward_args.moe
        hidden_dropout = ampipe_forward_args.hidden_dropout

        global_args = get_args()
        pipe_degree = global_args.ampipe_degree
        AttMoEPipe.save_args_to_ctx(ctx, ampipe_forward_args, global_args)

        # 初始化反向保存tensor列表
        flash_tensor_list = []
        dense_tensor_list = []
        bdal_tensor_list = []
        gate_tensor_list = []
        mlp_tensor_list = []
        post_mlp_tensor_list = []

        # 初始化临时列表
        ln_input_list = []
        moe_output_list = []
        weights_list = [None] * pipe_degree
        token_ec_idx_list = [None] * pipe_degree
        mlp_inputs, a2a_inputs, a2a_events, ag_events = AttMoEPipe._init_fwd_comm_list()

        # 初始化attention相关变量
        q_shape = q.shape
        ctx.head = q_shape[2]
        q = rearrange(q, "s b n d -> s b (n d)")
        fa_fwd_args = AttMoEPipe._init_attention_args(pipe_degree, q_shape, attention_dense, flash_tensor_list)
        # 切分残差以及bias
        hidden_states_chunks = hidden_states.chunk(pipe_degree, dim=0)
        bias_chunks = attention_dense.bias.chunk(pipe_degree, dim=0) if attention_dense.bias is not None else None
        ln_seq_len = hidden_states.shape[0]
        ctx.fa_computer = fa_computer = FlashAttentionComputer(fa_fwd_args)
        for c in range(pipe_degree):
            # Attention(FA)
            fa_fwd_args.cur_degree = c
            fwd_comm_args = FwdCommArgs(c, mlp_inputs, a2a_inputs, a2a_events, ag_events)
            ctx.async_comm = async_comm = AsyncCommunication(fwd_comm_args)
            detach_attn_out, attn_out, attn_bias = fa_computer.forward(ctx, q, k, v, attention_mask)
            fa_fwd_args.q_token_start_idx += fa_fwd_args.chunk_len

            # Bias + Dropout + Add + LN
            bias_chunk = bias_chunks[c] if attention_dense.bias is not None else None
            bdal_fwd_args = BiasDropoutAddNormArgs(bias_dropout_add_func, post_attention_norm,
                                                   hidden_states_chunks[c], bias_chunk, hidden_dropout)
            ctx.bdal_computer = bdal_computer = BiasDropoutAddNormComputer(bdal_tensor_list, bdal_fwd_args)
            ln_output, ln_input = bdal_computer.forward(ctx, attn_out)
            attn_out.untyped_storage().resize_(0)
            dense_tensor_list.append(detach_attn_out)
            dense_tensor_list.append(attn_out)
            ln_input_list.append(ln_input)

            # MoE Gating以及token重排
            ctx.gate_computer = gate_computer = MoEGatingComputer(moe, gate_tensor_list)
            gate_output = gate_computer.forward(ln_output)
            if global_args.enable_token_rearrange_opt:
                dispatched_input, l_aux, token_ec_idx_list[c], weights_list[c] = gate_output
            else:
                dispatched_input, l_aux, weights_list[c] = gate_output
            ln_output.untyped_storage().resize_(0)
            bdal_tensor_list.append(ln_output)

            # mlp前第一次all2all以及allgather通信
            mlp_inputs = async_comm.comm_before_moe_mlp_fwd(ctx, dispatched_input)
            dispatched_input.untyped_storage().resize_(0)
            gate_tensor_list.append(dispatched_input)

        # MoE MLP
        mlp_fwd_args = MLPFwdArgs(a2a_events, ag_events)
        ctx.mlp_computer = mlp_computer = MoEMLPComputer(moe, mlp_tensor_list, mlp_fwd_args)
        mlp_outputs = mlp_computer.forward(ctx, mlp_inputs, a2a_inputs)

        # token反重排
        post_mlp_fwd_args = PostMLPArgs(ln_seq_len // pipe_degree, a2a_events,
                                        moe_output_list, weights_list, token_ec_idx_list)
        ctx.post_mlp_computer = post_mlp_computer = MoEPostMLPComputer(post_mlp_tensor_list, post_mlp_fwd_args)
        moe_output_list = post_mlp_computer.forward(ctx, mlp_outputs)
        AttMoEPipe.save_tensors_for_bwd(ctx, [flash_tensor_list, dense_tensor_list, bdal_tensor_list,
                                              gate_tensor_list, mlp_tensor_list, post_mlp_tensor_list])
        ret = torch.cat(moe_output_list), torch.cat(ln_input_list)
        return ret

    @staticmethod
    def backward(ctx, grad_moe_outs, grad_ln_ins):
        global_args = get_args()
        pipe_degree = ctx.pipe_degree
        context_parallel = global_args.context_parallel_size > 1
        sequence_parallel = global_args.sequence_parallel

        # 取前向保存的tensor
        saved_tensors_list = list(ctx.saved_tensors)
        (flash_tensor_list_len, dense_tensor_list_len,
         bdal_tensor_list_len, gate_tensor_list_len,
         mlp_tensor_list_len, post_mlp_tensor_list_len) = ctx.tensor_list_length
        start_index = 0
        segments = []

        for length in ctx.tensor_list_length:
            end_index = start_index + length
            segments.append(saved_tensors_list[start_index:end_index])
            start_index = end_index
        (flash_tensor_list, dense_tensor_list,
         bdal_tensor_list, gate_tensor_list,
         mlp_tensor_list, post_mlp_tensor_list) = segments

        # 切分传入backward的grad
        grad_moe_out_list = grad_moe_outs.chunk(pipe_degree)
        grad_ln_ins_list = grad_ln_ins.chunk(pipe_degree)
        # 初始化临时变量
        grad_hidden, grad_q, grad_k, grad_v = [], [], None, None
        grad_mlp_input_list, grad_a2a_input_list, a2a_events, ag_events = AttMoEPipe._init_bwd_comm_list(ctx)

        for c in range(pipe_degree - 1, -1, -1):
            # 计算token反重排的反向
            grad_moe_out_chunk = grad_moe_out_list[c].view(-1, ctx.hidden_size)
            post_mlp_list_slice_len = post_mlp_tensor_list_len // pipe_degree
            grad_post_mlp = ctx.post_mlp_computer.backward(
                post_mlp_tensor_list[c * post_mlp_list_slice_len:(c + 1) * post_mlp_list_slice_len],
                grad_moe_out_chunk
            )
            # 反向第一次all2all以及allgather通信
            bwd_comm_args = BwdCommArgs(c, grad_mlp_input_list, grad_a2a_input_list, a2a_events, ag_events)
            ctx.async_comm.bwd_args = bwd_comm_args
            grad_mlp_input_list = ctx.async_comm.comm_before_moe_mlp_bwd(ctx, grad_post_mlp)
            del post_mlp_tensor_list[c * post_mlp_list_slice_len:(c + 1) * post_mlp_list_slice_len]
        # 手动清理ctx中computer保存的tensor，以减少峰值内存
        ctx.post_mlp_computer = None
        ctx.async_comm = None
        # 专家mlp反向计算
        bwd_mlp_args = MLPBwdArgs(sequence_parallel, mlp_tensor_list_len, a2a_events, ag_events, mlp_tensor_list)
        if ctx.pipe_experts:
            bwd_mlp_args.second_a2a_events = []
        ctx.mlp_computer.mlp_bwd_args = bwd_mlp_args
        mlp_bwd_grads = ctx.mlp_computer.backward(ctx, grad_mlp_input_list, grad_a2a_input_list)
        # 手动清理ctx中computer保存的tensor，以减少峰值内存
        ctx.mlp_computer = None

        fa_bwd_args = FlashAttentionBwdArgs(grad_q, grad_k, grad_v, flash_tensor_list, dense_tensor_list,
                                            flash_tensor_list_len=flash_tensor_list_len,
                                            dense_tensor_list_len=dense_tensor_list_len)
        if context_parallel:
            fa_bwd_args.kv_list = []
            fa_bwd_args.dkv_list = []
            fa_bwd_args.dout_list = []
        else:
            fa_bwd_args.v = flash_tensor_list.pop()
            fa_bwd_args.k = flash_tensor_list.pop()
        ctx.fa_computer.fa_bwd_args = fa_bwd_args
        for c in range(pipe_degree - 1, -1, -1):
            # 反向等待最后一次all2all
            grad_mlp = AttMoEPipe.bwd_second_all2all_wait_last(ctx, c, mlp_bwd_grads, a2a_events, bwd_mlp_args)
            # gating&token重排反向
            gate_list_slice_len = gate_tensor_list_len // pipe_degree
            grad_ln_out = ctx.gate_computer.backward(
                gate_tensor_list[c * gate_list_slice_len:(c + 1) * gate_list_slice_len],
                grad_mlp
            )
            del gate_tensor_list[c * gate_list_slice_len:(c + 1) * gate_list_slice_len]

            # bias dropout add ln 反向
            bdal_list_slice_len = bdal_tensor_list_len // pipe_degree
            bdal_list_slice = bdal_tensor_list[c * bdal_list_slice_len:(c + 1) * bdal_list_slice_len]
            grad_dense, d_hidden_grad, d_bias_grad = ctx.bdal_computer.backward(ctx, bdal_list_slice,
                                                                                grad_ln_out, grad_ln_ins_list[c])
            grad_hidden.insert(0, d_hidden_grad)
            del bdal_list_slice
            del bdal_tensor_list[c * bdal_list_slice_len:(c + 1) * bdal_list_slice_len]

            # fa反向
            fa_bwd_args.cur_degree = c
            grad_q, grad_k, grad_v = ctx.fa_computer.backward(ctx, grad_dense)
        # 手动清理ctx中computer保存的tensor，以减少峰值内存
        ctx.gate_computer = None
        ctx.bdal_computer = None
        ctx.fa_computer = None
        if not context_parallel:
            grad_q = torch.cat(grad_q, dim=0)
        grad_q = rearrange(grad_q, "s b (n d) -> s b n d", n=ctx.head)
        return grad_q, grad_k, grad_v, torch.cat(grad_hidden), None, None

    @staticmethod
    def save_args_to_ctx(ctx, ampipe_forward_args, global_args):
        ctx.ampipe_forward_args = ampipe_forward_args
        ctx.sequence_parallel = global_args.sequence_parallel
        ctx.num_experts = global_args.num_experts
        ctx.num_local_experts = global_args.num_experts // global_args.expert_model_parallel_size
        ctx.ep_size = global_args.expert_model_parallel_size
        ctx.hidden_size = global_args.hidden_size
        ctx.pipe_degree = global_args.ampipe_degree
        ctx.ampipe_tp_sp_comm_overlap = global_args.ampipe_tp_sp_comm_overlap
        ctx.pipe_experts = global_args.use_pipe_experts
        ctx.pipe_experts_multi_data = global_args.pipe_experts_multi_data
        ctx.pipe_experts_multi_stream = global_args.pipe_experts_multi_stream
        ctx.flash_args = []
        ctx.mlp_args = []

    @staticmethod
    def save_tensors_for_bwd(ctx, tensor_list):
        flat_list = itertools.chain.from_iterable(tensor_list)
        ctx.save_for_backward(*flat_list)
        ctx.tensor_list_length = [len(x) for x in tensor_list]
        for lst in tensor_list:
            lst.clear()

    @staticmethod
    def _init_attention_args(pipe_degree, q_shape, attention_dense, flash_tensor_list):
        seqlen, batch_size, head_num, head_dim = q_shape
        chunk_len = seqlen // pipe_degree
        softmax_scale = head_dim ** (-0.5)
        return FlashAttentionFwdArgs(flash_tensor_list, attention_dense, head_num, softmax_scale, chunk_len)

    @staticmethod
    def bwd_second_all2all_wait_last(ctx, cur_degree, mlp_bwd_grads, a2a_events, mlp_bwd_args):
        grad_mlp_last = mlp_bwd_grads[cur_degree]
        if ctx.use_ampipe_with_pipe_expert and cur_degree == 0:
            mlp_bwd_args.second_a2a_events[-1].wait()
            grad_combine = torch.cat([torch.cat(i, dim=1) for i in grad_mlp_last], dim=1)
            grad_mlp_last = grad_combine.reshape(ctx.num_experts, -1, ctx.hidden_size)
        elif ctx.ampipe_tp_sp_comm_overlap and cur_degree == 0:
            a2a_events[-1].wait()
            grad_combine = torch.cat(grad_mlp_last, dim=1)
            grad_mlp_last = grad_combine.reshape(ctx.num_experts, -1, ctx.hidden_size)

        if not ctx.ampipe_tp_sp_comm_overlap:
            a2a_events[cur_degree].wait()
        return grad_mlp_last

    @staticmethod
    def _init_fwd_comm_list():
        global_args = get_args()
        pipe_degree = global_args.ampipe_degree
        num_local_experts = global_args.num_experts // global_args.expert_model_parallel_size
        pipe_experts_multi_data = global_args.pipe_experts_multi_data
        pipe_experts_multi_stream = global_args.pipe_experts_multi_stream
        a2a_inputs = []
        ag_events = []

        if not global_args.ampipe_tp_sp_comm_overlap:
            mlp_inputs = [None] * pipe_degree
            a2a_events = []
        elif not global_args.use_pipe_experts or pipe_experts_multi_data <= pipe_degree:
            mlp_inputs = [None] * (pipe_degree * num_local_experts)
            a2a_events = [None] * (pipe_degree * num_local_experts)
        else:
            mlp_inputs = [None] * (pipe_experts_multi_data * num_local_experts)
            a2a_events = [None] * (pipe_experts_multi_data * num_local_experts)

        if pipe_experts_multi_stream:
            ag_events = [None] * (pipe_experts_multi_data * num_local_experts)
            get_async_comm_utils_data_instance().fw_ag_output = [None] * (pipe_experts_multi_data * num_local_experts)
        CommList = namedtuple("CommList", ["mlp_inputs", "a2a_inputs", "a2a_events", "ag_events"])
        comm_list = CommList(mlp_inputs, a2a_inputs, a2a_events, ag_events)
        return comm_list

    @staticmethod
    def _init_bwd_comm_list(ctx):
        if not ctx.ampipe_tp_sp_comm_overlap:
            grad_mlp_input_list = [None] * ctx.pipe_degree
            grad_a2a_input_list = [None] * ctx.pipe_degree
            a2a_events = []
        elif not ctx.pipe_experts or ctx.pipe_experts_multi_data <= ctx.pipe_degree:
            grad_mlp_input_list = [None] * (ctx.pipe_degree * ctx.num_local_experts)
            grad_a2a_input_list = [None] * (ctx.pipe_degree * ctx.num_local_experts)
            a2a_events = [None] * (ctx.pipe_degree * ctx.num_local_experts)
        else:
            grad_mlp_input_list = [None] * (ctx.pipe_experts_multi_data * ctx.num_local_experts)
            grad_a2a_input_list = [None] * (ctx.pipe_experts_multi_data * ctx.num_local_experts)
            a2a_events = [None] * (ctx.pipe_experts_multi_data * ctx.num_local_experts)

        ag_events = []
        if ctx.pipe_experts_multi_stream:
            ag_events = [None] * (ctx.pipe_experts_multi_data * ctx.num_local_experts)
        CommList = namedtuple("CommList", ["mlp_inputs", "a2a_inputs", "a2a_events", "ag_events"])
        comm_list = CommList(grad_mlp_input_list, grad_a2a_input_list, a2a_events, ag_events)
        return comm_list
