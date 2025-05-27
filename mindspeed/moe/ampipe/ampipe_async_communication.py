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
import torch

from megatron.training import get_args
from mindspeed.moe.async_comm_utils import async_fw_all_gather, async_all_to_all, async_all_gather


class AsyncCommunication:
    def __init__(self, fwd_args, bwd_args=None):
        self.bwd_args = bwd_args
        self.fwd_args = fwd_args

    def comm_before_moe_mlp_fwd(self, ctx, dispatched_input):
        cur_degree = self.fwd_args.cur_degree
        a2a_events = self.fwd_args.a2a_events
        mlp_inputs = self.fwd_args.mlp_inputs
        a2a_inputs = self.fwd_args.a2a_inputs
        args = get_args()
        pipe_experts = args.use_pipe_experts
        num_experts = args.num_experts
        num_local_experts = num_experts // args.expert_model_parallel_size

        # 不开启ampipe_tp_sp_comm_overlap时，不切分专家维度，直接做全量专家的all2all
        if not args.ampipe_tp_sp_comm_overlap:
            a2a_tokens, a2a_handle = async_all_to_all(dispatched_input)
            a2a_events.append(a2a_handle)
            mlp_inputs[cur_degree] = a2a_tokens
            return mlp_inputs

        # 开启ampipe_tp_sp_comm_overlap时，按照专家切分token后再all2all
        chunk_list = dispatched_input.chunk(num_experts)
        for exp_index in range(num_local_experts):
            chunks = chunk_list[exp_index:num_experts:num_local_experts]
            a2a_tokens = torch.cat(chunks)
            # pipe-experts适配
            if pipe_experts:
                comm_result = self._pipe_expert_comm_before_moe_mlp_fwd(ctx, exp_index, a2a_tokens)
                if comm_result is not None:
                    continue
            # 不开启pipe_experts或者pipe_experts_multi_data < ampipe_degree时不再切分token，直接all2all
            output, a2a_handle = async_all_to_all(a2a_tokens)
            index = cur_degree * num_local_experts + exp_index
            mlp_inputs[index] = output
            a2a_events[index] = a2a_handle
            # 不提前析构通信tensor，保证正常释放通信后tensor内存
            a2a_inputs.append(a2a_tokens)
        return mlp_inputs

    def comm_before_moe_mlp_bwd(self, ctx, grad_moe_out_chunk):
        cur_degree = self.bwd_args.cur_degree
        a2a_events = self.bwd_args.a2a_events
        grad_mlp_input_list = self.bwd_args.grad_mlp_input_list
        grad_a2a_input_list = self.bwd_args.grad_a2a_input_list
        # 反向第一次all2all
        # 纯ep通信隐藏
        if not ctx.ampipe_tp_sp_comm_overlap:
            grad_mlp_input_list[cur_degree], a2a_handle = async_all_to_all(grad_moe_out_chunk)
            a2a_events.insert(0, a2a_handle)
            return grad_mlp_input_list

        # tp-sp域&ep域通信隐藏适配
        chunk_list = grad_moe_out_chunk.chunk(ctx.num_experts)
        for exp_index in range(ctx.num_local_experts):
            chunks = chunk_list[exp_index:ctx.num_experts:ctx.num_local_experts]
            grad_mlp_tokens = torch.cat(chunks)
            # pipe-experts适配
            if ctx.pipe_experts:
                comm_result = self._pipe_expert_comm_before_moe_mlp_bwd(ctx, exp_index, grad_mlp_tokens)
                if comm_result is not None:
                    continue
            # 不开启pipe_experts或者pipe_experts_multi_data < ampipe_degree时不再切分token，直接all2all
            grad_a2a_tokens, a2a_handle = async_all_to_all(grad_mlp_tokens)
            index = (ctx.pipe_degree - 1 - cur_degree) * ctx.num_local_experts + exp_index
            grad_mlp_input_list[index] = grad_a2a_tokens
            a2a_events[index] = a2a_handle
            # 不提前析构通信tensor，保证正常释放通信后tensor内存
            grad_a2a_input_list[index] = grad_mlp_tokens
        return grad_mlp_input_list

    def _pipe_expert_comm_before_moe_mlp_fwd(self, ctx, exp_index, input_tokens):
        cur_degree = self.fwd_args.cur_degree
        a2a_events = self.fwd_args.a2a_events
        mlp_inputs = self.fwd_args.mlp_inputs
        a2a_inputs = self.fwd_args.a2a_inputs
        ag_events = self.fwd_args.ag_events
        args = get_args()
        pipe_degree = args.ampipe_degree
        pipe_experts_multi_data = args.pipe_experts_multi_data
        pipe_experts_multi_stream = args.pipe_experts_multi_stream
        # pipe_experts_multi_data > ampipe_degree时， 对token的C维度再切分
        ctx.slice_size = slice_size = pipe_experts_multi_data // pipe_degree
        a2a_token_chunk = input_tokens.chunk(slice_size, dim=1)
        # 多流场景下pipe_experts_multi_data必须大于等于ampipe_degree
        if pipe_experts_multi_data >= pipe_degree and pipe_experts_multi_stream:
            for i in range(slice_size):
                # 计算列表中索引适配pipe_experts
                index = cur_degree * slice_size + exp_index * pipe_experts_multi_data + i
                if (cur_degree + exp_index + i) == 0 and args.sequence_parallel:
                    a2a_token, a2a_handle = async_all_to_all(a2a_token_chunk[i])
                else:
                    a2a_token, a2a_handle = async_all_to_all(a2a_token_chunk[i], ag_events[index])
                a2a_events[index] = a2a_handle
                mlp_inputs[index] = a2a_token
                if args.sequence_parallel:
                    ag_token, ag_handle = async_fw_all_gather(a2a_token, a2a_handle, ampipe_with_mlp_multistream=True,
                                                              index=index)
                    ag_events[index] = ag_handle
                    mlp_inputs[index] = ag_token
            return mlp_inputs
        # 非多流场景下pipe_experts_multi_data必须大于ampipe_degree
        elif pipe_experts_multi_data > pipe_degree and not pipe_experts_multi_stream:
            for i in range(slice_size):
                a2a_token, a2a_handle = async_all_to_all(a2a_token_chunk[i])
                index = cur_degree * slice_size + exp_index * pipe_experts_multi_data + i
                a2a_events[index] = a2a_handle
                mlp_inputs[index] = a2a_token
                a2a_inputs.append(a2a_token_chunk[i])
            return mlp_inputs
        return None

    def _pipe_expert_comm_before_moe_mlp_bwd(self, ctx, exp_index, grad_tokens):
        cur_degree = self.bwd_args.cur_degree
        a2a_events = self.bwd_args.a2a_events
        grad_mlp_input_list = self.bwd_args.grad_mlp_input_list
        ag_events = self.bwd_args.ag_events
        args = get_args()
        pipe_degree = args.ampipe_degree
        grad_token_list = grad_tokens.chunk(ctx.slice_size, dim=1)
        # 多流场景下pipe_experts_multi_data必须大于等于ampipe_degree
        if ctx.pipe_experts_multi_data >= pipe_degree and ctx.pipe_experts_multi_stream:
            for i in range(ctx.slice_size):
                # 计算列表中索引适配pipe_experts
                index = (pipe_degree - 1 - cur_degree) * ctx.slice_size + exp_index * ctx.pipe_experts_multi_data + i
                if cur_degree == pipe_degree - 1 and (exp_index + i) == 0 and args.sequence_parallel:
                    a2a_token, a2a_handle = async_all_to_all(grad_token_list[i])
                else:
                    a2a_token, a2a_handle = async_all_to_all(grad_token_list[i], ag_events[index])
                a2a_events[index] = a2a_handle
                grad_mlp_input_list[index] = a2a_token
                if args.sequence_parallel:
                    ag_token, ag_handle = async_all_gather(a2a_token, a2a_handle, is_bwd=True)
                    ag_events[index] = ag_handle
                    grad_mlp_input_list[index] = ag_token
            return grad_mlp_input_list
        # 非多流场景下pipe_experts_multi_data必须大于ampipe_degree
        elif ctx.pipe_experts_multi_data > pipe_degree and not ctx.pipe_experts_multi_stream:
            for i in range(ctx.slice_size):
                a2a_token, a2a_handle = async_all_to_all(grad_token_list[i])
                index = (pipe_degree - 1 - cur_degree) * ctx.slice_size + exp_index * ctx.pipe_experts_multi_data + i
                a2a_events[index] = a2a_handle
                grad_mlp_input_list[index] = a2a_token
            return grad_mlp_input_list
        return None

    def fw_all_gather_not_multistream(self):
        self.fwd_args.a2a_events[0].wait()
        # 释放通信内存
        self.fwd_args.a2a_inputs.pop()
        _, ag_handle = async_fw_all_gather(self.fwd_args.mlp_inputs[0])
        self.fwd_args.ag_events.append(ag_handle)
