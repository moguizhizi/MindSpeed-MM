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
from mindspeed.moe.ampipe.ampipe_args import MLPSaveForBwdArgs
from mindspeed.moe.async_comm_utils import async_all_to_all, async_all_gather, async_fw_all_gather, \
    async_fw_all_reduce_scatter_ampipe, get_fw_ar_rs_output_ampipe
from mindspeed.moe.pipe_experts import PipeExpert


class MoEMLPComputer:
    def __init__(self, moe, save_tensor_list, mlp_fwd_args, mlp_bwd_args=None):
        super().__init__()
        self.mlp_bwd_args = mlp_bwd_args
        self.mlp_fwd_args = mlp_fwd_args
        self.save_tensor_list = save_tensor_list
        self.moe = moe

    def forward(self, ctx, mlp_inputs, a2a_inputs):
        global_args = get_args()
        mlp_save_for_bwd_args = MLPSaveForBwdArgs()
        a2a_events = self.mlp_fwd_args.a2a_events
        ag_events = self.mlp_fwd_args.ag_events
        pipe_degree = global_args.ampipe_degree
        sequence_parallel = global_args.sequence_parallel
        num_local_experts = global_args.num_experts // global_args.expert_model_parallel_size
        ep_size = global_args.expert_model_parallel_size
        hidden_size = global_args.hidden_size
        pipe_experts = global_args.use_pipe_experts
        multi_data = global_args.pipe_experts_multi_data
        multi_stream = global_args.pipe_experts_multi_stream

        ctx.use_ampipe_with_pipe_expert = (pipe_experts and
                                           (multi_data >= pipe_degree and multi_stream)
                                           or (multi_data > pipe_degree and not multi_stream))
        if ctx.use_ampipe_with_pipe_expert:
            second_a2a_event = []
            pipe_expert_args = [mlp_inputs, ep_size, num_local_experts, sequence_parallel, multi_data, multi_stream,
                                a2a_events, second_a2a_event, ag_events, hidden_size, self.save_tensor_list]
            mlp_outputs = PipeExpert.forward(mlp_save_for_bwd_args, self.moe.moe_layer.experts, *pipe_expert_args)
            ctx.mlp_args = mlp_save_for_bwd_args
        elif global_args.ampipe_tp_sp_comm_overlap:
            mlp_outputs = self.ampipe_experts_forward(mlp_save_for_bwd_args, mlp_inputs, a2a_inputs)
            ctx.mlp_args = mlp_save_for_bwd_args
        else:
            mlp_outputs = []
            for c in range(pipe_degree):
                a2a_events.pop(0).wait()
                expert_input = mlp_inputs[c].reshape(ep_size, num_local_experts, -1, hidden_size)
                detach_expert_input = expert_input.detach()
                detach_expert_input.requires_grad = True
                with torch.enable_grad():
                    expert_output = self.moe.moe_layer.experts(detach_expert_input)
                    self.save_tensor_list.extend([detach_expert_input, expert_output])
                mlp_inputs[c] = expert_output
                a2a_tokens, a2a_handle = async_all_to_all(expert_output)
                a2a_events.append(a2a_handle)
                mlp_outputs.append(a2a_tokens)
        return mlp_outputs

    def backward(self, ctx, grad_mlp_input_list, grad_a2a_input_list):
        a2a_events = self.mlp_bwd_args.a2a_events
        ag_events = self.mlp_bwd_args.ag_events
        mlp_tensor_list = self.mlp_bwd_args.mlp_tensor_list
        mlp_bwd_grads = []
        multi_stream = ctx.pipe_experts_multi_stream
        # 适配pipe-experts
        if ctx.use_ampipe_with_pipe_expert:
            if self.mlp_bwd_args.sequence_parallel and not multi_stream:
                a2a_events[0].wait()
                grad_a2a_input_list.pop(0)
                grad_mlp_input_list[0], ag_handle = async_all_gather(grad_mlp_input_list[0], is_bwd=True)
                ag_events.append(ag_handle)
            mlp_bwd_grads = PipeExpert.backward(ctx.mlp_args, grad_mlp_input_list, a2a_events, ag_events,
                                                self.mlp_bwd_args.second_a2a_events, mlp_tensor_list)
        # mlp反向tp-sp&ep通信隐藏流水实现
        elif ctx.ampipe_tp_sp_comm_overlap:
            if self.mlp_bwd_args.sequence_parallel:
                a2a_events[0].wait()
                grad_a2a_input_list.pop(0)
                grad_mlp_input_list[0], ag_handle = async_all_gather(grad_mlp_input_list[0], is_bwd=True)
                ag_events.append(ag_handle)
            mlp_bwd_grads = self.ampipe_experts_backward(ctx.mlp_args, mlp_tensor_list, grad_mlp_input_list,
                                                         grad_a2a_input_list, a2a_events, ag_events)
        # mlp反向纯ep通信隐藏流水实现
        else:
            mlp_list_slice_len = self.mlp_bwd_args.mlp_tensor_list_len // ctx.pipe_degree
            for c in range(ctx.pipe_degree - 1, -1, -1):
                a2a_events.pop().wait()
                expert_input, expert_output = mlp_tensor_list[c * mlp_list_slice_len:(c + 1) * mlp_list_slice_len]
                expert_output.backward(grad_mlp_input_list[c])
                grad_mlp_input = expert_input.grad.reshape(self.moe.num_experts, -1, self.moe.hidden_size)
                a2a_grad_mlp_input, a2a_handle = async_all_to_all(grad_mlp_input)
                mlp_bwd_grads.insert(0, a2a_grad_mlp_input)
                a2a_events.insert(0, a2a_handle)
        mlp_tensor_list.clear()
        return mlp_bwd_grads

    def ampipe_experts_forward(self, ctx, inputs, a2a_inputs):
        ctx.ampipe_degree = pipe_degree = get_args().ampipe_degree
        ctx.ep_size = ep_size = get_args().expert_model_parallel_size
        ctx.num_local_experts = num_local_experts = get_args().num_experts // ep_size
        ctx.hidden_size = hidden_size = get_args().hidden_size
        ctx.sequence_parallel = sequence_parallel = get_args().sequence_parallel
        ag_events = self.mlp_fwd_args.ag_events
        a2a_events = self.mlp_fwd_args.a2a_events

        output_list = []
        before_exp_input_list = []
        after_exp_out_list = []

        for c in range(pipe_degree):
            for i in range(num_local_experts):
                cur_index = c * num_local_experts + i
                # pre expert process
                if sequence_parallel:
                    ag_events[cur_index].wait()
                    if cur_index < num_local_experts * pipe_degree - 1:
                        a2a_events[cur_index + 1].wait()
                        a2a_inputs.pop()
                        _, ag_handle = async_fw_all_gather(inputs[cur_index + 1],
                                                           is_use_global_memory_buffer=False)
                        ag_events.append(ag_handle)
                else:
                    a2a_events[cur_index].wait()
                    a2a_inputs.pop()
                # expert compute
                detach_input_chunk = inputs[cur_index].detach()
                detach_input_chunk.requires_grad = True
                before_exp_input_list.append(detach_input_chunk)
                with torch.enable_grad():
                    out = self.moe.moe_layer.experts.experts[i](detach_input_chunk)
                if isinstance(out, tuple):
                    if cur_index > 0:
                        out, last_chunk_out = out[0], out[-1]
                    else:
                        out = out[0]  # Ignore the bias term for now

                # post expert comm
                async_fw_all_reduce_scatter_ampipe(out, sequence_parallel)
                after_exp_out_list.append(out)
                if cur_index > 0:
                    after_exp_out_list[cur_index - 1].untyped_storage().resize_(0)
                    output_list.append(last_chunk_out)
                if cur_index == pipe_degree * num_local_experts - 1:
                    ar_rs_out = get_fw_ar_rs_output_ampipe(sequence_parallel)
                    a2a_out, a2a2_handle = async_all_to_all(ar_rs_out)
                    a2a2_handle.wait()
                    output_list.append(a2a_out)

        for t in after_exp_out_list:
            t.untyped_storage().resize_(0)
        self.save_tensor_list.extend(before_exp_input_list)
        self.save_tensor_list.extend(after_exp_out_list)
        outputs = []
        for c in range(pipe_degree):
            cur_pipe_out_list = output_list[c * num_local_experts:(c + 1) * num_local_experts]
            cur_pipe_out = torch.cat(cur_pipe_out_list, dim=1)
            cur_pipe_out = cur_pipe_out.reshape((num_local_experts * ep_size), -1, hidden_size)
            outputs.append(cur_pipe_out)
        return outputs

    def ampipe_experts_backward(self, ctx, saved_tensor_list, *args):
        pipe_degree = ctx.ampipe_degree
        num_local_experts = ctx.num_local_experts
        ep_size = ctx.ep_size
        hidden_size = ctx.hidden_size
        sequence_parallel = ctx.sequence_parallel

        before_exp_input_list = saved_tensor_list[:num_local_experts * pipe_degree]
        after_exp_out_list = saved_tensor_list[num_local_experts * pipe_degree:]
        grad_output_list, grad_a2a_input_list, a2a_event, ag_events = args
        grad_a2a2_input_list = []
        output_list = []

        for c in range(pipe_degree - 1, -1, -1):
            for i in range(num_local_experts):
                reversed_index = c * num_local_experts + i
                normal_index = (pipe_degree - c - 1) * num_local_experts + i
                # pre expert process
                if sequence_parallel:
                    ag_events[normal_index].wait()
                    if normal_index < num_local_experts * pipe_degree - 1:
                        a2a_event[normal_index + 1].wait()
                        grad_a2a_input_list.pop(0)
                        grad_output = grad_output_list[normal_index + 1]
                        ag_grad_output, ag_handle = async_all_gather(grad_output, is_bwd=True)
                        grad_output_list[normal_index + 1] = ag_grad_output
                        ag_events.append(ag_handle)
                else:
                    a2a_event[normal_index].wait()
                    grad_a2a_input_list.pop(0)
                # expert backward compute
                mlp_grad_output = grad_output_list[normal_index]
                after_exp_out_list[reversed_index].backward(mlp_grad_output)
                grad_input = before_exp_input_list[reversed_index].grad
                mlp_grad_output.untyped_storage().resize_(0)
                before_exp_input_list[reversed_index].untyped_storage().resize_(0)
                # post expert process
                a2a_grad_input, a2a1_handle = async_all_to_all(grad_input)
                output_list.append(a2a_grad_input)
                grad_a2a2_input_list.append(grad_input)
                if normal_index > 0:
                    a2a_event[-1].wait()
                    grad_a2a2_input_list.pop(0)
                a2a_event.append(a2a1_handle)

        outputs = []
        for c in range(pipe_degree):
            cur_pipe_out_list = output_list[c * num_local_experts:(c + 1) * num_local_experts]
            if c == pipe_degree - 1:
                outputs.insert(0, cur_pipe_out_list)
                continue
            cur_pipe_out = torch.cat(cur_pipe_out_list, dim=1)
            cur_pipe_out = cur_pipe_out.reshape((num_local_experts * ep_size), -1, hidden_size)
            outputs.insert(0, cur_pipe_out)
        return outputs
