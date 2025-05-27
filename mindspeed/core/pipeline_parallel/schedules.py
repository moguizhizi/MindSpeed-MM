# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
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
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reversed.
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import contextlib
import time
import torch
from functools import wraps
from megatron.core.enums import ModelType
from megatron.training import get_args
from megatron.core import parallel_state
from megatron.core.utils import get_attr_wrapped_model, get_model_config, get_model_type
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.pipeline_parallel.schedules import set_current_microbatch
from mindspeed.core.pipeline_parallel import flexible_schedules
from mindspeed.core.pipeline_parallel.ripipe_schedules import forward_backward_ripipe_pipelining
from mindspeed.core.pipeline_parallel import multiparameter_schedules
from mindspeed.core.auto_parallel.mm_search.help import PROFILE_CONTENT

LOSS_BACKWARD_SCALE = torch.tensor(1.0)


def get_forward_backward_func_wrapper(get_forward_backward_func):
    @wraps(get_forward_backward_func)
    def wrapper(*args, **kwargs):
        arguments = get_args()
        if arguments.optimize_send_recv_comm and arguments.num_layers_per_virtual_pipeline_stage is None:
            return flexible_schedules.forward_backward_pipelining_without_interleaving

        if arguments.automated_pipeline_perf and arguments.pp_schedule_list:
            return flexible_schedules.forward_backward_pipelining_without_interleaving

        if (arguments.recompute_in_bubble or arguments.recompute_in_advance) and torch.is_grad_enabled():
            return forward_backward_ripipe_pipelining

        if parallel_state.get_pipeline_model_parallel_world_size() > 1 \
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None \
            and arguments.use_nanopipe:
            return flexible_schedules.forward_backward_pipelining_with_interleaving_nano_pipe

        if arguments.use_multiparameter_pipeline_model_parallel:
            pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
            if pipeline_model_parallel_size > 1 \
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                return multiparameter_schedules.forward_backward_pipelining_with_interleaving

        return get_forward_backward_func(*args, **kwargs)
    return wrapper


def forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data=False,
        checkpoint_activations_microbatch=None,
        is_first_microbatch=False,
        current_microbatch=None,
):

    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    arguments = get_args()
    if arguments.auto_parallel_profile:
        torch.cuda.synchronize()
        start_time = time.time()
        torch.npu.reset_max_memory_allocated()
        start_memory = torch.npu.memory_allocated()

    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()
    if current_microbatch is not None:
        set_current_microbatch(model, current_microbatch)

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        if checkpoint_activations_microbatch is None:
            output_tensor, loss_func = forward_step_func(data_iterator, model)
        else:
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )

    num_tokens = torch.tensor(0, dtype=torch.int)
    if parallel_state.is_pipeline_last_stage():
        if not collect_non_loss_data:
            outputs = loss_func(output_tensor)
            if len(outputs) == 3:
                output_tensor, num_tokens, loss_reduced = outputs
                if not config.calculate_per_token_loss:
                    output_tensor /= num_tokens
                    output_tensor /= num_microbatches
            else:
                # preserve legacy loss averaging behavior (ie, over the number of microbatches)
                assert len(outputs) == 2
                output_tensor, loss_reduced = outputs
                output_tensor /= num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    # Set the loss scale for the auxiliary loss of the MoE layer.
    # Since we use a trick to do backward on the auxiliary loss, we need to set the scale explicitly.
    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(LOSS_BACKWARD_SCALE)
            if config.grad_scale_func is not None
            else torch.tensor(1.0)
        )
        # Set the loss scale
        MoEAuxLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    if arguments.auto_parallel_profile:
        torch.cuda.synchronize()
        PROFILE_CONTENT["fwd_time"].append((time.time() - start_time) * 1000)                                      # time: ms
        PROFILE_CONTENT["act_mem"].append((torch.npu.memory_allocated() - start_memory) / 1024 ** 2)        # memory: MB
        max_mem = torch.npu.max_memory_allocated()

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)
    if (
            parallel_state.is_pipeline_stage_after_split()
            and model_type == ModelType.encoder_and_decoder
    ):
        return [output_tensor, input_tensor[-1]], num_tokens

    if unwrap_output_tensor:
        return output_tensor, num_tokens
    return [output_tensor], num_tokens
 
 
def get_tensor_shapes_wrapper(get_tensor_shapes):
    @wraps(get_tensor_shapes)
    def wrapper(*args, **kwargs):
        # [s, b, h]
        tensor_shapes = get_tensor_shapes(*args, **kwargs)
        arguments = get_args()
        if arguments.tp_2d:
            tensor_shapes = [[tensor_shape[0] // arguments.tp_x, tensor_shape[1], tensor_shape[2] // arguments.tp_y]
                             for tensor_shape in tensor_shapes]
 
        return tensor_shapes
    return wrapper
