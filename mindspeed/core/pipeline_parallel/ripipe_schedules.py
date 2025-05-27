# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import collections
from typing import Iterator, List, Union
import contextlib

import torch
from megatron.training import get_args
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import deallocate_output_tensor, forward_step, backward_step, \
    check_first_val_step
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.utils import get_model_config, get_model_type
from megatron.core.enums import ModelType

from mindspeed.core.tensor_parallel.checkpoint_manager import get_pipeline_checkpoint_manager
from mindspeed.core.weight_grad_store import WeightGradStore


def forward_backward_ripipe_pipelining(
        *,
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: int = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
        first_val_step: bool = None,
):
    """Almost directly copied from megatron's forward_backward_pipelining_with_interleaving
    function, all modifications are annotated with 'ripipe related' or 'nanopipe related' """
    # ripipe related, setup checkpoint manager.
    pipeline_checkpoint_manager = get_pipeline_checkpoint_manager(
        num_of_chunks=parallel_state.get_virtual_pipeline_model_parallel_world_size())
    args = get_args()
    if args.recompute_in_bubble or args.recompute_in_advance:
        pipeline_checkpoint_manager.open_ri_pipe = True
        pipeline_checkpoint_manager.do_pre_recompute = True


    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    assert isinstance(model, list), "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
    assert isinstance(
        data_iterator, list
    ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"

    config = get_model_config(model[0])
    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in config.no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    forward_data_store = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    if num_microbatches % pipeline_parallel_size != 0:
        msg = f'number of microbatches ({num_microbatches}) is not divisible by '
        msg += f'pipeline-model-parallel-size ({pipeline_parallel_size}) '
        msg += 'when using interleaved schedule'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Interleaving is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != seq_length:
        raise RuntimeError(
            "Interleaving is not supported with a different decoder sequence length."
        )

    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // parallel_state.get_context_parallel_world_size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()
    tensor_shape[0] = tensor_shape[0] // args.tp_x
    tensor_shape[-1] = tensor_shape[-1] // args.tp_y
    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    total_num_microbatches = num_microbatches * num_model_chunks
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    else:
        # ripipe related, no special handling of 'num_warmup_microbatches' when 'num_microbatches == pipeline_parallel_size'
        num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
        num_warmup_microbatches += (num_model_chunks - 1) * pipeline_parallel_size
        num_warmup_microbatches = min(num_warmup_microbatches, total_num_microbatches)
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    num_fwd = min((pipeline_parallel_size - 1) * 2 + (num_model_chunks - 1) * pipeline_parallel_size, total_num_microbatches)
    num_dx = num_fwd - num_warmup_microbatches
    overlap_chunks_num = (num_dx + pipeline_parallel_size - 1) // pipeline_parallel_size 
    nano_flag = [True] * len(model)
    for i in range(overlap_chunks_num):
        nano_flag[-i - 1] = False
    # ripipe related, calculate the variables needed by the recompute_in_bubble function
    num_microbatches_recompute, num_microbatches_recompute_forward, num_microbatches_recompute_steady_groups, \
        num_microbatches_recompute_tail = get_ripipe_recompute_count_params(num_microbatches,
                                                                            num_model_chunks,
                                                                            num_warmup_microbatches)

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func[0](model[0].parameters())
        config.param_sync_func[1](model[1].parameters())

    def get_chunk_batch_id(microbatch_id, forward):
        """ripipe related, needed by recompute_in_bubble function."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        group_id = microbatch_id // (pipeline_parallel_size * num_model_chunks)
        intra_chunk_batch_id = (microbatch_id_in_group % pipeline_parallel_size)
        return group_id, intra_chunk_batch_id, model_chunk_id

    def should_recompute(fk):
        """ripipe related, needed by recompute_in_bubble function, used to determine
        whether a mircobatch needs to be recomputed in the 1f1b stage."""
        gid, intro_group_bid, chunk_id = get_chunk_batch_id(fk, forward=True)
        if chunk_id == 0:
            if gid < 2:
                return False
            elif gid < 2 + num_microbatches_recompute_steady_groups:
                if intro_group_bid >= (1 + 2 * pipeline_parallel_rank):
                    return True
            else:
                if intro_group_bid >= pipeline_parallel_size - num_microbatches_recompute_tail:
                    return True
        return False

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def is_first_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == 0:
            return microbatch_id_in_group % pipeline_parallel_size == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == num_microbatch_groups - 1:
            return microbatch_id_in_group % pipeline_parallel_size == pipeline_parallel_size - 1
        else:
            return False

    def forward_step_helper(microbatch_id, checkpoint_activations_microbatch):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.param_sync_func is not None:
            param_sync_microbatch_id = microbatch_id + pipeline_parallel_rank
            if (
                    param_sync_microbatch_id < total_num_microbatches
                    and is_first_microbatch_for_model_chunk(param_sync_microbatch_id)
            ):
                param_sync_chunk_id = get_model_chunk_id(param_sync_microbatch_id, forward=True) + 1
                if 1 < param_sync_chunk_id < num_model_chunks:
                    config.param_sync_func[param_sync_chunk_id](
                        model[param_sync_chunk_id].parameters()
                    )

        # forward step
        if parallel_state.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]
        output_tensor, _ = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, is_first_microbatch_for_model_chunk(microbatch_id),
            ),
        )
        output_tensors[model_chunk_id].append(output_tensor)

        # if forward-only, no need to save tensors for a backward pass
        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        # ripipe related, when a microbatch finish its forward pass, save needed recomputation
        # functions for this microbatch.
        if args.recompute_in_bubble or args.recompute_in_advance:
            pipeline_checkpoint_manager.batch_fin(model_chunk_id)

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch grad synchronization (default)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id) and nano_flag[model_chunk_id]:
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )

        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.grad_sync_func is not None:
            grad_sync_microbatch_id = microbatch_id - pipeline_parallel_rank
            if grad_sync_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                    grad_sync_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(grad_sync_microbatch_id, forward=False)
                if nano_flag[grad_sync_chunk_id]:
                    enable_grad_sync()
                    config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                    synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

        return input_tensor_grad

    # Run warmup forward passes.
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(p2p_communication.recv_forward(tensor_shape, config))

    fwd_wait_handles = None
    bwd_wait_handles = None

    for k in range(num_warmup_microbatches):

        if fwd_wait_handles is not None:
            for req in fwd_wait_handles:
                req.wait()

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                    k % max_outstanding_backprops
                    >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        # ripipe related, when use recompute_in_bubble function, do not do recompute
        # for the first pp * vp microbatches.
        if args.recompute_in_bubble:
            if k < pipeline_parallel_size * num_model_chunks:
                pipeline_checkpoint_manager.disable_recompute()
            else:
                num_microbatches_recompute_forward -= 1
        output_tensor = forward_step_helper(k, checkpoint_activations_microbatch)
        if args.recompute_in_bubble or args.recompute_in_advance:
            pipeline_checkpoint_manager.enable_recompute()

        # Determine if tensor should be received from previous stage.
        next_forward_model_chunk_id = get_model_chunk_id(k + 1, forward=True)
        recv_prev = True
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            if next_forward_model_chunk_id == 0:
                recv_prev = False
        if k == (total_num_microbatches - 1):
            recv_prev = False

        # Don't send tensor downstream if on last stage.
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if not config.overlap_p2p_comm:
            if (
                    k == (num_warmup_microbatches - 1)
                    and not forward_only
                    and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                (
                    input_tensor,
                    output_tensor_grad,
                ) = p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                )
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            else:
                input_tensor = p2p_communication.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape, config=config
                )
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        else:
            input_tensor, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )

            if (
                    k == (num_warmup_microbatches - 1)
                    and not forward_only
                    and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False

                (
                    output_tensor_grad,
                    bwd_wait_handles,
                ) = p2p_communication.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                    overlap_p2p_comm=True,
                )

                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            input_tensors[next_forward_model_chunk_id].append(input_tensor)

        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                    forward_k % max_outstanding_backprops
                    >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        if config.overlap_p2p_comm:
            if fwd_wait_handles is not None:
                for req in fwd_wait_handles:
                    req.wait()

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            # ripipe related, determine whether this microbatch should be recomputed
            # when using recompute_in_bubble function.
            if args.recompute_in_bubble:
                if num_microbatches_recompute_forward > 0:
                    num_microbatches_recompute_forward -= 1
                elif num_microbatches_recompute > 0 and should_recompute(forward_k):
                    pass
                else:
                    pipeline_checkpoint_manager.disable_recompute()
            output_tensor = forward_step_helper(forward_k, checkpoint_activations_microbatch)
            if args.recompute_in_bubble or args.recompute_in_advance:
                pipeline_checkpoint_manager.enable_recompute()
            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

            # Last virtual stage no activation tensor to send
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True
                )
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Send activation tensor to the next stage and receive activation tensor from the
            # previous stage
            input_tensor, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )
            # assert fwd_wait_handles is not None

            # ripipe related, actually do the recomputation.
            if args.recompute_in_advance or args.recompute_in_bubble:
                vpp_rank = get_model_chunk_id(k, forward=False)
                parallel_state.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                if not parallel_state.is_pipeline_last_stage() or args.recompute_in_bubble:
                    pipeline_checkpoint_manager.recompute_next(vpp_rank)

            if bwd_wait_handles is not None:
                for req in bwd_wait_handles:
                    req.wait()

            # Backward pass.
            backward_k = k
            if k < num_dx and args.use_nanopipe:
                WeightGradStore.start_decouple()

            if args.use_nanopipe:
                WeightGradStore.resize_ori_storage(args.use_nanopipe_swap)

            input_tensor_grad = backward_step_helper(backward_k)
            if args.use_nanopipe:
                if WeightGradStore.is_decoupleBlock:
                    WeightGradStore.flush()
                if k == num_dx - 1:
                    WeightGradStore.end_decouple()

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)

            # First virtual stage no activation gradient tensor to send
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if the current virtual stage has an activation gradient tensor to receive
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

            output_tensor_grad, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                input_tensor_grad,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )

        else:  # no p2p overlap
            output_tensor = forward_step_helper(forward_k, checkpoint_activations_microbatch)

            # Backward pass.
            backward_k = k
            if k < num_dx and args.use_nanopipe:
                WeightGradStore.start_decouple()

            if args.use_nanopipe:
                WeightGradStore.resize_ori_storage(args.use_nanopipe_swap)

            input_tensor_grad = backward_step_helper(backward_k)
            if k == num_dx - 1 and args.use_nanopipe:
                WeightGradStore.end_decouple()

            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True
                )
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
            (
                input_tensor,
                output_tensor_grad,
            ) = p2p_communication.send_forward_backward_recv_forward_backward(
                output_tensor,
                input_tensor_grad,
                recv_prev=recv_prev,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                config=config,
            )
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

        # Put input_tensor and output_tensor_grad in data structures in the
        # right location.
        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        # ripipe related, actually do the recomputation.
        if args.recompute_in_advance:
            vpp_rank = get_model_chunk_id(num_microbatches_remaining, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(vpp_rank)
            if not parallel_state.is_pipeline_last_stage():
                pipeline_checkpoint_manager.recompute_next(vpp_rank)
        if args.recompute_in_bubble and num_microbatches_recompute > 0:
            old_vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)
            pipeline_checkpoint_manager.recompute_next_force(0)
            parallel_state.set_virtual_pipeline_model_parallel_rank(old_vpp_rank)
        if config.overlap_p2p_comm and bwd_wait_handles is not None:
            for wait_handle in bwd_wait_handles:
                wait_handle.wait()

        if all_warmup_microbatches:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communication.recv_backward(tensor_shape, config=config)
            )

        # ripipe related
        if args.recompute_in_bubble:
            num_microbatches_recompute_forward = 1
        for k in range(num_microbatches_remaining, total_num_microbatches):
            input_tensor_grad = backward_step_helper(k)
            next_backward_model_chunk_id = get_model_chunk_id(k + 1, forward=False)
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == (num_model_chunks - 1):
                    recv_next = False
            if k == (total_num_microbatches - 1):
                recv_next = False

            # ripipe related, use async communication
            out_tensor, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape, config=config, overlap_p2p_comm=True
            )
            output_tensor_grads[next_backward_model_chunk_id].append(
                out_tensor
            )

            if args.use_nanopipe and args.use_nanopipe_swap and k == max(num_microbatches_remaining + 1, (total_num_microbatches + num_microbatches_remaining) // 2):
                WeightGradStore.swap_tensors()

            # ripipe related, actually do the recomputation
            if args.recompute_in_bubble and num_microbatches_recompute > 0 and \
                    num_microbatches_recompute_forward < num_microbatches_recompute:
                old_vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)
                pipeline_checkpoint_manager.recompute_next_force(0)
                parallel_state.set_virtual_pipeline_model_parallel_rank(old_vpp_rank)
                num_microbatches_recompute_forward += 1
            if args.recompute_in_advance and k != (total_num_microbatches - 1):
                vpp_rank = get_model_chunk_id(k + 1, forward=False)
                parallel_state.set_virtual_pipeline_model_parallel_rank(vpp_rank)
                if not parallel_state.is_pipeline_last_stage():
                    pipeline_checkpoint_manager.recompute_next(vpp_rank)
            # ripipe related, use async communication
            if config.overlap_p2p_comm and bwd_wait_handles is not None:
                for wait_handle in bwd_wait_handles:
                    wait_handle.wait()

        # nanopipe related
        if args.use_nanopipe:
            if nano_flag[0] and 0 not in synchronized_model_chunks:
                config.grad_sync_func[0](model[0].parameters())
                synchronized_model_chunks.add(0)
            overlap_arg = [pipeline_parallel_size, nano_flag, synchronized_model_chunks, config.grad_sync_func, model]
            WeightGradStore.pop(overlap_arg)

        # Launch any remaining grad reductions.
        enable_grad_sync()
        if config.grad_sync_func is not None:
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(model)

    # ripipe related, check all the needed recomputation is done.
    if args.recompute_in_bubble or args.recompute_in_advance:
        pipeline_checkpoint_manager.iter_fin()

    return forward_data_store


def get_ripipe_recompute_count_params(num_microbatches, num_model_chunks, num_warmup_microbatches):
    """ripipe related, calculate the variables needed by the recompute_in_bubble function"""
    args = get_args()
    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    num_microbatches_recompute_steady_groups = 0
    num_microbatches_recompute_tail = 0
    num_microbatches_recompute = 0
    num_microbatches_recompute_forward = 0
    if args.recompute_in_bubble and num_microbatches // pipeline_parallel_size > 1:
        num_microbatches_recompute = num_warmup_microbatches + 1 - num_model_chunks * pipeline_parallel_size
        if num_microbatches_recompute < 0:
            num_microbatches_recompute = 0

        num_microbatches_recompute_forward = num_microbatches_recompute
        if num_microbatches_recompute > 0 and num_microbatches // pipeline_parallel_size >= 3:
            num_microbatches_recompute_steady_groups = (num_microbatches // pipeline_parallel_size) - 3
            num_microbatches_recompute_tail = 2 + 2 * pipeline_parallel_rank
            if num_microbatches_recompute_steady_groups == 0:
                if num_microbatches_recompute_tail >= pipeline_parallel_size - 1 - 2 * pipeline_parallel_rank:
                    num_microbatches_recompute_tail = 0
                    num_microbatches_recompute_steady_groups = 1
            else:
                num_microbatches_recompute_tail = 1

    params = collections.namedtuple('RecomputeCountParams',
                                    ['num_microbatches_recompute', 'num_microbatches_recompute_forward',
                                     'num_microbatches_recompute_steady_groups', 'num_microbatches_recompute_tail'])
    return params(num_microbatches_recompute, num_microbatches_recompute_forward,
                  num_microbatches_recompute_steady_groups, num_microbatches_recompute_tail)
