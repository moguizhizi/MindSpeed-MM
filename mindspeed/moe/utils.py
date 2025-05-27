# Copyright (c) Microsoft Corporation.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

# copied from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py
# copied from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_utils.py
# reworked/refactored some parts to make it run.
from typing import Any
from typing import Callable, Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from megatron.core import mpu

gumbel_map: Dict[torch.device, Callable] = {}
USE_EINSUM = False
ampipe_slices_map = {}


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


def all_gather_along_first_dim(input_, is_use_global_memory_buffer=False):
    world_size = mpu.get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size
    if is_use_global_memory_buffer:
        ag_out = mpu.get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
    else:
        ag_out = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._all_gather_base(
        ag_out, input_.contiguous(), group=mpu.get_tensor_model_parallel_group()
    )
    return ag_out


def get_reshape_index_select(num_local_experts, ep_size):
    reshape_index_select = []
    for i in range(num_local_experts):
        index = i
        for j in range(ep_size):
            reshape_index_select.append(index)
            index += num_local_experts
    return reshape_index_select


def get_slice_indices_from_order_to_disorder(seq_length, pipe_degree, device):
    if ampipe_slices_map.get('order_to_disorder') is not None:
        return ampipe_slices_map.get('order_to_disorder')
    tp_size = mpu.get_tensor_model_parallel_world_size()
    slice_size = seq_length // tp_size // pipe_degree

    output = []
    for out_idx in range(0, seq_length // tp_size, slice_size):
        for i in range(out_idx, seq_length, pipe_degree * slice_size):
            for j in range(slice_size):
                output.append(i + j)
    output = torch.tensor(output, dtype=torch.int32, device=device)
    ampipe_slices_map['order_to_disorder'] = output
    return output


def get_slice_indices_from_disorder_to_order(seq_length, pipe_degree, device):
    if ampipe_slices_map.get('disorder_to_order') is not None:
        return ampipe_slices_map.get('disorder_to_order')
    tp_size = mpu.get_tensor_model_parallel_world_size()
    slice_size = seq_length // tp_size // pipe_degree

    output = []
    for out_idx in range(0, seq_length // pipe_degree, slice_size):
        for i in range(out_idx, seq_length, tp_size * slice_size):
            for j in range(slice_size):
                output.append(i + j)
    output = torch.tensor(output, dtype=torch.int32, device=device)
    ampipe_slices_map['disorder_to_order'] = output
    return output


def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()


def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # gates has shape of S,E
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    max_capacity = num_tokens
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    elif capacity > max_capacity:
        capacity = torch.tensor(max_capacity, dtype=torch.int64)
    return capacity


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.
def einsum(rule, a, b):
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == 's,se->se':
        return a.reshape(a.shape[0], -1) * b
    elif rule == 'se,sc->sec':
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == 'se,se->s':
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == 'sec,sm->ecm':
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == 'sec,ecm->sm':
        return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == 'ks,ksm->sm':
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)


class MoEAuxLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that compute and scales the grad for auxiliary loss.

    """

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        """Preserve the aux_loss by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            aux_loss (torch.Tensor): The auxiliary loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for auxiliary loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled auxiliary loss gradient.
        """
        (aux_loss,) = ctx.saved_tensors
        aux_loss_backward_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the aux loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in matches the scale of the main_loss.
        """
        MoEAuxLossAutoScaler.main_loss_backward_scale = scale