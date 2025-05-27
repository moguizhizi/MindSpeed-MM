# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import torch
from torch import Tensor
from torch import distributed
import torch.distributed as dist

from megatron.core.parallel_state import get_global_memory_buffer
from mindspeed.core.tensor_parallel.comm_group_api import CollectiveCommIntf
from mindspeed.core.tensor_parallel.comm_group_api import TPXCollectiveComm


def _split_along_last_dim(
    local_rank_input: Tensor, comm_intf: CollectiveCommIntf = TPXCollectiveComm
):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = comm_intf.get_comm_group_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return local_rank_input

    # Split along last dimension.
    last_dim = local_rank_input.dim() - 1
    last_dim_size = local_rank_input.size()[last_dim] // world_size
    # Split.
    tensor_list = torch.split(local_rank_input, last_dim_size, dim=last_dim)

    # Note: torch.split does not create contiguous tensors by default.
    rank = comm_intf.get_comm_rank()
    output = tensor_list[rank].contiguous()

    return output


def _split_along_first_dim(local_rank_input, comm_intf: CollectiveCommIntf = TPXCollectiveComm):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = comm_intf.get_comm_group_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return local_rank_input

    # Split along first dimension.
    dim_size = local_rank_input.size()[0]
    if dim_size % world_size:
        raise AssertionError("First dimension of the tensor should be divisible by parallel size")
    local_dim_size = dim_size // world_size
    rank = comm_intf.get_comm_rank()
    dim_offset = rank * local_dim_size

    output = local_rank_input[dim_offset : dim_offset + local_dim_size].contiguous()

    return output


def _gather_along_last_dim(
    local_rank_input: Tensor, ag_comm_intf: CollectiveCommIntf = TPXCollectiveComm
):
    """Gather tensors and concatinate along the last dimension."""

    world_size = ag_comm_intf.get_comm_group_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return local_rank_input

    tensor_list = [torch.empty_like(local_rank_input) for _ in range(world_size)]
    torch.distributed.all_gather(
        tensor_list, local_rank_input, group=ag_comm_intf.get_comm_group(), async_op=False
    )

    # Note: torch.cat already creates a contiguous tensor.
    last_dim = local_rank_input.dim() - 1
    output = torch.cat(tensor_list, dim=last_dim).contiguous()
    return output


def sync_gather_along_last_dim(
    local_rank_tensor: Tensor, ag_comm_intf: CollectiveCommIntf = TPXCollectiveComm
):
    """Gather tensors and concatinate along the last dimension synchronously.

    :param local_rank_tensor: input of current rank.
    :param ag_comm_intf: the communication process group interface.
    :return: the AllGather-ed result.
    """

    world_size = ag_comm_intf.get_comm_group_world_size()
    # Bypass the function if we are using only 1 GPU/NPU.
    if world_size == 1:
        return local_rank_tensor

    gathered_tensors = [torch.empty_like(local_rank_tensor) for _ in range(world_size)]
    torch.distributed.all_gather(
        gathered_tensors,
        local_rank_tensor.contiguous(),
        group=ag_comm_intf.get_comm_group(),
        async_op=False,
    )

    return torch.cat(gathered_tensors, dim=local_rank_tensor.dim() - 1).contiguous()


def async_gather_tensors(
    local_rank_input: Tensor,
    ag_comm_intf: CollectiveCommIntf = TPXCollectiveComm,
    buffer_name="mpu-async-tp-2d",
):
    """Gather tensors and concatinate along the last dimension asynchronously.

    :param local_rank_input: input of current rank.
    :param ag_comm_intf: the AllGather communication process group interface.
    :param buffer_name: buffer name of str type.
    :return: the AllGather op handle and tensor list storing the op result tensors.

        Note: the result tensors may be handled as following according to your need:
        output = torch.cat(gathered_tensors, dim=xx_dim).contiguous()
    """

    world_size = ag_comm_intf.get_comm_group_world_size()
    # Bypass the function if we are using only 1 NPU/GPU.
    if world_size == 1:
        return None, local_rank_input

    dim_size = list(local_rank_input.size())
    dim_size[0] *= world_size

    ag_out = torch.empty(dim_size, dtype=local_rank_input.dtype, device=torch.cuda.current_device())
    handle = torch.distributed._all_gather_base(
        ag_out, local_rank_input, group=ag_comm_intf.get_comm_group(), async_op=True
    )

    return handle, ag_out


def sync_gather_along_first_dim(
    local_rank_input: Tensor,
    comm_intf: CollectiveCommIntf = TPXCollectiveComm,
    buffer_name=None,
):
    """Gather tensors and concatinate along the first dimension."""

    world_size = comm_intf.get_comm_group_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return local_rank_input

    dim_size = list(local_rank_input.size())
    dim_size[0] *= world_size

    if buffer_name is None:
        output = torch.empty(dim_size, dtype=local_rank_input.dtype, device=torch.cuda.current_device())
    else:
        output = get_global_memory_buffer().get_tensor(dim_size, local_rank_input.dtype, buffer_name)
    torch.distributed._all_gather_base(
        output, local_rank_input.contiguous(), group=comm_intf.get_comm_group()
    )

    return output


def sync_reduce_scatter_along_first_dim(
    local_rank_input, comm_intf: CollectiveCommIntf = TPXCollectiveComm
):
    """Reduce-scatter the input tensor across specified parallel group."""
    world_size = comm_intf.get_comm_group_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return local_rank_input

    dim_size = list(local_rank_input.size())
    if dim_size[0] % world_size:
        raise AssertionError("First dimension of the tensor should be divisible by tensor parallel size")

    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(dim_size, dtype=local_rank_input.dtype, device=torch.cuda.current_device())
    dist.reduce_scatter_tensor(
        output, local_rank_input.contiguous(), group=comm_intf.get_comm_group(), async_op=False
    )

    return output


def async_reduce_scatter_along_first_dim(
    local_rank_input, comm_intf: CollectiveCommIntf = TPXCollectiveComm
):
    """Reduce-scatter the input tensor across parallel group specified by comm_intf."""
    world_size = comm_intf.get_comm_group_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return None, local_rank_input

    dim_size = list(local_rank_input.size())
    if dim_size[0] % world_size:
        raise AssertionError("First dimension of the tensor should be divisible by parallel size")

    dim_size[0] = dim_size[0] // world_size

    rs_output = torch.empty(
        dim_size, dtype=local_rank_input.dtype, device=torch.cuda.current_device()
    )
    handle = dist.reduce_scatter_tensor(
        rs_output, local_rank_input.contiguous(), group=comm_intf.get_comm_group(), async_op=True
    )
    return handle, rs_output


def async_gather_along_last_dim(input_, comm_intf: CollectiveCommIntf = TPXCollectiveComm):
    world_size = comm_intf.get_comm_group_world_size()
    # Bypass the function if we are using only 1 GPU/NPU.
    if world_size == 1:
        return None, input_

    gathered_tensors = [torch.empty_like(input_) for _ in range(world_size)]
    handle = torch.distributed.all_gather(
        gathered_tensors, input_.contiguous(), group=comm_intf.get_comm_group(), async_op=True,
    )

    return handle, gathered_tensors


def sync_reduce_scatter_along_last_dim(
    local_rank_input, rs_comm_intf: CollectiveCommIntf = TPXCollectiveComm
):
    """Reduce-scatter the input tensor across specified parallel group."""
    world_size = rs_comm_intf.get_comm_group_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return local_rank_input

    local_rank_input = local_rank_input.transpose(0, -1)
    output = sync_reduce_scatter_along_first_dim(local_rank_input, rs_comm_intf)
    return output.transpose(0, -1).contiguous()


def async_reduce_scatter_along_last_dim(
    local_rank_input, rs_comm_intf: CollectiveCommIntf = TPXCollectiveComm
):
    """Reduce-scatter the input tensor across model parallel group.

    :param local_rank_input: input of local rank
    :param rs_comm_intf: Reduce scatter comm intf.
    :return:

    Note: the result tensors should be handled as following:
        rs_output = rs_output.transpose(0, 2).contiguous()

    """
    world_size = rs_comm_intf.get_comm_group_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return None, local_rank_input

    local_rank_input = local_rank_input.transpose(0, 2)
    return async_reduce_scatter_along_first_dim(local_rank_input, rs_comm_intf)
