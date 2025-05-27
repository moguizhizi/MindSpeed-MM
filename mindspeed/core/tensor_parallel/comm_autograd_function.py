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
import torch.distributed

from mindspeed.core.tensor_parallel.comm_group_api import CollectiveCommIntf
from mindspeed.core.tensor_parallel.comm_group_api import TPXCollectiveComm
from mindspeed.core.tensor_parallel.comm_group_api import TPYCollectiveComm
from mindspeed.core.tensor_parallel.comm_utils import _gather_along_last_dim
from mindspeed.core.tensor_parallel.comm_utils import _split_along_first_dim
from mindspeed.core.tensor_parallel.comm_utils import _split_along_last_dim
from mindspeed.core.tensor_parallel.comm_utils import sync_gather_along_first_dim
from mindspeed.core.tensor_parallel.comm_utils import sync_gather_along_last_dim
from mindspeed.core.tensor_parallel.comm_utils import sync_reduce_scatter_along_first_dim


class _SyncGatherAlongFirstDim(torch.autograd.Function):
    """Gather the input from model parallel X region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return sync_gather_along_first_dim(input_, TPXCollectiveComm)

    @staticmethod
    def forward(ctx, input_, comm_intf: CollectiveCommIntf):
        ctx.comm_intf = comm_intf
        return sync_gather_along_first_dim(input_, comm_intf)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_first_dim(grad_output, ctx.comm_intf), None


class _SyncGatherAlongLastDim(torch.autograd.Function):
    """Gather the input from model parallel Y region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return sync_gather_along_last_dim(input_, TPYCollectiveComm)

    @staticmethod
    def forward(ctx, input_, comm_intf: CollectiveCommIntf):
        ctx.comm_intf = comm_intf
        return sync_gather_along_last_dim(input_, comm_intf)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output, ctx.comm_intf), None


def _reduce(input_, tp_intf: CollectiveCommIntf = TPXCollectiveComm):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if tp_intf.get_comm_group_world_size() == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=tp_intf.get_comm_group())
    return input_


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, tp_intf: CollectiveCommIntf = TPXCollectiveComm):
        return _reduce(input_, tp_intf), None

    @staticmethod
    def forward(ctx, input_, tp_intf: CollectiveCommIntf = TPXCollectiveComm):
        return _reduce(input_, tp_intf)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _GatherFromParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_, comm_intf: CollectiveCommIntf):
        ctx.comm_intf = comm_intf
        return _gather_along_last_dim(input_, comm_intf)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output, ctx.comm_intf), None


class _ScatterAlongLastDim(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, comm_intf: CollectiveCommIntf):
        return _split_along_last_dim(input_, comm_intf)

    @staticmethod
    def forward(ctx, input_, comm_intf: CollectiveCommIntf):
        ctx.comm_intf = comm_intf
        return _split_along_last_dim(input_, comm_intf)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output, ctx.comm_intf), None


class _ScatterAlongFirstDim(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, comm_intf: CollectiveCommIntf):
        return _split_along_first_dim(input_, comm_intf)

    @staticmethod
    def forward(ctx, input_, comm_intf: CollectiveCommIntf):
        ctx.comm_intf = comm_intf
        return _split_along_first_dim(input_, comm_intf)

    @staticmethod
    def backward(ctx, grad_output):
        return sync_gather_along_first_dim(grad_output, ctx.comm_intf), None


class _ScatterAlongFirstDimThenLastDim(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, local_rank_input, first_dim_comm_intf, last_dim_comm_intf):
        graph.first_dim_comm_intf = first_dim_comm_intf
        graph.last_dim_comm_intf = last_dim_comm_intf

        first_dim_split_output = _split_along_first_dim(local_rank_input, first_dim_comm_intf)
        return _split_along_last_dim(first_dim_split_output, last_dim_comm_intf)

    @staticmethod
    def forward(ctx, local_rank_input, first_dim_comm_intf, last_dim_comm_intf):
        ctx.first_dim_comm_intf = first_dim_comm_intf
        ctx.last_dim_comm_intf = last_dim_comm_intf

        first_dim_split_output = _split_along_first_dim(local_rank_input, first_dim_comm_intf)
        return _split_along_last_dim(first_dim_split_output, last_dim_comm_intf)

    @staticmethod
    def backward(ctx, grad_output):
        last_dim_gather_output = _gather_along_last_dim(grad_output, ctx.last_dim_comm_intf)
        first_dim_gather_output = sync_gather_along_first_dim(
            last_dim_gather_output, ctx.first_dim_comm_intf)
        return first_dim_gather_output, None, None


class _SyncGatherAlongFirstDimRS(torch.autograd.Function):
    """Gather the input from model parallel X region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, comm_intf: CollectiveCommIntf):
        return sync_gather_along_first_dim(input_, comm_intf)

    @staticmethod
    def forward(ctx, input_, comm_intf: CollectiveCommIntf):
        ctx.comm_intf = comm_intf
        return sync_gather_along_first_dim(input_, comm_intf)

    @staticmethod
    def backward(ctx, grad_output):
        return sync_reduce_scatter_along_first_dim(grad_output, ctx.comm_intf), None


class _SyncReduceScatterAlongFirstDim(torch.autograd.Function):
    """Reduce scatter the input along first dim"""

    @staticmethod
    def symbolic(graph, input_, comm_intf: CollectiveCommIntf):
        return sync_reduce_scatter_along_first_dim(input_, comm_intf)

    @staticmethod
    def forward(ctx, input_, comm_intf: CollectiveCommIntf):
        ctx.comm_intf = comm_intf
        return sync_reduce_scatter_along_first_dim(input_, comm_intf)

    @staticmethod
    def backward(ctx, grad_output):
        return sync_gather_along_first_dim(grad_output, ctx.comm_intf), None


def auto_grad_sync_gather_along_first_dim(input_, comm_intf: CollectiveCommIntf):
    return _SyncGatherAlongFirstDim.apply(input_, comm_intf)


def auto_grad_sync_gather_along_last_dim(input_, comm_intf: CollectiveCommIntf):
    return _SyncGatherAlongLastDim.apply(input_, comm_intf)


def scatter_to_tensor_parallel_y_region(input_):
    return _ScatterAlongLastDim.apply(input_)


def auto_grad_scatter_along_last_dim(input_, comm_intf: CollectiveCommIntf):
    return _ScatterAlongLastDim.apply(input_, comm_intf)


def auto_grad_scatter_along_first_dim(input_, comm_intf: CollectiveCommIntf):
    return _ScatterAlongFirstDim.apply(input_, comm_intf)


def auto_grad_scatter_along_first_dim_then_last_dim(
    local_rank_input: torch.Tensor,
    first_dim_comm_intf: CollectiveCommIntf,
    last_dim_comm_intf: CollectiveCommIntf,
):
    return _ScatterAlongFirstDimThenLastDim.apply(
        local_rank_input, first_dim_comm_intf, last_dim_comm_intf
    )


def reduce_from_parallel_region(input_, tp_intf: CollectiveCommIntf = TPXCollectiveComm):
    return _ReduceFromModelParallelRegion.apply(input_, tp_intf)


def gather_from_parallel_region(input_, comm_intf: CollectiveCommIntf):
    return _GatherFromParallelRegion.apply(input_, comm_intf)


def auto_grad_sync_gather_along_first_dim_rs(input_, comm_intf: CollectiveCommIntf):
    return _SyncGatherAlongFirstDimRS.apply(input_, comm_intf)


def auto_grad_reduce_scatter_along_first_dim(input_, comm_intf: CollectiveCommIntf):
    return _SyncReduceScatterAlongFirstDim.apply(input_, comm_intf)
