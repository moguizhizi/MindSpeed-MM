# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import Optional, Sequence, Tuple, List, Dict
import torch
import torch_npu
from ..parallel_state import _is_pipeline_first_stage, _is_pipeline_last_stage, get_global_pipeline_parallel_rank
from ..config.dist_train_config import get_dist_model_index, get_rank_number_to_model_index, get_dist_model_config

TENSOR_SYNC_TOOL: "TensorSyncTool" = None


def init_tensor_sync_tool():
    global TENSOR_SYNC_TOOL
    if TENSOR_SYNC_TOOL is None:
        TENSOR_SYNC_TOOL = TensorSyncTool()
    return TENSOR_SYNC_TOOL


class TensorSyncTool:
    def __init__(self):
        dtypes = []
        for name in dir(torch):
            attr = getattr(torch, name)
            if isinstance(attr, torch.dtype) and attr not in torch_npu.unsupported_dtype:
                dtypes.append(attr)
        # Sorting enables different machines to obtain dtypes in the same sequence.
        dtypes = sorted(set(dtypes), key=lambda x: str(x))

        self.type_to_int = {None: -1}
        self.type_to_int.update({dtype: i for i, dtype in enumerate(dtypes)})
        self.int_to_type = {v: k for k, v in self.type_to_int.items()}
        # fixed_header_len (10) = dtype (1) + req_grads (1) + len_shape (1) + shape (x) + pad(10 - 3 - x)
        # Thus, the maximum dimension of tensors that can be supported here is 7.
        self.fixed_header_len = 10

    def encode_tensor_header(self, tensor: torch.Tensor):
        """
        | int32 |   int32   |  int32      |  int32 | int32 |
        | type  | req_grads |  len(shape) |  shape |  pad  |
        """
        header = [0] * self.fixed_header_len

        header[0] = self.type_to_int.get(tensor.dtype, -1) if tensor is not None else -1
        if header[0] not in self.type_to_int.values():
            if header[0] == -1:  # `-1` matches `None`
                return header
            raise RuntimeError(f"The tensor dtype is not supported or recorded on this device: {tensor.dtype}")
        header[1] = int(tensor.requires_grad)
        header[2] = len(tensor.shape)
        if self.fixed_header_len - 3 < len(tensor.shape):  # `3` equals the len of [dtype, req_grads, len_shape]
            raise ValueError('`len(tensor.shape)` is too long to be stored in the remaining space of the header.')
        header[3:] = tensor.shape

        device = torch.npu.current_device()
        index = list(range(len(header)))
        index = torch.tensor(index, dtype=torch.int32, device=device)
        header = torch.tensor(header, dtype=torch.int32, device=device)
        header_tensor = torch.zeros(TENSOR_SYNC_TOOL.fixed_header_len, dtype=torch.int32, device=device)
        header_tensor.scatter_(0, index, header)
        return header_tensor

    def decode_tensor_header(self, header_tensor: torch.Tensor):
        dtype = self.int_to_type.get(int(header_tensor[0]), None)
        if dtype is None:
            return dtype, None, None
        requires_grad = bool(header_tensor[1])
        shape_len = header_tensor[2]
        shape = header_tensor.tolist()[3:3 + shape_len]
        return dtype, shape, requires_grad


def send_recv(tensor: Optional[torch.Tensor], is_recv: bool, ranks: Sequence) -> Optional[Sequence[torch.Tensor]]:
    """
    force_send is used for text_only backward situations.pre_subworld skips backward if recv None tensor.
    """
    if isinstance(tensor, Sequence):
        tensor = tensor[0]

    recv_tensor = None
    # To prevent deadlocks caused by different pipeline stages receiving tensor simultaneously.
    if not get_global_pipeline_parallel_rank() % 2:
        if tensor is not None:
            _send_tensor(tensor, ranks)
        if is_recv:
            recv_tensor = _recv_tensor(ranks)
    else:
        if is_recv:
            recv_tensor = _recv_tensor(ranks)
        if tensor is not None:
            _send_tensor(tensor, ranks)

    if is_recv and not isinstance(recv_tensor, list):
        recv_tensor = [recv_tensor]

    return recv_tensor


def send_recv_tensor_list(
    tensor_list: Optional[Sequence[torch.Tensor]],
    is_recv: bool,
    dst_ranks: Sequence[int],
) -> Optional[Sequence[Sequence[torch.Tensor]]]:
    if tensor_list is None:
        if not is_recv:
            raise ValueError('`tensor_list` can be set to `None` only on the receive side.')
    elif isinstance(tensor_list, Sequence) and len(tensor_list) > 0 and isinstance(tensor_list[0], Sequence):
        tensor_list = tensor_list[0]
    else:
        if not isinstance(tensor_list, Sequence):
            raise TypeError(f'`tensor_list` is an unsupported type: {type(tensor_list)}')
        if not isinstance(tensor_list[0], torch.Tensor):
            raise TypeError(f'item of `tensor_list` is an unsupported type: {type(tensor_list[0])}')

    tensor_list_ret = None
    # To prevent deadlocks caused by different pipeline stages receiving tensor simultaneously.
    if not get_global_pipeline_parallel_rank() % 2:
        if tensor_list is not None:
            send_tensor_list(tensor_list, dst_ranks)
        if is_recv:
            tensor_list_ret = recv_tensor_list(dst_ranks)
    else:
        if is_recv:
            tensor_list_ret = recv_tensor_list(dst_ranks)
        if tensor_list is not None:
            send_tensor_list(tensor_list, dst_ranks)

    return tensor_list_ret


def recv_tensor_list(src_ranks: Sequence[int]) -> Optional[Sequence[Sequence[torch.Tensor]]]:
    tensor_list_len = []
    recv_tensor = torch.tensor([0], device=torch.npu.current_device())
    for rank in src_ranks:
        torch.distributed.recv(recv_tensor, rank)
        tensor_list_len.append(recv_tensor.item())

    if not all(tensor_list_len[0] == len_ for len_ in tensor_list_len[1:]):
        raise ValueError(f'Tensor sequences of different lengths cannot be received from different cards.')
    tensor_list_ret = [_recv_tensor(src_ranks) for _ in range(tensor_list_len[0])]

    return tensor_list_ret


def send_tensor_list(tensor_list: Optional[Sequence[torch.Tensor]], dst_ranks: Sequence[int]) -> None:
    tensor_list_len = len(tensor_list)
    if tensor_list_len == 0:
        return
    send_tensor = torch.tensor([tensor_list_len], device=torch.npu.current_device())
    for rank in dst_ranks:
        torch.distributed.send(send_tensor, rank)
    for i in range(tensor_list_len):
        _send_tensor(tensor_list[i], dst_ranks)


def _send_header(tensor: torch.Tensor, dst: int) -> None:
    header_tensor = TENSOR_SYNC_TOOL.encode_tensor_header(tensor)
    torch.distributed.send(header_tensor, dst)


def _send_tensor(tensor: torch.tensor, dst_ranks: Sequence) -> None:
    if tensor is None:
        return
    for dst in dst_ranks:
        _send_header(tensor, dst)
        torch.distributed.send(tensor=tensor, dst=dst)


def _recv_header(src: int) -> Tuple[Optional[torch.dtype], Optional[List[int]], Optional[bool]]:
    device = torch.npu.current_device()
    header_tensor = torch.zeros(TENSOR_SYNC_TOOL.fixed_header_len, dtype=torch.int32, device=device)
    torch.distributed.recv(header_tensor, src)
    header = TENSOR_SYNC_TOOL.decode_tensor_header(header_tensor)
    return header


def _recv_tensor(dst_ranks: Sequence) -> Optional[Sequence[torch.Tensor]]:
    """Asynchronously receiving tensors

    first receive the shape and dtype, use these to initialize an empty tensor,
    then receive the tensor data, and finally return the tensor.
    """
    recv_tensors = []
    for rank in dst_ranks:
        # recv header
        dtype, shape, requires_grad = _recv_header(rank)
        device = torch.npu.current_device()
        if dtype is None:
            print('[WARNING] Get dtype=None from received header.')
            return None
        # recv tensor
        tensor_recv_prev = torch.empty(tuple(shape), dtype=dtype, device=device, requires_grad=requires_grad)
        torch.distributed.recv(tensor=tensor_recv_prev, src=rank)

        recv_tensors.append(tensor_recv_prev)
    return recv_tensors


def generate_send_recv_mask(rank: int = None) -> Dict[str, bool]:
    model_index = get_dist_model_index(rank)
    rank_number_to_model_index = get_rank_number_to_model_index()
    if model_index not in rank_number_to_model_index:
        raise RuntimeError(f"model_index ({model_index}) not in _RANK_NUMBER_TO_MODEL_INDEX")

    result = {
        'send_forward': False,
        'send_backward': False,
        'recv_forward': False,
        'recv_backward': False
    }
    if _is_pipeline_first_stage(is_global=False):
        for i, index in enumerate(rank_number_to_model_index):
            if index < model_index:
                result['recv_forward'] = True
                if (not get_dist_model_config(rank=i).forward_only) \
                        and (not get_dist_model_config(rank=rank).forward_only):
                    result['send_backward'] = True
                break

    if _is_pipeline_last_stage(is_global=False):
        for i, index in enumerate(rank_number_to_model_index):
            if index > model_index:
                result['send_forward'] = True
                if (not get_dist_model_config(rank=i).forward_only) \
                        and (not get_dist_model_config(rank=rank).forward_only):
                    result['recv_backward'] = True
                break

    return result


init_tensor_sync_tool()
