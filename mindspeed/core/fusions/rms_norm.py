import os
from functools import wraps
import torch
import torch_npu
from torch import nn
from megatron.legacy.model.rms_norm import RMSNorm
from megatron.training import get_args

from mindspeed.core.tensor_parallel.mapping import reduce_from_tensor_model_parallel_region_nd


def rms_norm_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        _args = get_args()
        self.use_fused_rmsnorm = _args.use_fused_rmsnorm
        self.use_nd_matmul = _args.use_nd_matmul
        dim = args[0] if len(args) > 0 else kwargs.get('dim')
        if self.use_nd_matmul:
            if self.use_fused_rmsnorm:
                raise RuntimeError('nd_matmul does not support fused_rmsnorm temporarily')
            self.tensor_model_parallel_size = _args.tensor_model_parallel_size
            self.weight = torch.nn.Parameter(
                torch.ones(dim // self.tensor_model_parallel_size)
            )
    return wrapper


def rms_norm_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, x):
        if int(os.getenv('NPU_ASD_ENABLE', '0')):
            from torch_npu.utils import register_asd_hook
            register_asd_hook(x, self.weight)
        if self.use_fused_rmsnorm:
            return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
        return fn(self, x)
    return wrapper


def rms_norm_norm_wrapper(fn):
    @wraps(fn)
    def wrapper(self, x):
        if self.use_nd_matmul:
            pow_mean = x.pow(2).mean(-1, keepdim=True)
            all_pow_mean = reduce_from_tensor_model_parallel_region_nd(pow_mean)
            pow_mean = torch.div(all_pow_mean, self.tensor_model_parallel_size)
            return x * torch.rsqrt(pow_mean + self.eps)
        return fn(self, x)
    return wrapper
