import torch
import torch.nn as nn
from einops import rearrange

from megatron.legacy.model import RMSNorm
from mindspeed_mm.models.common.communications import _conv_split, _conv_gather, all_to_all
from mindspeed_mm.models.common.conv import ContextParallelCausalConv3d
from mindspeed_mm.utils.utils import (get_context_parallel_rank, get_context_parallel_world_size,
                                      get_context_parallel_group)


class LayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-6, elementsize_affine=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.norm = torch.nn.LayerNorm(num_channels, eps=eps, elementwise_affine=elementsize_affine)

    def forward(self, x):
        if x.dim() == 5:
            x = rearrange(x, "b c t h w -> b t h w c")
            x = self.norm(x)
            x = rearrange(x, "b t h w c -> b c t h w")
        else:
            x = rearrange(x, "b c h w -> b h w c")
            x = self.norm(x)
            x = rearrange(x, "b h w c -> b c h w")
        return x


def normalize(in_channels, num_groups=32, eps=1e-6, affine=True, norm_type="groupnorm", gather=False, **kwargs):
    if not gather:
        if norm_type == "groupnorm":
            return torch.nn.GroupNorm(
                num_groups=num_groups, num_channels=in_channels, eps=eps, affine=affine
            )
        elif norm_type == "aelayernorm":
            return LayerNorm(num_channels=in_channels, eps=eps)
        elif norm_type == "layernorm":
            return nn.LayerNorm(in_channels, eps=eps, elementwise_affine=affine)
        elif norm_type == "rmsnorm":
            return RMSNorm(dim=in_channels, eps=eps, **kwargs)
        else:
            raise ValueError(f"unsupported norm type: {norm_type}. ")
    else:
        return ContextParallelGroupNorm(num_groups=32, num_channels=in_channels, eps=eps, affine=affine)


class _ConvolutionScatterToContextParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, kernel_size):
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        return _conv_split(input_, dim, kernel_size)

    @staticmethod
    def backward(ctx, grad_output):
        return _conv_gather(grad_output, ctx.dim, ctx.kernel_size), None, None


class _ConvolutionGatherFromContextParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, kernel_size):
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        return _conv_gather(input_, dim, kernel_size)

    @staticmethod
    def backward(ctx, grad_output):
        return _conv_split(grad_output, ctx.dim, ctx.kernel_size), None, None


def conv_scatter_to_context_parallel_region(input_, dim, kernel_size):
    return _ConvolutionScatterToContextParallelRegion.apply(input_, dim, kernel_size)


def conv_gather_from_context_parallel_region(input_, dim, kernel_size):
    return _ConvolutionGatherFromContextParallelRegion.apply(input_, dim, kernel_size)


class ContextParallelGroupNorm(torch.nn.GroupNorm):
    def forward(self, input_, enable_cp=True):
        if not enable_cp:
            return super().forward(input_)

        gather_flag = input_.shape[2] > 1
        if gather_flag:
            cp_world_size = get_context_parallel_world_size()
            if cp_world_size == 1:
                return super().forward(input_)
            group = get_context_parallel_group()
            cp_rank = get_context_parallel_rank()
            _, ch, t, _, _ = input_.shape
            group_size = ch // self.num_groups
            scatter_sizes = torch.tensor_split(torch.ones(self.num_groups) * group_size, cp_world_size)
            scatter_sizes = [int(s.sum().item()) for s in scatter_sizes]
            if cp_rank == 0:
                t -= 1
            gather_sizes = [t] * cp_world_size
            gather_sizes[0] += 1
            input_ = all_to_all(input_, group, 1, 2, scatter_sizes, gather_sizes)
            begin = sum(scatter_sizes[:cp_rank])
            end = begin + scatter_sizes[cp_rank]
            output = torch.nn.functional.group_norm(
                input_, scatter_sizes[cp_rank] // group_size, self.weight[begin: end], self.bias[begin: end], self.eps)
            output = all_to_all(output, group, 2, 1, gather_sizes, scatter_sizes)
        else:
            output = super().forward(input_)
        return output


def Normalize3D(
        in_channels,
        zq_ch=None,
        add_conv=False,
        gather=False,
):
    if gather:
        return SpatialNorm3D(
            in_channels,
            zq_ch,
            gather=gather,
            freeze_norm_layer=False,
            add_conv=add_conv,
            num_groups=32,
            eps=1e-6,
            affine=True,
        )
    else:
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialNorm3D(nn.Module):
    def __init__(
        self,
        f_channels,
        zq_channels,
        freeze_norm_layer=False,
        add_conv=False,
        pad_mode="constant",
        gather=False,
        **norm_layer_params,
    ):
        super().__init__()

        self.norm_layer = torch.nn.GroupNorm(num_channels=f_channels, **norm_layer_params)
        if freeze_norm_layer:
            for p in self.norm_layer.parameters:
                p.requires_grad = False

        self.add_conv = add_conv
        if add_conv:
            self.conv = ContextParallelCausalConv3d(
                chan_in=zq_channels,
                chan_out=zq_channels,
                kernel_size=3,
            )

        self.conv_y = ContextParallelCausalConv3d(
            chan_in=zq_channels,
            chan_out=f_channels,
            kernel_size=1,
        )
        self.conv_b = ContextParallelCausalConv3d(
            chan_in=zq_channels,
            chan_out=f_channels,
            kernel_size=1,
        )

    def forward(self, f, zq, clear_fake_cp_cache=True, enable_cp=True):
        if f.shape[2] > 1 and get_context_parallel_rank() == 0 and enable_cp:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            zq_first, zq_rest = zq[:, :, :1], zq[:, :, 1:]
            zq_first = torch.nn.functional.interpolate(zq_first, size=f_first_size, mode="nearest")

            zq_rest_splits = torch.split(zq_rest, 32, dim=1)
            interpolated_splits = [
                torch.nn.functional.interpolate(split, size=f_rest_size, mode="nearest")
                for split in zq_rest_splits
            ]

            zq_rest = torch.cat(interpolated_splits, dim=1)

            zq = torch.cat([zq_first, zq_rest], dim=2)
        else:
            f_size = f.shape[-3:]

            zq_splits = torch.split(zq, 32, dim=1)
            interpolated_splits = [
                torch.nn.functional.interpolate(split, size=f_size, mode="nearest")
                for split in zq_splits
            ]
            zq = torch.cat(interpolated_splits, dim=1)

        if self.add_conv:
            zq, _ = self.conv(zq, clear_cache=clear_fake_cp_cache, enable_cp=enable_cp)

        norm_f = self.norm_layer(f)

        conv_y_out, _ = self.conv_y(zq, enable_cp=enable_cp)
        conv_b_out, _ = self.conv_b(zq, enable_cp=enable_cp)
        new_f = norm_f * conv_y_out + conv_b_out
        return new_f
