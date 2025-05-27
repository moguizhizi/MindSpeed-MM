import os
from functools import wraps

import torch


class FusedLayerNormAffineFunction:
    @staticmethod
    def apply(input_, weight, bias, normalized_shape, eps):
        return torch.nn.functional.layer_norm(input_, normalized_shape, weight, bias, eps)

    @staticmethod
    def forward(*args, **kwargs):
        return FusedLayerNormAffineFunction.apply(*args, **kwargs)


class FastLayerNormFN:
    @staticmethod
    def apply(input_, weight, bias, eps):
        normalized_shape = torch.Size(weight.numel())
        return torch.nn.functional.layer_norm(input_, normalized_shape, weight, bias, eps)


def fused_layer_norm_affine(input_, weight, bias, normalized_shape, eps):
    return torch.nn.functional.layer_norm(input_, normalized_shape, weight, bias, eps)
