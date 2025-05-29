from torch import Tensor
from typing import List, Dict, Union, Optional, Tuple
import torch

def calculate_split_indices(tensor_shapes: Tensor, merge_shape: bool=False) -> Tuple[List[int], List[int]]:
    """
    Calculate tensor sizes and split indices based on tensor shapes.
    
    Args:
        tensor_shapes: A tensor shape
    
    Returns:
        A tuple containing:
            - tensor_sizes: A list of total elements in each tensor
            - split_indices: A list of indices to split the flattened tensor
    """
    if merge_shape:
        from megatron.training import get_args
        merge_size = get_args().mm.model.image_encoder.vision_encoder.spatial_merge_size

    if isinstance(tensor_shapes, List):
        tensor_shapes = torch.cat(tensor_shapes)

    tensor_sizes = []
    for shape in tensor_shapes:
        size = shape.prod()
        if merge_shape:
            size //= (merge_size * merge_size)
        tensor_sizes.append(size.item())
    
    split_indices = [0]
    for size in tensor_sizes:
        split_indices.append(split_indices[-1] + size)
    
    return tensor_sizes, split_indices

# 声明一个 32×1 的正整数 Tensor
tensor_shapes = torch.arange(1, 33, dtype=torch.int32).reshape(32, 1)
print(tensor_shapes.size())

tensor_sizes, split_indices = calculate_split_indices(tensor_shapes)
print(tensor_sizes)

print(split_indices)