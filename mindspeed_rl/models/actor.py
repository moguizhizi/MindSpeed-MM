# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from typing import Dict, Callable

import torch
from torch import Tensor

from mindspeed_rl.models.base.base_training_engine import BaseTrainingEngine
from mindspeed_rl.utils.utils import mstx_timer_decorator


class Actor(BaseTrainingEngine):
    """
    Actor class. This class implements the simple logics.
    Args:
        model:  The network model to be trained.
        optimizer: The optimizer for updating model parameters (e.g., Adam).
        opt_param_scheduler: The scheduler for optimizer parameters (e.g., learning rate scheduler).
        beta: float = 0  The weight coefficient for KL divergence (used in algorithms like PPO).
        mini_batch_size_per_dp: int = 1  The size of the mini-batch for each data parallel stage.
        epochs: int = 1   The number of training epochs.
        shuffle_mini_batch: bool = False   Whether to shuffle the mini-batch data at each epoch.
        stage: str = None   The training stage identifier (e.g., pretrain/finetune).
        clip_ratio: float = 0.1   The clipping ratio threshold for PPO (limits the policy update range).
        forward_backward_func: Callable = None   The forward-backward function for distributed training.
        **kwargs:  # Additional parameters for base class argument passing.
    """
    def __init__(
            self,
            model,
            optimizer,
            opt_param_scheduler,
            beta: float = 0,
            mini_batch_size_per_dp: int = 1,
            epochs: int = 1,
            shuffle_mini_batch: bool = False,
            stage: str = None,
            clip_ratio: float = 0.1,
            temperature: float = 1.0,
            forward_backward_func: Callable = None,
            **kwargs
    ):
        super(Actor, self).__init__(
            model,
            optimizer,
            opt_param_scheduler,
            beta=beta,
            mini_batch_size_per_dp=mini_batch_size_per_dp,
            epochs=epochs,
            shuffle_mini_batch=shuffle_mini_batch,
            stage=stage,
            clip_ratio=clip_ratio,
            temperature=temperature,
            role='actor',
            forward_backward_func=forward_backward_func,
            **kwargs)

    def get_loss_meta_func(self):
        meta_info = {}
        if self.clip_ratio is not None:
            meta_info["clip_ratio"] = self.clip_ratio
        if self.beta is not None:
            meta_info["beta"] = self.beta
        if self.kl_ctrl is not None:
            meta_info["kl_ctrl"] = self.kl_ctrl
        return meta_info

    def post_process_forward_backward_output(self, output: torch.Tensor,
                                             batch: Dict[str, torch.Tensor]) -> (Tensor, Dict):
        return output, batch

    @mstx_timer_decorator
    def compute_log_prob(self, data: Dict, llm_only: bool = True) -> (Tensor, Dict):
        return super().forward(data, llm_only=llm_only)
    
    @mstx_timer_decorator
    def compute_image_embeds(self, data: Dict, vit_only: bool = True) -> (Tensor, Dict):
        return super().forward(data, vit_only=vit_only)
    
    @mstx_timer_decorator
    def update_actor(self, data: Dict, kl_ctrl=None) -> Dict[str, torch.Tensor]:
        return super().update(data, kl_ctrl)
