# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain SoRA."""

import torch
import torch_npu

import mindspeed.megatron_adaptor

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    unwrap_model,
)

from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.training import pretrain
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.constants import (
    VIDEO,
    PROMPT_IDS,
    PROMPT_MASK,
    VIDEO_MASK
)
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.models.sora_model import SoRAModel
from mindspeed_mm.patchs import dummy_optimizer_patch


def model_provider(pre_process=True, post_process=True):
    """Builds the model."""
    args = get_args()
    print_rank_0("building SoRA model ...")
    model = SoRAModel(args.mm.model)

    if mpu.get_pipeline_model_parallel_world_size() > 1:
        if not hasattr(model.predictor, "initialize_pipeline_tensor_shapes"):
            raise AttributeError("The predictor should provide initialize_pipeline_tensor_shapes for PP_size>1. ")
        args.pipeline_tensor_shapes = model.predictor.initialize_pipeline_tensor_shapes()
        setattr(forward_step, 'pipeline_tensor_shapes', args.pipeline_tensor_shapes)

    return model


def get_batch_on_this_tp_rank(data_iterator):
    args = get_args()
    interleaved = args.mm.model.interleaved \
        if hasattr(args.mm.model, "interleaved") else False
    if data_iterator is not None:
        batch = next(data_iterator, None)
    else:
        return None
    # data is loaded in cpu for interleaved.
    if batch is not None and not interleaved:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(torch_npu.npu.current_device())
    return batch


def get_batch(data_iterator):
    """Generate a batch."""
    if mpu.is_pipeline_first_stage():
        batch = get_batch_on_this_tp_rank(data_iterator)
        return batch
    else:
        return None


def loss_func(output_tensor):
    """Loss function."""
    loss = output_tensor[0].mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)
    return loss, {"loss": averaged_loss[0]}


def get_batch_for_step(data_iterator):
    args = get_args()
    args.curr_forward_iteration += 1
    enable_encoder_dp = args.mm.model.enable_encoder_dp if hasattr(args.mm.model, "enable_encoder_dp") else False
    tp_cp_group_size = None
    if enable_encoder_dp:
        tp_cp_group_size = torch.distributed.get_world_size(mpu.get_tensor_and_context_parallel_group())

    if tp_cp_group_size is None or not enable_encoder_dp or tp_cp_group_size <= 1:
        return get_batch(data_iterator)

    # Only the first step of a round needs to get batch when enable encoder dp
    batch = get_batch(data_iterator) if args.curr_forward_iteration % tp_cp_group_size == 1 else {}

    return batch


def get_interleaved_batch_for_step(data_iterator):
    """Generate batches with interleaved steps."""
    args = get_args()
    args.curr_forward_iteration += 1
    interleaved_steps = args.mm.model.interleaved_steps \
        if hasattr(args.mm.model, "interleaved_steps") else 1
    if interleaved_steps < 1 or not isinstance(interleaved_steps, int):
        raise AssertionError("interleaved_steps should be an integer greater than or equal to 1.")

    batches = []
    if interleaved_steps == 1:
        batch = get_batch(data_iterator)
        batches.append(batch)

    elif args.curr_forward_iteration % interleaved_steps == 1:
        # When interleaved is enabled, the get_batch needs to be obtained for every interleaving step.
        for _ in range(interleaved_steps):
            batch = get_batch(data_iterator)
            # When the length of dataiter is not divided by interleaved_steps,  dataiter return None 
            if batch is not None:
                batches.append(batch)

    return batches


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    interleaved = args.mm.model.interleaved \
        if hasattr(args.mm.model, "interleaved") else False

    batch, video, prompt_ids, video_mask, prompt_mask = {}, None, None, None, None
    if mpu.is_pipeline_first_stage():
        if not interleaved:        
            batch = get_batch_for_step(data_iterator)
            video = batch.pop(VIDEO, None)
            prompt_ids = batch.pop(PROMPT_IDS, None)
            video_mask = batch.pop(VIDEO_MASK, None)
            prompt_mask = batch.pop(PROMPT_MASK, None)
        else:
            batches = get_interleaved_batch_for_step(data_iterator)
            
            if len(batches) > 0:
                video, prompt_ids, video_mask, prompt_mask = [], [], [], []
                # List for these params when interleaved is enabled.
                for batch in batches:
                    video_batch = batch.pop(VIDEO, None)
                    prompt_ids_batch = batch.pop(PROMPT_IDS, None)
                    video_mask_batch = batch.pop(VIDEO_MASK, None)
                    prompt_mask_batch = batch.pop(PROMPT_MASK, None)
                    
                    video.append(video_batch)
                    prompt_ids.append(prompt_ids_batch)
                    video_mask.append(video_mask_batch)
                    prompt_mask.append(prompt_mask_batch)

    output_tensor_list = model(video, prompt_ids, video_mask, prompt_mask=prompt_mask, **batch)
    return output_tensor_list, loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    data_config = args.mm.data
    train_dataset = build_mm_dataset(data_config.dataset_param)

    enable_encoder_dp = args.mm.model.enable_encoder_dp if hasattr(args.mm.model, "enable_encoder_dp") else False
    if enable_encoder_dp:
        process_group = torch.distributed.group.WORLD
    else:
        process_group = mpu.get_data_parallel_group()

    train_dataloader = build_mm_dataloader(
        train_dataset,
        data_config.dataloader_param,
        process_group=process_group,
        consumed_samples=args.consumed_train_samples,
        dataset_param=data_config.dataset_param,
    )
    data_iterator, _, _ = build_iterations(train_dl=train_dataloader)
    return data_iterator, None, None


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=mm_extra_args_provider,
        args_defaults={"dataloader_type": "external", "vision_pretraining": False, "curr_forward_iteration": 0},
    )
