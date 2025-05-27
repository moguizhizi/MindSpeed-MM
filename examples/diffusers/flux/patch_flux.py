#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc

import torch
from diffusers import FluxPipeline
from peft.utils import get_peft_model_state_dict
from torch.distributed._shard.sharded_tensor.api import ShardedTensor


class TorchPatcher:

    @staticmethod
    def new_get_preferred_device(self) -> torch.device:
        """
        Return the preferred device to be used when creating tensors for collectives.
        This method takes into account the asccociated process group
        This patch method makes the torch npu available for distribution
        """
        if dist.get_backend(self._process_group) == dist.Backend.NCCL:
            return torch.device(torch.cuda.current_device())
        try:
            import torch_npu

            return torch.device(torch_npu.npu.current_device())
        except Exception as e:
            return torch.device("cpu")

    @classmethod
    def apply_patch(cls):
        # Apply the patch for npu distribution
        ShardedTensor._get_preferred_device = cls.new_get_preferred_device


def config_gc():
    # set gc threshold, best range from experiments
    gc.set_threshold(700, 50, 1000)


# Save Lora weights for checkpointing steps
def create_save_model_hook(
    accelerator, unwrap_model, transformer, text_encoder_one, args, weight_dtype
):
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_model = unwrap_model(model)
                    if args.upcast_before_saving:
                        transformer_model = transformer_model.to(torch.float32)
                    else:
                        transformer_model = transformer_model.to(weight_dtype)
                    transformer_lora_layers_to_save = get_peft_model_state_dict(
                        transformer_model
                    )

                elif (
                    isinstance(
                        unwrap_model(model), type(unwrap_model(text_encoder_one))
                    )
                    and args.train_text_encoder
                ):
                    text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(
                        model.to(torch.float32)
                    )

                elif (
                    isinstance(
                        unwrap_model(model), type(unwrap_model(text_encoder_one))
                    )
                    and not args.train_text_encoder
                ):
                    text_encoder_one_lora_layers_to_save = None

                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            FluxPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            )

    return save_model_hook
