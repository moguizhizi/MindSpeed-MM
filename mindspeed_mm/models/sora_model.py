# coding=utf-8
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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

from logging import getLogger
from typing import Any, Mapping

import torch
import torch_npu
from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from torch import nn

from mindspeed_mm.data.data_utils.constants import (
    INPUT_MASK,
    LATENTS,
    MASKED_VIDEO,
    PROMPT,
    PROMPT_MASK,
    VIDEO_MASK,
)
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.common.communications import collect_tensors_across_ranks
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.predictor import PredictModel
from mindspeed_mm.models.text_encoder import TextEncoder

logger = getLogger(__name__)


class SoRAModel(nn.Module):
    """
    Instantiate a video generation model from config.
    SoRAModel is an assembled model, which may include text_encoder, video_encoder, predictor, and diffusion model

    Args:
        config (dict): the general config for Multi-Modal Model
        {
            "ae": {...},
            "text_encoder": {...},
            "predictor": {...},
            "diffusion": {...},
            "load_video_features":False,
            ...
        }
    """

    def __init__(self, config):
        super().__init__()
        self.config = core_transformer_config_from_args(get_args())
        self.task = config.task if hasattr(config, "task") else "t2v"
        self.enable_encoder_dp = config.enable_encoder_dp if hasattr(config, "enable_encoder_dp") else False
        if self.enable_encoder_dp and mpu.get_pipeline_model_parallel_world_size() > 1:
            raise AssertionError("Encoder DP cannot be used with PP")
        
        # Encoder inference interleaves with DIT training
        self.interleaved = getattr(config, "interleaved", False)
        self.interleaved_steps = getattr(config, "interleaved_steps", 1)
        
        # Interleaved_steps is disabled when encoder DP is enabled 
        # Interleaved_steps is only used in t2v task
        if hasattr(config, "enable_encoder_dp") and self.enable_encoder_dp or self.task == "i2v": 
            self.interleaved = False
            self.interleaved_steps = 1

        # Track the current index to save or fetch the encoder cache when encoder dp is enabled
        self.cache = {}
        self.index = 0
        args = get_args()
        if args.dist_train:
            from mindspeed.multi_modal.dist_train.parallel_state import is_in_subworld
        self.pre_process = mpu.is_pipeline_first_stage() if not args.dist_train else is_in_subworld('vae') # vae subworld
        self.post_process = mpu.is_pipeline_last_stage()
        self.input_tensor = None
        # to avoid grad all-reduce and reduce-scatter in megatron, since SoRAModel has no embedding layer.
        self.share_embeddings_and_output_weights = False
        self.is_enable_lora = True if hasattr(args, "lora_target_modules") and args.lora_target_modules else False

        if mpu.get_pipeline_model_parallel_rank() == 0:
            self.load_video_features = config.load_video_features
            self.load_text_features = config.load_text_features
            if args.dist_train and not is_in_subworld('vae'):
                self.load_video_features = self.load_text_features = False
            if not self.load_video_features:
                print_rank_0(f"init AEModel....")
                self.ae = AEModel(config.ae).eval()
                self.ae.requires_grad_(False)
            if not self.load_text_features:
                print_rank_0(f"init TextEncoder....")
                self.text_encoder = TextEncoder(config.text_encoder).eval()
                self.text_encoder.requires_grad_(False)

        self.diffusion = DiffusionModel(config.diffusion).get_model()
        if args.dist_train:
            from mindspeed.multi_modal.dist_train.parallel_state import is_in_subworld
            if is_in_subworld('dit'):
                self.predictor = PredictModel(config.predictor).get_model()
        else:
            self.predictor = PredictModel(config.predictor).get_model()

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor
        if get_args().dist_train and mpu.is_pipeline_first_stage(is_global=False):  # 开启dist_train后会应用Patch
            self.input_tensor = input_tensor
        else:
            self.predictor.set_input_tensor(input_tensor)

    def forward(self, video, prompt_ids, video_mask=None, prompt_mask=None, **kwargs):
        """
        video: raw video tensors, or ae encoded latent
        prompt_ids: tokenized input_ids, or encoded hidden states
        video_mask: mask for video/image
        prompt_mask: mask for prompt(text)
        """
        args = get_args()
        if self.pre_process:
            with torch.no_grad():
                if self.interleaved:
                    i2v_results = None
                    if video is not None:
                        print_rank_0(f"Encoding with interleaved_steps: {self.interleaved_steps}")

                        # The offload of text_encoder is recommanded to be used with interleaved_steps greater than 10
                        self.text_encoder.to(torch_npu.npu.current_device())

                        self.interleaved_cache = []
                        self.interleaved_index = 0
                        interleaved_length = len(video)

                        for batch_index in range(interleaved_length):
                            # Text encoder and vae, return tensors in cpu.
                            latents, prompt, video_mask_batch, prompt_mask_batch = self.interleaved_encode(
                                video[batch_index], 
                                prompt_ids[batch_index], 
                                video_mask[batch_index], 
                                prompt_mask[batch_index]
                            )
                            
                            self.interleaved_cache.append((latents, prompt, video_mask_batch, 
                                                        prompt_mask_batch, i2v_results))
                        print_rank_0(f"Encoding done")
                else:
                    i2v_results = None
                    if video is not None:
                        self.index = 0

                        # Visual Encode
                        if self.load_video_features:
                            latents = video
                        else:
                            if self.task == "t2v":
                                latents, _ = self.ae.encode(video)
                            elif self.task == "i2v":
                                latents, i2v_results = self.ae.encode(video, **kwargs)
                                kwargs.update(i2v_results)
                            else:
                                raise NotImplementedError(f"Task {self.task} is not Implemented!")

                        # Text Encode
                        if self.load_text_features:
                            prompt = prompt_ids
                            if isinstance(prompt_ids, list) or isinstance(prompt_ids, tuple):
                                prompt = [p.npu() for p in prompt]
                        else:
                            prompt, prompt_mask = self.text_encoder.encode(prompt_ids, prompt_mask, **kwargs)
            
            if self.interleaved:
                # The offload of text_encoder is recommanded to be used with interleaved_steps greater than 10
                if self.interleaved_index == 0:
                    self.text_encoder.to(torch.device('cpu'))
                    torch_npu.npu.empty_cache()

                latents, prompt, video_mask, prompt_mask, i2v_results = \
                                    self.interleaved_cache[self.interleaved_index]

                cur_device = torch_npu.npu.current_device()
                latents, prompt, video_mask, prompt_mask = self.interleaved_to_device(
                    latents, 
                    prompt, 
                    video_mask, 
                    prompt_mask, 
                    cur_device
                )
                
                self.interleaved_index += 1

            # Gather the results after encoding of ae and text_encoder
            if self.enable_encoder_dp:
                if self.index == 0:
                    self.init_cache(latents, prompt, video_mask, prompt_mask, i2v_results)
                latents, prompt, video_mask, prompt_mask, i2v_results = self.get_feature_from_cache()
            if self.task == "i2v" and i2v_results is not None:
                kwargs.update(i2v_results)
            noised_latents, noise, timesteps = self.diffusion.q_sample(latents, model_kwargs=kwargs, mask=video_mask)
            predictor_input_latent, predictor_timesteps, predictor_prompt = noised_latents, timesteps, prompt
            predictor_video_mask, predictor_prompt_mask = video_mask, prompt_mask

            if args.dist_train:
                return [predictor_input_latent, predictor_timesteps, predictor_prompt, predictor_video_mask,
                        predictor_prompt_mask,
                        latents, noised_latents, timesteps, noise, video_mask]
        else:
            if args.dist_train and mpu.is_pipeline_first_stage(is_global=False):
                [predictor_input_latent, predictor_timesteps, predictor_prompt, predictor_video_mask, predictor_prompt_mask,
                 latents, noised_latents, timesteps, noise, video_mask] = self.input_tensor
            else:
                if not hasattr(self.predictor, "pipeline_set_prev_stage_tensor"):
                    raise ValueError(f"PP has not been implemented for {self.predictor_cls} yet. ")
                predictor_input_list, training_loss_input_list = self.predictor.pipeline_set_prev_stage_tensor(
                    self.input_tensor, extra_kwargs=kwargs)
                predictor_input_latent, predictor_timesteps, predictor_prompt, predictor_video_mask, predictor_prompt_mask \
                    = predictor_input_list
                latents, noised_latents, timesteps, noise, video_mask = training_loss_input_list

        output = self.predictor(
            predictor_input_latent,
            timestep=predictor_timesteps,
            prompt=predictor_prompt,
            video_mask=predictor_video_mask,
            prompt_mask=predictor_prompt_mask,
            **kwargs,
        )

        if self.post_process:
            loss = self.compute_loss(
                output if isinstance(output, torch.Tensor) else output[0],
                latents,
                noised_latents,
                timesteps,
                noise,
                video_mask,
                **kwargs
            )
            return [loss]

        return self.predictor.pipeline_set_next_stage_tensor(
            input_list=[latents, noised_latents, timesteps, noise, video_mask],
            output_list=output,
            extra_kwargs=kwargs)

    def compute_loss(
            self, model_output, latents, noised_latents, timesteps, noise, video_mask, **kwargs
    ):
        """compute diffusion loss"""
        loss_dict = self.diffusion.training_losses(
            model_output=model_output,
            x_start=latents,
            x_t=noised_latents,
            noise=noise,
            t=timesteps,
            mask=video_mask,
            **kwargs
        )
        return loss_dict

    def train(self, mode=True):
        if hasattr(self, "predictor"):
            self.predictor.train()

    def state_dict_for_save_lora_checkpoint(self, state_dict):
        state_dict_ = dict()
        for key in state_dict:
            if 'lora' in key:
                state_dict_[key] = state_dict[key]
        return state_dict_

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """Customized state_dict"""
        if not get_args().dist_train:
            state_dict = self.predictor.state_dict(prefix=prefix, keep_vars=keep_vars)
            if self.is_enable_lora:
                state_dict = self.state_dict_for_save_lora_checkpoint(state_dict)
            return state_dict
        from mindspeed.multi_modal.dist_train.parallel_state import is_in_subworld
        if is_in_subworld('dit'):
            return self.predictor.state_dict(prefix=prefix, keep_vars=keep_vars)
        return None

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        """Customized load."""
        if get_args().dist_train:
            from mindspeed.multi_modal.dist_train.parallel_state import is_in_subworld
            if is_in_subworld('vae'):
                return None
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")

        missing_keys, unexpected_keys = self.predictor.load_state_dict(state_dict, False)

        if missing_keys is not None:
            logger.info(f"Missing keys in state_dict: {missing_keys}.")
        if unexpected_keys is not None:
            logger.info(f"Unexpected key(s) in state_dict: {unexpected_keys}.")
        return None

    def init_cache(self, latents, prompt, video_mask, prompt_mask, i2v_results=None):
        """Initialize cache in the first step."""
        self.cache = {}
        group = mpu.get_tensor_and_context_parallel_group()
        # gather as list
        for key, value in [(LATENTS, latents), (PROMPT, prompt), (VIDEO_MASK, video_mask), (PROMPT_MASK, prompt_mask)]:
            if value is not None:
                self.cache.update({key: collect_tensors_across_ranks(value, group=group)})

        if self.task != "i2v" or not i2v_results:
            return

        for key in [MASKED_VIDEO, INPUT_MASK]:
            if key in i2v_results:
                self.cache[key] = collect_tensors_across_ranks(i2v_results[key], group=group)

    def get_feature_from_cache(self):
        """Get from the cache"""

        latents = self.cache[LATENTS][self.index] if LATENTS in self.cache else None
        prompt = self.cache[PROMPT][self.index] if PROMPT in self.cache else None
        video_mask = self.cache[VIDEO_MASK][self.index] if VIDEO_MASK in self.cache else None
        prompt_mask = self.cache[PROMPT_MASK][self.index] if PROMPT_MASK in self.cache else None

        i2v_results = {}
        if self.task == "i2v":
            i2v_results[MASKED_VIDEO] = self.cache[MASKED_VIDEO][self.index] if MASKED_VIDEO in self.cache else None
            i2v_results[INPUT_MASK] = self.cache[INPUT_MASK][self.index] if INPUT_MASK in self.cache else None

        self.index += 1
        return latents, prompt, video_mask, prompt_mask, i2v_results

    def interleaved_encode(self, video_batch, prompt_ids_batch, video_mask_batch, prompt_mask_batch):
        """Text_Encoder and AE with interleaved steps"""

        cur_device = torch_npu.npu.current_device()
        video_batch, prompt_ids_batch, video_mask_batch, prompt_mask_batch = self.interleaved_to_device(
            video_batch, 
            prompt_ids_batch, 
            video_mask_batch, 
            prompt_mask_batch, 
            cur_device
        )

        # Visual Encode
        if self.load_video_features:
            latents = video_batch
        else:
            if self.task == "t2v":
                latents, _ = self.ae.encode(video_batch)
            else:
                raise NotImplementedError(f"Task {self.task} is not Implemented with Interleaved!")
        
        # Text Encode
        if self.load_text_features:
            prompt = prompt_ids_batch
        else:
            prompt, prompt_mask_batch = self.text_encoder.encode(prompt_ids_batch, prompt_mask_batch)

        cur_device = torch.device('cpu')
        latents, prompt, video_mask_batch, prompt_mask_batch = self.interleaved_to_device(
            latents, 
            prompt, 
            video_mask_batch, 
            prompt_mask_batch, 
            cur_device
        )

        return latents, prompt, video_mask_batch, prompt_mask_batch

    def interleaved_to_device(
        self, video_batch, prompt_ids_batch, video_mask_batch, prompt_mask_batch, cur_device
    ):
        """torch_npu.npu.current_device() or torch.device('cpu') for cur_device"""

        video_batch = video_batch.to(cur_device)
        if isinstance(prompt_ids_batch, list) or isinstance(prompt_ids_batch, tuple):
            prompt_ids_batch = [p.to(cur_device) for p in prompt_ids_batch]
        else:
            prompt_ids_batch = prompt_ids_batch.to(cur_device)

        video_mask_batch = video_mask_batch.to(cur_device)
        prompt_mask_batch = prompt_mask_batch.to(cur_device)

        return video_batch, prompt_ids_batch, video_mask_batch, prompt_mask_batch
