# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

from typing import List, Optional, Union
import html
import math

from PIL.Image import Image
import ftfy
import regex as re
import torch
from torchvision.transforms import v2
from transformers import CLIPVisionModel
from megatron.training import get_args
from megatron.core import mpu
from mindspeed_mm.utils.utils import get_device

from .pipeline_base import MMPipeline
from .pipeline_mixin.encode_mixin import MMEncoderMixin
from .pipeline_mixin.inputs_checks_mixin import InputsCheckMixin

NEGATIVE_PROMOPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


class WanPipeline(MMPipeline, InputsCheckMixin, MMEncoderMixin):

    def __init__(self, vae, tokenizer, text_encoder, scheduler, predict_model, image_encoder=None, config=None):
        super().__init__()

        args = get_args()
        args = args.mm.model

        if hasattr(args, "image_encoder"):
            image_encoder_config = args.image_encoder.to_dict()
            image_encoder = CLIPVisionModel.from_pretrained(image_encoder_config["from_pretrained"])
            image_encoder.to(dtype=image_encoder_config["dtype"], device=get_device(args.device)).eval()
            self.model_cpu_offload_seq = "text_encoder->image_encoder->predict_model->vae"
        else:
            image_encoder = None
            self.model_cpu_offload_seq = "text_encoder->predict_model->vae"

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            predict_model=predict_model,
            image_encoder=image_encoder,
        )

        self.cp_size = mpu.get_context_parallel_world_size()
        self.vae_scale_factor_temporal = (
            2 ** sum(self.vae.model.config.temperal_downsample) if getattr(self, "vae", None) else 4
        )
        self.vae_scale_factor_spatial = (
            2 ** len(self.vae.model.config.temperal_downsample) if getattr(self, "vae", None) else 8
        )
        self.patch_size = self.predict_model.patch_size if getattr(self, "predict_model", None) else (1, 2, 2)

        self.num_frames, self.height, self.width = config.input_size
        self.generator = None if not hasattr(config, "seed") else torch.Generator().manual_seed(config.seed)

        self.cpu_offload = getattr(config, "cpu_offload", False)
        if self.cpu_offload:
            self.enable_model_cpu_offload(torch.distributed.get_rank())

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[Union[Image, List[Image]]] = None,
        video: List[Image] = None,
        negative_prompt: Union[str, List[str]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        device: torch.device = get_device("npu"),
        max_sequence_length: int = 512,
        **kwargs,
    ):
        # 1. Check inputs. Raise error if not correct
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = NEGATIVE_PROMOPT
        self.check_inputs(
            prompt,
            negative_prompt,
            self.height,
            self.width,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        self.strength = kwargs["strength"]

        # 3. Encode input prompt
        do_classifier_free_guidance = self.scheduler.do_classifier_free_guidance
        prompt_embeds, negative_prompt_embeds = self.encode_texts(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # 4. Prepare latents and model_kwargs
        if image is not None:
            latents, clip_features, vae_features = self.prepare_image_latents(
                batch_size, image, device, prompt_embeds.dtype
            )
        elif video is not None:
            latents, video_vae_features = self.prepare_video_latents(
                batch_size, video, device, prompt_embeds.dtype
            )
            clip_features, vae_features = None, None
        else:
            shape = (
                batch_size,
                self.predict_model.in_dim,
                (self.num_frames - 1) // self.vae_scale_factor_temporal + 1,
                self.height // self.vae_scale_factor_spatial,
                self.width // self.vae_scale_factor_spatial,
            )
            latents = self.prepare_latents(shape, generator=self.generator, device=device, dtype=prompt_embeds.dtype)
            clip_features, vae_features = None, None

        model_kwargs = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "i2v_clip_feature": clip_features,
            "i2v_vae_feature": vae_features,
        }

        # 5. Denoising to get clean latents
        num_inference_steps = self.scheduler.num_inference_steps
        timesteps = self.scheduler.timesteps

        ## v2v task has to add noise
        if video is not None:
            timesteps, _ = self.get_timesteps(num_inference_steps, timesteps, self.strength)
            latent_timestep = timesteps[:1].repeat(batch_size)
            latents = self.scheduler.add_noise(video_vae_features, latents, latent_timestep)

        num_warmup_steps = self.scheduler.num_warmup_steps
        guidance_scale = self.scheduler.guidance_scale
        self.scheduler.diffusion.set_timesteps(num_inference_steps)  # reset timesteps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = latents.to(self.predict_model.dtype)
                timestep = t.expand(latents.shape[0]).to(device=latents.device).float()

                noise_pred = self.predict_model(
                    latent_model_input, timestep, model_kwargs.get("prompt_embeds"), **model_kwargs
                )[0]

                if do_classifier_free_guidance:
                    noise_uncond = self.predict_model(
                        latent_model_input, timestep, model_kwargs.get("negative_prompt_embeds"), **model_kwargs
                    )[0]
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps):
                    progress_bar.update()

        # 6. Post process latents to get video
        latents = latents.to(self.vae.model.dtype)
        latents_mean = (
            torch.tensor(self.vae.model.config.latents_mean)
            .view(1, self.vae.model.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.model.config.latents_std).view(
            1, self.vae.model.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        video = self.decode_latents(latents)
        return video

    def encode_texts(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_prompt_embeds(
                prompt=prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_prompt_embeds(
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        self.text_prompt_checks(prompt, negative_prompt, prompt_embeds, negative_prompt_embeds)

    def get_timesteps(self, num_inference_steps, timesteps, strength=0.7):
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start * self.scheduler.order:]
        return timesteps, num_inference_steps - t_start

    def prompt_preprocess(self, prompt):

        def basic_clean(text):
            text = ftfy.fix_text(text)
            text = html.unescape(html.unescape(text))
            return text.strip()

        def whitespace_clean(text):
            text = re.sub(r"\s+", " ", text)
            text = text.strip()

            return text

        return whitespace_clean(basic_clean(prompt))

    def prepare_image_latents(self, batch_size, image, device, dtype):
        to_tensor = v2.ToTensor()
        image = torch.stack(to_tensor(image), dim=0)
        h, w = image.shape[-2:]

        max_area = self.height * self.width
        aspect_ratio = h / w
        latent_h = round(
            math.sqrt(max_area * aspect_ratio)
            // self.vae_scale_factor_spatial
            // self.patch_size[1]
            // self.cp_size
            * self.patch_size[1]
            * self.cp_size
        )
        latent_w = round(
            math.sqrt(max_area / aspect_ratio)
            // self.vae_scale_factor_spatial
            // self.patch_size[2]
            // self.cp_size
            * self.patch_size[2]
            * self.cp_size
        )

        h = latent_h * self.vae_scale_factor_spatial
        w = latent_w * self.vae_scale_factor_spatial

        shape = (
            batch_size,
            self.vae.model.config.z_dim,
            (self.num_frames - 1) // self.vae_scale_factor_temporal + 1,
            latent_h,
            latent_w,
        )

        noise = self.prepare_latents(shape, generator=self.generator, device=device, dtype=dtype)
        msk = torch.ones(batch_size, self.num_frames, latent_h, latent_w).to(dtype=dtype, device=device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(-1, msk.shape[1] // 4, 4, latent_h, latent_w).transpose(1, 2)

        # clip encode
        clip_transform = v2.Compose(
            [
                v2.Resize(size=[224, 224]),
                v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ]
        )
        clip_input = clip_transform(image).to(device=device, dtype=dtype)
        clip_feature = self.image_encoder(clip_input, output_hidden_states=True).hidden_states[-2]

        # vae encode
        vae_transform = v2.Compose([v2.Resize(size=[h, w]), v2.Normalize(mean=[0.5], std=[0.5])])
        vae_input = vae_transform(image)
        vae_input = torch.concat(
            [vae_input.unsqueeze(2), torch.zeros(batch_size, 3, self.num_frames - 1, h, w)], dim=2
        ).to(device=device, dtype=dtype)
        vae_feature = self.vae.encode(vae_input)
        vae_feature = torch.concat([msk, vae_feature], dim=1)

        return noise, clip_feature, vae_feature

    def prepare_video_latents(self, batch_size, video, device, dtype):
        # process video data
        video = torch.stack(video, dim=0)
        video = video.to(torch.float32) / 255.0 # normalize to [0, 1]
        video = video.permute(0, 2, 1, 3, 4)

        h, w = video.shape[-2:]

        max_area = self.height * self.width
        aspect_ratio = h / w
        latent_h = round(
            math.sqrt(max_area * aspect_ratio)
            // self.vae_scale_factor_spatial
            // self.patch_size[1]
            // self.cp_size
            * self.patch_size[1]
            * self.cp_size
        )
        latent_w = round(
            math.sqrt(max_area / aspect_ratio)
            // self.vae_scale_factor_spatial
            // self.patch_size[2]
            // self.cp_size
            * self.patch_size[2]
            * self.cp_size
        )

        h = latent_h * self.vae_scale_factor_spatial
        w = latent_w * self.vae_scale_factor_spatial
        num_latent_frames = (video.size(2) - 1) // self.vae_scale_factor_temporal + 1

        # vae encode
        vae_transform = v2.Compose([v2.Resize(size=[h, w])])
        video = vae_transform(video).to(device)

        ## normalize to [-1, 1]
        video = 2.0 * video - 1.0
        vae_feature = self.vae.encode(video).to(dtype)

        shape = (
            batch_size,
            self.vae.model.config.z_dim,
            num_latent_frames,
            latent_h,
            latent_w,
        )

        noise = self.prepare_latents(shape, generator=self.generator, device=device, dtype=dtype)

        return noise, vae_feature

    def _get_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        prompt = [self.prompt_preprocess(u) for u in prompt]
        batch_size = len(prompt)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)

        return prompt_embeds.to(self.predict_model.dtype)
