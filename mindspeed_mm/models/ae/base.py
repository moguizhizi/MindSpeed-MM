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


import torch.nn as nn

from mindspeed_mm.utils.extra_processor import I2VProcessor
from mindspeed_mm.models.ae.diffusers_ae_model import DiffusersAEModel
from mindspeed_mm.models.ae.vae import VideoAutoencoderKL, VideoAutoencoder3D
from mindspeed_mm.models.ae.casualvae import CausalVAE
from mindspeed_mm.models.ae.wfvae import WFVAE
from mindspeed_mm.models.ae.contextparallel_causalvae import ContextParallelCasualVAE
from mindspeed_mm.models.ae.autoencoder_kl_hunyuanvideo import AutoencoderKLHunyuanVideo
from mindspeed_mm.models.ae.wan_video_vae import WanVideoVAE
from mindspeed_mm.models.ae.stepvideo_vae import StepVideoVae


AE_MODEL_MAPPINGS = {
    "vae": VideoAutoencoderKL,
    "vae3D": VideoAutoencoder3D,
    "casualvae": CausalVAE,
    "wfvae": WFVAE,
    "contextparallelcasualvae": ContextParallelCasualVAE,
    "autoencoder_kl_hunyuanvideo": AutoencoderKLHunyuanVideo,
    "wan_video_vae": WanVideoVAE,
    "stepvideovae": StepVideoVae
}


class AEModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config.to_dict()

        self.i2v_processor_config = self.config.pop("i2v_processor", None)
        if self.i2v_processor_config is not None:
            self.i2v_processor = I2VProcessor(self.i2v_processor_config).get_processor()
        else:
            self.i2v_processor = None

        if config.model_id in AE_MODEL_MAPPINGS:
            self.model = AE_MODEL_MAPPINGS[config.model_id](**self.config)
        else:
            # import auto encoder from diffusers
            self.model = DiffusersAEModel(
                model_name=config.model_id, config=self.config
            )

    def get_model(self):
        return self.model

    def encode(self, x, **kwargs):
        video_latents = self.model.encode(x)
        if self.i2v_processor is None:
            return video_latents, None
        else:
            i2v_results = self.i2v_processor(
                vae_model=self.model, videos=x, video_latents=video_latents, **kwargs
            )
            return video_latents, i2v_results

    def decode(self, x):
        return self.model.decode(x)

    def forward(self, x):
        raise NotImplementedError("forward function is not implemented")
