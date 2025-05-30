import torch

from mindspeed_mm.data.data_utils.constants import INPUT_MASK, MASKED_VIDEO
from mindspeed_mm.utils.mask_utils import MaskCompressor


class OpenSoraPlanI2VProcessor:
    """

    Args:
        config (dict): the processor config
        {
            "vae_scale_factor": [4, 8 ,8]
        }
    """

    def __init__(self, config):
        vae_scale_factor = config.get("vae_scale_factor", [4, 8, 8])
        ae_stride_t, ae_stride_h, ae_stride_w = vae_scale_factor
        self.mask_compressor = MaskCompressor(ae_stride_h=ae_stride_h, ae_stride_w=ae_stride_w, ae_stride_t=ae_stride_t)

    def __call__(self, vae_model, videos, video_latents, **kwargs):
        masked_video = kwargs.get(MASKED_VIDEO, None).to(videos.dtype)
        mask = kwargs.get(INPUT_MASK, None).to(videos.dtype)
        masked_video = vae_model.encode(masked_video)
        mask = self.mask_compressor(mask)

        return {MASKED_VIDEO: masked_video, INPUT_MASK: mask}
