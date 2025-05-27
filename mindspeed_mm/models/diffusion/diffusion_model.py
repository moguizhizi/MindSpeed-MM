from torch import nn

from .ddpm import DDPM
from .iddpm import IDDPM
from .rflow import RFlow
from .cogvideo_diffusion import CogVideoDiffusion
from .flow_match_discrete_scheduler import FlowMatchDiscreteScheduler
from .wan_flow_match_scheduler import WanFlowMatchScheduler
from .hunyuanvideo_i2v_diffusion import HunyuanVideoI2VDiffusion
from .diffusers_scheduler import DIFFUSERS_SCHEDULE_MAPPINGS, DiffusersScheduler


DIFFUSION_MODEL_MAPPINGS = {
    "ddpm": DDPM,
    "iddpm": IDDPM,
    "rflow": RFlow,
    "cogvideo_diffusion": CogVideoDiffusion,
    "flow_match_discrete_scheduler": FlowMatchDiscreteScheduler,
    "wan_flow_match_scheduler": WanFlowMatchScheduler,
    "hunyuanvideo_i2v_diffusion": HunyuanVideoI2VDiffusion,
}


class DiffusionModel:
    """
    Factory class for all customized diffusion models and diffusers schedulers.
    Args:
        config:
        {
            "model_id": "ddpm",
            "num_timesteps": 1000,
            "beta_schedule": "linear",
            ...
        }
    """

    def __init__(self, config):
        if config.model_id in DIFFUSION_MODEL_MAPPINGS:
            model_cls = DIFFUSION_MODEL_MAPPINGS[config.model_id]
            self.diffusion = model_cls(**config.to_dict())
        else:
            self.diffusion = DiffusersScheduler(config.to_dict())

    def get_model(self):
        return self.diffusion
