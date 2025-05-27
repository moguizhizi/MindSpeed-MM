import torch
from transformers import CLIPVisionModel
from megatron.training import get_args
from mindspeed_mm.data.data_utils.transform_pipeline import get_transforms


class WanVideoI2VProcessor(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_encoder = CLIPVisionModel.from_pretrained(config["image_encoder"]).eval()

        args = get_args()
        first_frame_clip_preprocess = {
            "video": args.mm.data.dataset_param.preprocess_parameters.train_pipeline.first_frame_clip
        }

        first_frame_vae_preprocess = {
            "video": args.mm.data.dataset_param.preprocess_parameters.train_pipeline.first_frame_vae
        }

        global_shape_info = {
            "max_height": args.mm.data.dataset_param.preprocess_parameters.max_height,
            "max_width": args.mm.data.dataset_param.preprocess_parameters.max_width,
            "max_hxw": args.mm.data.dataset_param.preprocess_parameters.max_hxw,
        }

        self.first_frame_clip_transform = get_transforms(
            is_video=True, train_pipeline=first_frame_clip_preprocess, transform_size=global_shape_info
        )

        self.first_frame_vae_transform = get_transforms(
            is_video=True, train_pipeline=first_frame_vae_preprocess, transform_size=global_shape_info
        )

        # i2v_vae_encode_tiling mode:
        # 1. auto: image encode和video encode的配置保持一致
        # 2. true: image encode时强制开启
        # 3. false: image encode时强制关闭
        self.enable_i2v_vae_encode_tiling = config.get("i2v_vae_encode_tiling", "auto")

    def __call__(self, vae_model, videos, first_frame, **kwargs):
        image_encoder_input = self.first_frame_clip_transform(first_frame).to(
            dtype=self.image_encoder.dtype, device=self.image_encoder.device
        )
        clip_features = self.image_encoder(image_encoder_input, output_hidden_states=True).hidden_states[-2]

        bs, _, t, h, w = videos.shape
        mask = torch.ones(bs, t, h // 8, w // 8, device=videos.device)
        mask[:, 1:] = 0
        mask = torch.concat([torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]], dim=1)
        mask = mask.view(bs, mask.shape[1] // 4, 4, h // 8, w // 8).transpose(1, 2)

        vae_input = torch.concat(
            [self.first_frame_vae_transform(first_frame).unsqueeze(2), torch.zeros(bs, 3, t - 1, h, w)], dim=2
        ).to(dtype=videos.dtype, device=videos.device)

        # set vae tiling mode for i2v processor
        vae_model_tiling_state = vae_model.get_tiling_state()
        if self.enable_i2v_vae_encode_tiling != "auto" and self.enable_i2v_vae_encode_tiling != vae_model_tiling_state:
            self.set_vae_tiling_state(vae_model, self.enable_i2v_vae_encode_tiling)

        vae_features = vae_model.encode(vae_input)
        vae_features = torch.concat([mask.to(vae_features.dtype), vae_features], dim=1)

        # back vae tiling mode for video encode
        if vae_model.get_tiling_state() != vae_model_tiling_state:
            self.set_vae_tiling_state(vae_model, vae_model_tiling_state)
        
        return {
            "i2v_clip_feature": clip_features,
            "i2v_vae_feature": vae_features
        }
    
    def set_vae_tiling_state(self, vae_model, use_tiling):
        if use_tiling:
            vae_model.enable_tiling()
        else:
            vae_model.disable_tiling()
