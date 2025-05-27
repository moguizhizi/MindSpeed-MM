import math
from typing import Tuple

import os
import torch
import safetensors
from torch import nn
from einops import rearrange
import numpy as np

from megatron.core import mpu
from megatron.training import print_rank_0
from mindspeed_mm.models.common.activations import Sigmoid
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.conv import Conv2d, CausalConv3d, ContextParallelCausalConv3d
from mindspeed_mm.models.common.normalize import normalize, Normalize3D
from mindspeed_mm.models.common.attention import CausalConv3dAttnBlock
from mindspeed_mm.models.common.resnet_block import ResnetBlock2D, ResnetBlock3D, ContextParallelResnetBlock3D
from mindspeed_mm.models.common.updownsample import (SpatialDownsample2x, TimeDownsample2x, SpatialUpsample2x,
                                                     TimeUpsample2x,
                                                     TimeUpsampleRes2x, Upsample3D, Downsample, DownSample3D,
                                                     Spatial2xTime2x3DDownsample, Spatial2xTime2x3DUpsample)
from mindspeed_mm.models.common.regularizer import DiagonalGaussianDistribution
from mindspeed_mm.models.common.communications import _conv_split, _conv_gather
from mindspeed_mm.utils.utils import (
    is_context_parallel_initialized,
    initialize_context_parallel,
    get_context_parallel_group,
    get_context_parallel_group_rank,
    get_context_parallel_world_size,
    get_context_parallel_rank
)

CASUAL_VAE_MODULE_MAPPINGS = {
    "Conv2d": Conv2d,
    "ResnetBlock2D": ResnetBlock2D,
    "CausalConv3d": CausalConv3d,
    "AttnBlock3D": CausalConv3dAttnBlock,
    "ResnetBlock3D": ResnetBlock3D,
    "Downsample": Downsample,
    "SpatialDownsample2x": SpatialDownsample2x,
    "TimeDownsample2x": TimeDownsample2x,
    "SpatialUpsample2x": SpatialUpsample2x,
    "TimeUpsample2x": TimeUpsample2x,
    "TimeUpsampleRes2x": TimeUpsampleRes2x,
    "Spatial2xTime2x3DDownsample": Spatial2xTime2x3DDownsample,
    "Spatial2xTime2x3DUpsample": Spatial2xTime2x3DUpsample,
    "SiLU": nn.SiLU,
    "swish": Sigmoid,
    "ContextParallelResnetBlock3D": ContextParallelResnetBlock3D,
    "ContextParallelCausalConv3d": ContextParallelCausalConv3d,
    "DownSample3D": DownSample3D,
    "Upsample3D": Upsample3D
}


def model_name_to_cls(model_name):
    if model_name in CASUAL_VAE_MODULE_MAPPINGS:
        return CASUAL_VAE_MODULE_MAPPINGS[model_name]
    else:
        raise ValueError(f"Model name {model_name} not supported")


class ContextParallelCasualVAE(MultiModalModule):
    def __init__(
        self,
        from_pretrained: str = None,
        cp_size: str = 0,
        hidden_size: int = 128,
        z_channels: int = 4,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (),
        dropout: float = 0.0,
        resolution: int = 256,
        double_z: bool = True,
        embed_dim: int = 4,
        num_res_blocks: int = 2,
        q_conv: str = "CausalConv3d",
        conv_padding: int = 1,
        encoder_conv_in: str = "CausalConv3d",
        encoder_conv_out: str = "CausalConv3d",
        encoder_attention: str = "AttnBlock3D",
        encoder_nonlinearity: str = "SiLU",
        encoder_resnet_blocks: Tuple[str] = (
                "ResnetBlock3D",
                "ResnetBlock3D",
                "ResnetBlock3D",
                "ResnetBlock3D",
        ),
        encoder_spatial_downsample: Tuple[str] = (
                "SpatialDownsample2x",
                "SpatialDownsample2x",
                "SpatialDownsample2x",
                "",
        ),
        encoder_temporal_downsample: Tuple[str] = (
                "",
                "TimeDownsample2x",
                "TimeDownsample2x",
                "",
        ),
        encoder_mid_resnet: str = "ResnetBlock3D",
        decoder_conv_in: str = "CausalConv3d",
        decoder_conv_out: str = "CausalConv3d",
        decoder_attention: str = "AttnBlock3D",
        decoder_nonlinearity: str = "SiLU",
        decoder_resnet_blocks: Tuple[str] = (
                "ResnetBlock3D",
                "ResnetBlock3D",
                "ResnetBlock3D",
                "ResnetBlock3D",
        ),
        decoder_spatial_upsample: Tuple[str] = (
                "",
                "SpatialUpsample2x",
                "SpatialUpsample2x",
                "SpatialUpsample2x",
        ),
        decoder_temporal_upsample: Tuple[str] = ("", "", "TimeUpsample2x", "TimeUpsample2x"),
        decoder_mid_resnet: str = "ResnetBlock3D",
        tile_sample_min_size: int = 256,
        tile_sample_min_size_t: int = 33,
        tile_latent_min_size_t: int = 16,
        tile_overlap_factor: int = 0.125,
        vae_scale_factor: list = None,
        use_tiling: bool = False,
        use_quant_layer: bool = True,
        encoder_gather_norm: bool = False,
        decoder_gather_norm: bool = False,
        scale_factor: float = 0.7,
        tile_sample_min_height: int = 480,
        tile_sample_min_width: int = 720,
        tile_overlap_factor_height: float = 1 / 6,
        tile_overlap_factor_width: float = 1 / 5,
        frame_batch_size: int = 8,
        **kwargs
    ) -> None:
        super().__init__(config=None)
        self.cp_size = cp_size
        self.tile_sample_min_size = tile_sample_min_size
        self.tile_sample_min_size_t = tile_sample_min_size_t
        self.tile_latent_min_size = int(self.tile_sample_min_size / (2 ** (len(hidden_size_mult) - 1)))
        self.hidden_size_mult = hidden_size_mult

        self.tile_latent_min_size_t = tile_latent_min_size_t
        self.tile_overlap_factor = tile_overlap_factor
        self.vae_scale_factor = vae_scale_factor
        self.scale_factor = scale_factor
        self.use_tiling = use_tiling
        self.use_quant_layer = use_quant_layer

        self.encoder = Encoder(
            z_channels=z_channels,
            hidden_size=hidden_size,
            hidden_size_mult=hidden_size_mult,
            attn_resolutions=attn_resolutions,
            conv_in=encoder_conv_in,
            conv_out=encoder_conv_out,
            conv_padding=conv_padding,
            nonlinearity=encoder_nonlinearity,
            attention=encoder_attention,
            resnet_blocks=encoder_resnet_blocks,
            spatial_downsample=encoder_spatial_downsample,
            temporal_downsample=encoder_temporal_downsample,
            mid_resnet=encoder_mid_resnet,
            dropout=dropout,
            resolution=resolution,
            num_res_blocks=num_res_blocks,
            double_z=double_z,
            gather_norm=encoder_gather_norm
        )

        self.decoder = Decoder(
            z_channels=z_channels,
            hidden_size=hidden_size,
            hidden_size_mult=hidden_size_mult,
            attn_resolutions=attn_resolutions,
            conv_in=decoder_conv_in,
            conv_out=decoder_conv_out,
            conv_padding=conv_padding,
            nonlinearity=decoder_nonlinearity,
            attention=decoder_attention,
            resnet_blocks=decoder_resnet_blocks,
            spatial_upsample=decoder_spatial_upsample,
            temporal_upsample=decoder_temporal_upsample,
            mid_resnet=decoder_mid_resnet,
            dropout=dropout,
            resolution=resolution,
            num_res_blocks=num_res_blocks,
            gather_norm=decoder_gather_norm
        )
        if self.use_quant_layer:
            quant_conv_cls = model_name_to_cls(q_conv)
            self.quant_conv = quant_conv_cls(2 * z_channels, 2 * embed_dim, 1)
            self.post_quant_conv = quant_conv_cls(embed_dim, z_channels, 1)

        if from_pretrained is not None:
            self.load_checkpoint(from_pretrained)

        self.dp_group_nums = torch.distributed.get_world_size() // mpu.get_data_parallel_world_size()

        if self.cp_size > 0:
            if not is_context_parallel_initialized():
                initialize_context_parallel(self.cp_size)

        # tiling parameters
        self.tile_sample_min_height = tile_sample_min_height // 2
        self.tile_sample_min_width = tile_sample_min_width // 2
        self.tile_overlap_factor_height = tile_overlap_factor_height
        self.tile_overlap_factor_width = tile_overlap_factor_width
        self.frame_batch_size = frame_batch_size


    def get_encoder(self):
        if self.use_quant_layer:
            return [self.quant_conv, self.encoder]
        return [self.encoder]

    def get_decoder(self):
        if self.use_quant_layer:
            return [self.post_quant_conv, self.decoder]
        return [self.decoder]

    def _bs_split_and_pad(self, x, split_size):
        bs = x.shape[0]
        remain = bs % split_size
        if remain == 0:
            return torch.tensor_split(x, split_size, dim=0)
        else:
            print_rank_0(f"[WARNING]: data batch size {bs} is not divisible by split size {split_size}, which may cause waste!")
            x = torch.cat([x, x[-1:].repeat_interleave(split_size - remain, dim=0)], dim=0)
            return torch.tensor_split(x, split_size, dim=0)

    def encode(self, x, enable_cp=True, invert_scale_latents=False, generator=None, **kwargs):
        if self.cp_size <= 1:
            enable_cp = False
        if not enable_cp:
            return self._encode(x, enable_cp=False, invert_scale_latents=invert_scale_latents, generator=generator, **kwargs)

        if self.cp_size % self.dp_group_nums == 0 and self.cp_size > self.dp_group_nums:
            # loop cp
            data_list = [torch.empty_like(x) for _ in range(self.cp_size)]
            data_list[get_context_parallel_rank()] = x
            torch.distributed.all_gather(data_list, x, group=get_context_parallel_group())
            data_list = data_list[::self.dp_group_nums]
            latents = []
            for data in data_list:
                latents.append(self._encode(data, enable_cp=enable_cp, invert_scale_latents=invert_scale_latents, generator=generator, **kwargs))
            return latents[get_context_parallel_rank() // self.dp_group_nums]

        elif self.dp_group_nums % self.cp_size == 0 and self.cp_size < self.dp_group_nums:
            # split
            bs = x.shape[0]
            data_list = self._bs_split_and_pad(x, self.dp_group_nums // self.cp_size)
            data = data_list[get_context_parallel_rank() % (self.dp_group_nums // self.cp_size)]

            _latent = self._encode(data, enable_cp=enable_cp, invert_scale_latents=invert_scale_latents, generator=generator, **kwargs)

            if mpu.get_tensor_model_parallel_world_size() > 1:
                latents_tp = [torch.empty_like(_latent) for _ in range(mpu.get_tensor_model_parallel_world_size())]
                torch.distributed.all_gather(latents_tp, _latent, group=mpu.get_tensor_model_parallel_group())
                latents_tp = torch.cat(latents_tp, dim=0)
            else:
                latents_tp = _latent

            if mpu.get_context_parallel_world_size() > 1:
                latents_cp = [torch.empty_like(latents_tp) for _ in range(mpu.get_context_parallel_world_size())]
                torch.distributed.all_gather(latents_cp, latents_tp, group=mpu.get_context_parallel_group())
                latents = torch.cat(latents_cp, dim=0)
            else:
                latents = latents_tp

            latents = latents[::self.cp_size]
            return latents[:bs]

        elif self.cp_size == self.dp_group_nums:
            return self._encode(x, enable_cp=enable_cp, invert_scale_latents=invert_scale_latents, generator=generator, **kwargs)
        else:
            raise NotImplementedError(f"Not supported megatron data parallel group nums {self.dp_group_nums} and VAE cp_size {self.cp_size}!")

    def _encode(self, x, enable_cp=True, invert_scale_latents=False, generator=None, **kwargs):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        _, _, _, height, width = x.shape

        if self.use_tiling and enable_cp:
            raise NotImplementedError("Tiling and vae-cp cannot be supported at the same time.")

        if self.use_tiling:
            if width > self.tile_sample_min_width or height > self.tile_sample_min_height:
                tiled_output = self.tiled_encode(x, enable_cp=enable_cp)

                posterior = DiagonalGaussianDistribution(tiled_output)
                res = posterior.sample(generator=generator)
                if not invert_scale_latents:
                    res = self.scale_factor * res
                return res
            else:
                print_rank_0("Not use tiling encode.")

        if self.cp_size > 0 and enable_cp:
            global_src_rank = get_context_parallel_group_rank() * self.cp_size
            torch.distributed.broadcast(x, src=global_src_rank, group=get_context_parallel_group())
            x = _conv_split(x, dim=2, kernel_size=1)

        h, conv_cache = self.encoder(x, enable_cp=enable_cp, conv_cache=None)
        if self.use_quant_layer:
            h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)

        res = posterior.sample(generator=generator)

        if self.cp_size > 0 and enable_cp:
            res = _conv_gather(res, dim=2, kernel_size=1)

        if not invert_scale_latents:
            res = self.scale_factor * res

        return res

    def decode(self, z, enable_cp: bool = True, **kwargs):
        if self.cp_size <= 1:
            enable_cp = False
        if self.cp_size > 0 and enable_cp:
            global_src_rank = get_context_parallel_group_rank() * self.cp_size
            torch.distributed.broadcast(z, src=global_src_rank, group=get_context_parallel_group())

            z = _conv_split(z, dim=2, kernel_size=1)

        if self.use_tiling:
            if (z.shape[-1] > self.tile_latent_min_size
                    or z.shape[-2] > self.tile_latent_min_size
                    or z.shape[-3] > self.tile_latent_min_size_t):
                dec = self.tiling_decode(z, enable_cp=enable_cp, cogvideo_version=kwargs.get("cogvideo_version", 1.0))
        else:
            if self.use_quant_layer:
                z = self.post_quant_conv(z)
            dec = self.decoder(z, enable_cp=enable_cp)

        if self.cp_size > 0 and enable_cp:
            dec = _conv_gather(dec, dim=2, kernel_size=1)

        return dec

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + \
                               b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + \
                               b[:, :, :, :, x] * (x / blend_extent)
        return b

    def tiling_decode(self, z, enable_cp=True, cogvideo_version=1.0):
        n_samples = z.shape[0]
        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        for n in range(n_rounds):
            temp_z = z[n * n_samples : (n + 1) * n_samples, :, :]
            latent_time = temp_z.shape[2]  # check the time latent
            fake_cp_size = min(10, latent_time // 2)

            recons = []
            start_frame = 0
            split_first_frame = cogvideo_version > 1.0
            for i in range(fake_cp_size):
                end_frame = start_frame + latent_time // fake_cp_size + (1 if i < latent_time % fake_cp_size else 0)
                use_cp = True if i == 0 and enable_cp else False
                clear_fake_cp_cache = True if i == fake_cp_size - 1 else False
                with torch.no_grad():
                    recon = self.decoder(
                        temp_z[:, :, start_frame:end_frame].contiguous(),
                        clear_fake_cp_cache=clear_fake_cp_cache,
                        enable_cp=use_cp,
                        split_first_frame=split_first_frame
                    )
                recons.append(recon)
                start_frame = end_frame
            recons = torch.cat(recons, dim=2)
            all_out.append(recons)
        out = torch.cat(all_out, dim=0)
        return out

    def tiled_encode(self, x, enable_cp=True):
        batch_size, num_channels, num_frames, height, width = x.shape
        tile_latent_min_height = int(self.tile_sample_min_height / (2 ** (len(self.hidden_size_mult) - 1)))
        tile_latent_min_width = int(self.tile_sample_min_width / (2 ** (len(self.hidden_size_mult) - 1)))

        overlap_height = int(self.tile_sample_min_height * (1 - self.tile_overlap_factor_height))
        overlap_width = int(self.tile_sample_min_width * (1 - self.tile_overlap_factor_width))
        blend_extent_height = int(tile_latent_min_height * self.tile_overlap_factor_height)
        blend_extent_width = int(tile_latent_min_width * self.tile_overlap_factor_width)
        row_limit_height = tile_latent_min_height - blend_extent_height
        row_limit_width = tile_latent_min_width - blend_extent_width

        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                num_batches = max(num_frames // self.frame_batch_size, 1)
                conv_cache = None
                time = []

                for k in range(num_batches):
                    remaining_frames = num_frames % self.frame_batch_size
                    start_frame = self.frame_batch_size * k + (0 if k == 0 else remaining_frames)
                    end_frame = self.frame_batch_size * (k + 1) + remaining_frames
                    tile = x[
                           :,
                           :,
                           start_frame:end_frame,
                           i: i + self.tile_sample_min_height,
                           j: j + self.tile_sample_min_width,
                           ]
                    tile, conv_cache = self.encoder(tile, conv_cache=conv_cache, enable_cp=enable_cp, use_conv_cache=True)
                    if self.use_quant_layer:
                        tile = self.quant_conv(tile)
                    time.append(tile)

                row.append(torch.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent_width)
                result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
            result_rows.append(torch.cat(result_row, dim=4))

        moments = torch.cat(result_rows, dim=3)
        return moments

    def tiled_decode(self, x, enable_cp=True):
        t = x.shape[2]
        t_chunk_idx = [i for i in range(0, t, self.tile_latent_min_size_t - 1)]
        if len(t_chunk_idx) == 1 and t_chunk_idx[0] == 0:
            t_chunk_start_end = [[0, t]]
        else:
            t_chunk_start_end = [[t_chunk_idx[i], t_chunk_idx[i + 1] + 1] for i in range(len(t_chunk_idx) - 1)]
            if t_chunk_start_end[-1][-1] > t:
                t_chunk_start_end[-1][-1] = t
            elif t_chunk_start_end[-1][-1] < t:
                last_start_end = [t_chunk_idx[-1], t]
                t_chunk_start_end.append(last_start_end)
        dec_ = []
        for idx, (start, end) in enumerate(t_chunk_start_end):
            chunk_x = x[:, :, start: end]
            if idx != 0:
                dec = self.tiled_decode2d(chunk_x, enable_cp=enable_cp)[:, :, 1:]
            else:
                dec = self.tiled_decode2d(chunk_x, enable_cp=enable_cp)
            dec_.append(dec)
        dec_ = torch.cat(dec_, dim=2)
        return dec_


    def tiled_decode2d(self, z, enable_cp=True):
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[3], overlap_size):
            row = []
            for j in range(0, z.shape[4], overlap_size):
                tile = z[:, :, :,
                       i: i + self.tile_latent_min_size,
                       j: j + self.tile_latent_min_size,
                       ]
                if self.use_quant_layer:
                    tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile, enable_cp=enable_cp)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)
        return dec

    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling

    def disable_tiling(self):
        self.enable_tiling(False)

    def load_checkpoint(self, ckpt_path):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Could not find checkpoint at {ckpt_path}")

        if ckpt_path.endswith("pt") or ckpt_path.endswith("pth"):
            ckpt_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        elif ckpt_path.endswith(".safetensors"):
            ckpt_dict = safetensors.torch.load_file(ckpt_path)
        else:
            raise ValueError(f"Invalid checkpoint path: {ckpt_path}")

        if "state_dict" in ckpt_dict.keys():
            ckpt_dict = ckpt_dict["state_dict"]

        missing_keys, unexpected_keys = self.load_state_dict(ckpt_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")


class Encoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        hidden_size: int,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (16,),
        conv_in: str = "Conv2d",
        conv_out: str = "CasualConv3d",
        conv_padding: int = 1,
        attention: str = "AttnBlock2D",
        resnet_blocks: Tuple[str] = (
                "ResnetBlock2D",
                "ResnetBlock2D",
                "ResnetBlock2D",
                "ResnetBlock3D",
        ),
        spatial_downsample: Tuple[str] = (
                "Downsample",
                "Downsample",
                "Downsample",
                "",
        ),
        temporal_downsample: Tuple[str] = ("", "", "TimeDownsampleRes2x", ""),
        mid_resnet: str = "ResnetBlock3D",
        nonlinearity: str = "SiLU",
        dropout: float = 0.0,
        resolution: int = 256,
        num_res_blocks: int = 2,
        double_z: bool = True,
        temporal_compress_times: int = 4,
        gather_norm: bool = False,
    ) -> None:
        super().__init__()
        if len(resnet_blocks) != len(hidden_size_mult):
            raise AssertionError(f"the length of resnet_blocks and hidden_size_mult must be equal")
        # ---- Config ----
        self.temb_ch = 0
        self.num_resolutions = len(hidden_size_mult)
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks
        self.enable_nonlinearity = nonlinearity
        self.enbale_attn1 = attention
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        # ---- Nonlinearity ----
        if self.enable_nonlinearity:
            self.nonlinearity = model_name_to_cls(nonlinearity)()

        # ---- In ----
        self.conv_in = model_name_to_cls(conv_in)(
            3, hidden_size, kernel_size=3, stride=1, padding=conv_padding
        )

        # ---- Downsample ----
        curr_res = resolution
        in_ch_mult = (1,) + tuple(hidden_size_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = hidden_size * in_ch_mult[i_level]  # [1,1,2,2,4]
            block_out = hidden_size * hidden_size_mult[i_level]  # [1,2,2,4]
            for _ in range(self.num_res_blocks):
                block.append(
                    model_name_to_cls(resnet_blocks[i_level])(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        gather_norm=gather_norm,
                        temb_channels=self.temb_ch,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(model_name_to_cls(attention)(
                        in_channels=block_in,
                        out_channels=block_in
                    )
                    )
            down = nn.Module()
            down.block = block
            down.attn = attn

            if i_level != self.num_resolutions - 1:  # 3,最后一个downsample不压缩
                if i_level < self.temporal_compress_level:
                    down.downsample = DownSample3D(block_in, compress_time=True)
                else:
                    down.downsample = DownSample3D(block_in, compress_time=False)
                curr_res = curr_res // 2

            if temporal_downsample[i_level]:
                down.time_downsample = model_name_to_cls(temporal_downsample[i_level])(
                    block_in, block_in
                )
            self.down.append(down)

        # ---- Mid ----
        self.mid = nn.Module()

        self.mid.block_1 = model_name_to_cls(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            gather_norm=gather_norm,
            temb_channels=self.temb_ch,
        )

        if self.enbale_attn1:
            self.mid.attn_1 = model_name_to_cls(attention)(
                in_channels=block_in,
                out_channels=block_in
            )

        self.mid.block_2 = model_name_to_cls(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            gather_norm=gather_norm,
            temb_channels=self.temb_ch,
        )
        # ---- Out ----
        self.norm_out = normalize(block_in, gather=gather_norm)

        self.conv_out = model_name_to_cls(conv_out)(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=conv_padding
        )

    def forward(self, x, enable_cp=True, conv_cache=None, use_conv_cache=False):
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        h, new_conv_cache["conv_in"] = self.conv_in(x, enable_cp=enable_cp,
                                                    conv_cache=conv_cache.get("conv_cache"),
                                                    use_conv_cache=use_conv_cache)

        # 1. Down
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                conv_cache_key = f"resnet_{i_level}_{i_block}"
                h, new_conv_cache[conv_cache_key] = self.down[i_level].block[i_block](h, enable_cp=enable_cp,
                                                                                      conv_cache=conv_cache.get(conv_cache_key),
                                                                                      use_conv_cache=use_conv_cache)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h, enable_cp=enable_cp)

        # 2. Mid
        h, new_conv_cache["resnet_1"] = self.mid.block_1(h, enable_cp=enable_cp,
                                                         conv_cache=conv_cache.get("resnet_1"),
                                                         use_conv_cache=use_conv_cache)
        if self.enbale_attn1:
            h = self.mid.attn_1(h, enable_cp=enable_cp)
        h, new_conv_cache["resnet_2"] = self.mid.block_2(h, enable_cp=enable_cp,
                                                         conv_cache=conv_cache.get("resnet_2"),
                                                         use_conv_cache=use_conv_cache)

        # 3. Post-process
        h = self.norm_out(h, enable_cp=enable_cp)
        if self.enable_nonlinearity:
            h = self.nonlinearity(h)
        h, new_conv_cache["conv_out"] = self.conv_out(h, enable_cp=enable_cp,
                                                      conv_cache=conv_cache.get("conv_out"),
                                                      use_conv_cache=use_conv_cache)
        return h, new_conv_cache


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        hidden_size: int,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (16,),
        conv_in: str = "Conv2d",
        conv_out: str = "CasualConv3d",
        conv_padding: int = 1,
        attention: str = "AttnBlock2D",
        resnet_blocks: Tuple[str] = (
                "ResnetBlock3D",
                "ResnetBlock3D",
                "ResnetBlock3D",
                "ResnetBlock3D",
        ),
        spatial_upsample: Tuple[str] = (
                "",
                "SpatialUpsample2x",
                "SpatialUpsample2x",
                "SpatialUpsample2x",
        ),
        temporal_upsample: Tuple[str] = ("", "", "", "TimeUpsampleRes2x"),
        mid_resnet: str = "ResnetBlock3D",
        nonlinearity: str = "SiLU",
        dropout: float = 0.0,
        resolution: int = 256,
        num_res_blocks: int = 2,
        temporal_compress_times: int = 4,
        gather_norm: bool = False,
    ):
        super().__init__()
        # ---- Config ----
        self.temb_ch = 0
        self.num_resolutions = len(hidden_size_mult)
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks
        self.enable_attention = attention
        self.enable_nonlinearity = nonlinearity

        # ---- Nonlinearity ----
        if self.enable_nonlinearity:
            self.nonlinearity = model_name_to_cls(nonlinearity)()

        # log2 of temporal compress_times
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        # ---- In ----
        block_in = hidden_size * hidden_size_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.conv_in = model_name_to_cls(conv_in)(
            z_channels, block_in, kernel_size=3, padding=conv_padding
        )

        # ---- Mid ----
        self.mid = nn.Module()
        self.mid.block_1 = model_name_to_cls(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            zq_ch=z_channels,
            dropout=dropout,
            gather_norm=gather_norm,
            temb_channels=self.temb_ch,
            normalization=Normalize3D
        )
        if self.enable_attention:
            self.mid.attn_1 = model_name_to_cls(attention)(
                in_channels=block_in,
                out_channels=block_in
            )
        self.mid.block_2 = model_name_to_cls(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            zq_ch=z_channels,
            dropout=dropout,
            gather_norm=gather_norm,
            temb_channels=self.temb_ch,
            normalization=Normalize3D
        )

        # ---- Upsample ----
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = hidden_size * hidden_size_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    model_name_to_cls(resnet_blocks[i_level])(
                        in_channels=block_in,
                        out_channels=block_out,
                        zq_ch=z_channels,
                        dropout=dropout,
                        gather_norm=gather_norm,
                        temb_channels=self.temb_ch,
                        normalization=Normalize3D
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(model_name_to_cls(attention)(
                        in_channels=block_in,
                        out_channels=block_in
                    )
                    )
            up = nn.Module()
            up.block = block
            up.attn = attn
            if spatial_upsample[i_level]:
                compress_time = i_level >= self.num_resolutions - self.temporal_compress_level
                up.upsample = model_name_to_cls(spatial_upsample[i_level])(
                    block_in, block_in, compress_time=compress_time
                )
                curr_res = curr_res * 2
            if temporal_upsample[i_level]:
                up.time_upsample = model_name_to_cls(temporal_upsample[i_level])(
                    block_in, block_in
                )
            self.up.insert(0, up)

        # ---- Out ----
        self.norm_out = Normalize3D(block_in, z_channels, gather=gather_norm)
        self.conv_out = model_name_to_cls(conv_out)(
            block_in, 3, kernel_size=3, padding=conv_padding
        )

    def forward(self, z, enable_cp=True, clear_fake_cp_cache=True, **kwargs):
        zq = z
        is_encode = False
        h, _ = self.conv_in(z, clear_cache=clear_fake_cp_cache)
        h, _ = self.mid.block_1(h, zq=zq, clear_fake_cp_cache=clear_fake_cp_cache, enable_cp=enable_cp, is_encode=is_encode)
        if self.enable_attention:
            h = self.mid.attn_1(h)
        h, _ = self.mid.block_2(h, zq=zq, clear_fake_cp_cache=clear_fake_cp_cache, enable_cp=enable_cp, is_encode=is_encode)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h, _ = self.up[i_level].block[i_block](h, zq=zq, clear_fake_cp_cache=clear_fake_cp_cache,
                                                    enable_cp=enable_cp, is_encode=is_encode)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq=zq)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h, enable_cp=enable_cp)
            if hasattr(self.up[i_level], "time_upsample"):
                h = self.up[i_level].time_upsample(h, enable_cp=enable_cp)

        h = self.norm_out(h, zq=zq, clear_fake_cp_cache=clear_fake_cp_cache, enable_cp=enable_cp)
        if self.enable_nonlinearity:
            h = self.nonlinearity(h)
        h, _ = self.conv_out(h, clear_cache=clear_fake_cp_cache)
        return h
