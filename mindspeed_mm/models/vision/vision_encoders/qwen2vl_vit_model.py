# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLRotaryEmbedding

from megatron.core import mpu
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region, gather_from_sequence_parallel_region
from megatron.training import get_args

from mindspeed.core.context_parallel.unaligned_cp.mapping import cal_split_sizes, split_forward_gather_backward, \
    gather_forward_split_backward
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.vision.vision_encoders.vision_transformer_block import Qwen2VLVisionTransformerBlock

try:
    from mindspeed.utils import set_actual_seq_len
except ImportError:
    set_actual_seq_len = None


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Modified based on transformers.models.qwen2_vl.modeling_qwen2_vl
def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1, use_fused_rope=True):
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    if use_fused_rope:
        import torch_npu
        cos, sin = cos[:1], sin[:1]
        q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
        k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    else:
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Modified based on transformers.models.qwen2_vl.modeling_qwen2_vl
def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor, use_fused_rope=True) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    if use_fused_rope:
        import torch_npu
        output = torch_npu.npu_rotary_mul(tensor, cos, sin).to(orig_dtype)
    else:
        output = ((tensor * cos) + (rotate_half(tensor) * sin)).to(orig_dtype)
    return output


class Qwen2VLRotaryEmbedding_llm(Qwen2VLRotaryEmbedding):
    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__(config=config)

    @torch.no_grad()
    def forward(self, x_device, x_dtype, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x_device)

        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        device_type = x_device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return torch.concat((cos, sin), dim=-1).to(dtype=x_dtype)


class Qwen2vlSelfAttention(SelfAttention):
    def __init__(
            self,
            config: TransformerConfig,
            submodules: SelfAttentionSubmodules,
            layer_number: int,
            attn_mask_type=AttnMaskType.padding
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type
        )

        self.mrope_section = config.mrope_section

    def forward(
            self,
            hidden_states,
            attention_mask,
            key_value_states=None,
            inference_params=None,
            rotary_pos_emb=None,
            packed_seq_params=None,
    ):
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)  # s b h d
        
        if self.config.context_parallel_size > key.shape[2]:
            key = key.repeat_interleave(
                query.shape[2] // key.shape[2], dim=2
            )
            value = value.repeat_interleave(
                query.shape[2] // value.shape[2], dim=2
            )
            
        query = query.permute(1, 2, 0, 3).contiguous()  # b h s d
        key = key.permute(1, 2, 0, 3).contiguous()  # b h s d

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================

        # TODO, can apply positional embedding to value_layer so it has
        # absolute positional embedding.
        # otherwise, only relative positional embedding takes effect
        if rotary_pos_emb is not None:
            half_dim = rotary_pos_emb.shape[-1] // 2
            cos, sin = rotary_pos_emb[..., :half_dim], rotary_pos_emb[..., half_dim:]
            query, key = apply_multimodal_rotary_pos_emb(query, key, cos, sin, self.mrope_section,
                                                         use_fused_rope=self.config.use_fused_rotary_pos_emb)  # b h s d
        query = query.permute(2, 0, 1, 3).contiguous()  # s b h d
        key = key.permute(2, 0, 1, 3).contiguous()  # s b h d
        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        key, value, _, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params, key, value, None
        )
        # ==================================
        # core attention computation
        # ==================================
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )

        if packed_seq_params is not None:
            # reshape to same output shape as unpacked case
            # from (t, np, hn) to (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.linear_proj(core_attn_out)
        return output, bias


class Qwen2vlVitSelfAttention(SelfAttention):
    """
        Self-attention layer class for Qwen2VLVit

        Self-attention layer takes input with size [s, b, h]
        and returns output of the same size.
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: SelfAttentionSubmodules,
            layer_number: int,
            attn_mask_type=AttnMaskType.padding
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type
        )

    def forward(
            self,
            hidden_states,
            attention_mask,
            key_value_states=None,
            inference_params=None,
            rotary_pos_emb=None,
            packed_seq_params=None,
    ):

        # hidden_states: [sq, b, h]
        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)
        
        if self.config.context_parallel_size > key.shape[2]:
            key = key.repeat_interleave(
                query.shape[2] // key.shape[2], dim=2
            )
            value = value.repeat_interleave(
                query.shape[2] // value.shape[2], dim=2
            )
        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params, key, value, rotary_pos_emb
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================

        # TODO, can apply positional embedding to value_layer so it has
        # absolute positional embedding.
        # otherwise, only relative positional embedding takes effect
        if rotary_pos_emb is not None:
            query = apply_rotary_pos_emb_vision(query.transpose(0, 1), rotary_pos_emb[0],
                                                use_fused_rope=self.config.use_fused_rotary_pos_emb).transpose(0, 1)
            key = apply_rotary_pos_emb_vision(key.transpose(0, 1), rotary_pos_emb[0],
                                              use_fused_rope=self.config.use_fused_rotary_pos_emb).transpose(0, 1)

        # ==================================
        # core attention computation
        # ==================================
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )

        if packed_seq_params is not None:
            # reshape to same output shape as unpacked case
            # t is the pack size: sum (sq_i)
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.linear_proj(core_attn_out)
        return output, bias


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.bfloat16) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim
        self.theta = theta

    def forward(self, seqlen: int) -> torch.Tensor:
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.bfloat16) / self.dim)).to(
            self.inv_freq.device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=torch.bfloat16)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchEmbed(nn.Module):
    def __init__(
            self,
            patch_size: int = 14,
            temporal_patch_size: int = 2,
            in_channels: int = 3,
            embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen2VLViT(MultiModalModule):
    """
    Qwen2VLViT vision model.
    Instantiate a Qwen2VLViT model.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
    """

    def __init__(
            self,
            config: TransformerConfig,
            transformer_layer_spec: ModuleSpec,
            pre_process: bool = True,
            post_process: bool = True,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(config=config)

        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.pre_process = pre_process
        self.post_process = post_process

        if self.pre_process:
            self.patch_embed = PatchEmbed(
                patch_size=config.patch_size,
                temporal_patch_size=config.temporal_patch_size,
                in_channels=config.in_channels,
                embed_dim=config.hidden_size,
            )

        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = Qwen2VLVisionTransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            post_layer_norm=False,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """
        Sets pinput tensor to the model. only used when vit crop to multi pipeline, coming soon.

        Args:
            input_tensor (torch.Tensor):Sets the input tensor for the model.
        """
        self.blocks.set_input_tensor(input_tensor)

    def get_window_index(self, grid_thw):
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        vit_merger_window_size = self.config.window_attn_size // self.spatial_merge_size // self.config.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_size * self.spatial_merge_size + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward function of the Qwen2VL ViT Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        """
        if self.pre_process:
            if pixel_values is None or grid_thw is None:
                raise ValueError('You have to specify pixel_values and grid_thw')
            else:
                hidden_states = self.patch_embed(pixel_values)
                hidden_states = hidden_states.unsqueeze(1)
        else:
            hidden_states = None

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len = hidden_states.shape[0] if hidden_states is not None else pixel_values.shape[-2]
        window_index = None
        window_mask = None
        cu_window_seqlens = None
        if getattr(self.config, 'window_attn_size', None) is not None:
            if getattr(self.config, 'fullatt_block_indexes', None) is None:
                raise ValueError("The 'fullatt_block_indexes' attribute is required when using 'window_attn_size'.")
            window_index, cu_window_seqlens = self.get_window_index(grid_thw)
            cu_window_seqlens = torch.tensor(
                cu_window_seqlens,
                device=grid_thw.device,
                dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
            )
            cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

            spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
            if self.pre_process:
                hidden_states = hidden_states.squeeze(1)
                hidden_states = hidden_states.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
                hidden_states = hidden_states[window_index, :, :]
                hidden_states = hidden_states.reshape(seq_len, -1)
                hidden_states = hidden_states.unsqueeze(1)

            rotary_pos_emb = rotary_pos_emb.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
            rotary_pos_emb = rotary_pos_emb[window_index, :, :]
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

            window_mask = torch.full(
                [1, seq_len, seq_len], torch.finfo(pixel_values.dtype).min, device=pixel_values.device,
                dtype=torch.bool
            )
            for i in range(1, len(cu_window_seqlens)):
                window_mask[..., cu_window_seqlens[i - 1]: cu_window_seqlens[i], cu_window_seqlens[i - 1]: cu_window_seqlens[i]] = 0

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        attention_mask = torch.full(
            [1, seq_len, seq_len], torch.finfo(pixel_values.dtype).min, device=pixel_values.device,
            dtype=torch.bool
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]: cu_seqlens[i], cu_seqlens[i - 1]: cu_seqlens[i]] = 0
        if get_args().use_flash_attn:
            if set_actual_seq_len is None:
                raise AssertionError("Please check the commit id of your MindSpeed")
            set_actual_seq_len(tuple(cu_seqlens[1:].cpu().numpy().tolist()))
            
        if get_args().sequence_parallel:
            hidden_states = scatter_to_sequence_parallel_region(hidden_states)
            
        if mpu.get_context_parallel_world_size() > 1:
            split_gather_sizes = cal_split_sizes(hidden_states.shape[0], mpu.get_context_parallel_world_size())
            rotary_pos_emb = split_forward_gather_backward(
                rotary_pos_emb,
                mpu.get_context_parallel_group(),
                0,
                split_gather_sizes,
                "down"
            )
            hidden_states = split_forward_gather_backward(
                hidden_states, 
                mpu.get_context_parallel_group(), 
                0, 
                split_gather_sizes,
                "down"
            )
            
        hidden_states = self.blocks(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            window_mask=window_mask,
            cu_seqlens=cu_seqlens,
            cu_window_seqlens=cu_window_seqlens
        )
        
        if mpu.get_context_parallel_world_size() > 1:
            hidden_states = gather_forward_split_backward(
                hidden_states,
                mpu.get_context_parallel_group(),
                0,
                split_gather_sizes,
                "up"
            )
            
        if get_args().sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(hidden_states)
            
        if get_args().use_flash_attn:
            set_actual_seq_len(None)
        return hidden_states, window_index
