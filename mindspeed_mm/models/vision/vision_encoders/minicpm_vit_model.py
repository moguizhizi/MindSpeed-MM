#Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
#Copyright 2024 Google AI and The HuggingFace Team. All rights reserved.
#Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

from typing import Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn

from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor

from mindspeed_mm.models.common.module import MultiModalModule


class MiniCPMVisionEmbeddings(nn.Module):
    def __init__(
            self,
            config: TransformerConfig,
            image_size=448,
            patch_size=14
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_size = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_size**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def forward(self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor,
                tgt_sizes: Optional[torch.IntTensor] = None) -> torch.Tensor:
        batch_size = pixel_values.size(0)

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = torch.arange(1 / self.num_patches_per_size, 1.0, 1 / self.num_patches_per_size)
        position_ids = torch.full(
            size=(
                batch_size,
                max_nb_patches_h * max_nb_patches_w,
            ),
            fill_value=0,
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            if tgt_sizes is not None:
                nb_patches_h = tgt_sizes[batch_idx][0]
                nb_patches_w = tgt_sizes[batch_idx][1]
            else:
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_size + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        position_ids = position_ids.to(self.position_embedding.weight.device)

        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class MiniCPMSelfAttention(SelfAttention):
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

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
        )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.config.hidden_size,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.config.hidden_size,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        N, B, C = mixed_qkv.shape
        mixed_qkv = mixed_qkv.reshape(N, B, 3, self.num_attention_heads_per_partition,
                                      self.hidden_size_per_attention_head)
        mixed_qkv = mixed_qkv.permute(2, 0, 3, 1, 4)

        query, key, value = mixed_qkv.unbind(0)  # [sq, np, b, hn]

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        if self.q_layernorm is not None:
            N_, H_, B_, D_ = query.shape
            query = self.q_layernorm(query.permute(2, 0, 1, 3).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(0, 1)

        if self.k_layernorm is not None:
            N_, H_, B_, D_ = key.shape
            key = self.k_layernorm(key.permute(2, 0, 1, 3).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(0, 1)

        value = value.permute(0, 2, 1, 3)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value


class MiniCPMViT(MultiModalModule):
    def __init__(
            self,
            config: TransformerConfig,
            transformer_layers_spec: ModuleSpec,
            pre_process: bool = True,
            post_process: bool = False,
            *args,
            **kwargs,
    ):
        super().__init__(config=config)
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.pre_process = pre_process
        self.post_process = post_process

        self.seq_length = 1 + (self.image_size // self.patch_size) ** 2
        if self.pre_process:
            self.embeddings = MiniCPMVisionEmbeddings(config=config, image_size=self.image_size, patch_size=self.patch_size)
        self.encoder = TransformerBlock(
            config=config,
            spec=transformer_layers_spec,
            post_layer_norm=True,
            pre_process=self.pre_process,
            post_process=self.post_process
        )

    def forward(
            self,
            pixel_values,
            patch_attention_mask: Optional[torch.BoolTensor] = None,
            tgt_sizes: Optional[torch.IntTensor] = None,
            *args,
            **kwargs
    ):
        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_attention_mask = torch.ones(
                size=(
                    batch_size,
                    pixel_values.size(2) // self.patch_size,
                    pixel_values.size(3) // self.patch_size,
                ),
                dtype=torch.bool
            )

        hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask, tgt_sizes=tgt_sizes)

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)

        if not torch.any(~patch_attention_mask):
            attention_mask = None
        else:
            attention_mask = patch_attention_mask

        hidden_states = hidden_states.transpose(0, 1)
        encoder_outputs = self.encoder(hidden_states=hidden_states, attention_mask=attention_mask)

        return encoder_outputs