import random
from selectors import EpollSelector
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import dataclass

from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
from diffusers.utils import is_torch_version
from megatron.legacy.model.rms_norm import RMSNorm
from megatron.core import mpu, tensor_parallel
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from mindspeed_mm.models.common import MultiModalModule
from mindspeed_mm.models.common.embeddings import PatchEmbed2D, CombinedTimestepTextProjEmbeddings
from mindspeed_mm.models.common.ffn import FeedForward
from mindspeed_mm.models.common.attention import MultiHeadSparseMMAttentionSBH
from mindspeed_mm.models.common.communications import split_forward_gather_backward, gather_forward_split_backward


def create_custom_forward(module, return_dict=None):
    def custom_forward(*inputs):
        if return_dict is not None:
            return module(*inputs, return_dict=return_dict)
        else:
            return module(*inputs)
    return custom_forward


@dataclass
class BlockForwardInputs:
    embedded_timestep: torch.Tensor
    frames: int
    height: int
    width: int
    video_rotary_emb: Tuple[torch.Tensor]


@dataclass
class PositionParams:
    b: int
    t: int
    h: int
    w: int
    device: str
    training: bool = True


class SparseUMMDiT(MultiModalModule):
    """
    A video dit model for video generation. can process both standard continuous images of shape
    (batch_size, num_channels, width, height) as well as quantized image embeddings of shape
    (batch_size, num_image_vectors). Define whether input is continuous or discrete depending on config.

    Args:
        num_layers: The number of layers for VideoDiTBlock.
        num_heads: The number of heads to use for multi-head attention.
        head_dim: The number of channels in each head.
        in_channels: The number of channels in· the input (specify if the input is continuous).
        out_channels: The number of channels in the output.
        dropout: The dropout probability to use.
        cross_attention_dim: The number of prompt dimensions to use.
        attention_bias: Whether to use bias in VideoDiTBlock's attention.
        patch_size_thw: The shape of the patchs.
        activation_fn: The name of activation function use in VideoDiTBlock.
        norm_elementwise_affine: Whether to use learnable elementwise affine parameters for normalization.
        norm_eps: The eps of the normalization.
        interpolation_scale: The scale for interpolation.
    """

    def __init__(
        self,
        num_heads: int = 16,
        head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: Optional[List[int]] = None,
        sparse_n: Optional[List[int]] = None,
        double_ff: bool = False,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_size_h: Optional[int] = None,
        sample_size_w: Optional[int] = None,
        sample_size_t: Optional[int] = None,
        patch_size_thw: Optional[Tuple[int]] = None,
        activation_fn: str = "gelu-approximate",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-6,
        caption_channels: int = None,
        interpolation_scale_h: float = 1.0,
        interpolation_scale_w: float = 1.0,
        interpolation_scale_t: float = 1.0,
        sparse1d: bool = False,
        pooled_projection_dim: int = 1024,
        timestep_embed_dim: int = 512,
        norm_cls: str = 'rms_norm',
        skip_connection: bool = False,
        explicit_uniform_rope: bool = False,
        **kwargs
    ):
        super().__init__(config=None)
        args = get_args()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sequence_parallel = args.sequence_parallel
        self.gradient_checkpointing = True
        self.recompute_granularity = args.recompute_granularity
        self.distribute_saved_activations = args.distribute_saved_activations
        self.recompute_num_layers = args.recompute_num_layers
        if self.recompute_granularity == "selective":
            raise ValueError("recompute_granularity does not support selective mode in VideoDiT")
        if self.distribute_saved_activations:
            raise NotImplementedError("distribute_saved_activations is currently not supported")

        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        hidden_size = num_heads * head_dim
        self.num_layers = num_layers if num_layers is not None else [2, 4, 8, 4, 2]
        self.sparse_n = sparse_n if sparse_n is not None else [1, 4, 16, 4, 1]
        self.patch_size_t = patch_size_thw[0]
        self.patch_size = patch_size_thw[1]
        self.skip_connection = skip_connection

        if norm_cls == 'rms_norm':
            self.norm_cls = RMSNorm
        elif norm_cls == 'layer_norm':
            self.norm_cls = nn.LayerNorm


        if len(num_layers) != len(sparse_n):
            raise ValueError("num_layers and sparse_n must have the same length")
        if len(num_layers) % 2 != 1:
            raise ValueError("num_layers must have odd length")
        if any([i % 2 != 0 for i in num_layers]):
            raise ValueError("num_layers must have even numbers")

        if not sparse1d:
            self.sparse_n = [1] * len(num_layers)

        interpolation_scale_thw = (interpolation_scale_t, interpolation_scale_h, interpolation_scale_w)

        # 1. patch embedding
        self.patch_embed = PatchEmbed2D(
            patch_size=self.patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )

        # 2. time embedding and pooled text embedding
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            timestep_embed_dim=timestep_embed_dim, 
            embedding_dim=timestep_embed_dim, 
            pooled_projection_dim=pooled_projection_dim
        )

        # 3. anthor text embedding
        self.caption_projection = nn.Linear(caption_channels, hidden_size)

        # 4. rope
        self.rope = RoPE3D(interpolation_scale=interpolation_scale_thw)
        self.position_getter = PositionGetter3D(
            sample_size_t, sample_size_h, sample_size_w, explicit_uniform_rope, atten_layout="SBH"
        )

        # forward transformer blocks
        self.transformer_blocks = []
        self.skip_norm_linear = []
        self.skip_norm_linear_enc = []

        for idx, (num_layer, sparse_n) in enumerate(zip(self.num_layers, self.sparse_n)):
            is_last_stage = idx == len(num_layers) - 1
            if self.skip_connection and idx > len(num_layers) // 2:
                self.skip_norm_linear.append(
                    nn.Sequential(
                        self.norm_cls(
                            hidden_size * 2,
                            eps=norm_eps,
                            sequence_parallel=self.sequence_parallel,
                        ), 
                        nn.Linear(hidden_size * 2, hidden_size), 
                    )
                )
                self.skip_norm_linear_enc.append(
                    nn.Sequential(
                        self.norm_cls(
                            hidden_size * 2,
                            eps=norm_eps,
                            sequence_parallel=self.sequence_parallel,
                        ), 
                        nn.Linear(hidden_size * 2, hidden_size), 
                    )
                )
            stage_blocks = nn.ModuleList(
                [
                    SparseMMDiTBlock(
                        dim=hidden_size,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        timestep_embed_dim=timestep_embed_dim,
                        dropout=dropout,
                        activation_fn=activation_fn,
                        attention_bias=attention_bias,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        interpolation_scale_thw=interpolation_scale_thw,
                        double_ff=double_ff,
                        sparse1d=sparse1d if sparse_n > 1 else False,
                        sparse_n=sparse_n,
                        sparse_group=i % 2 == 1 if sparse_n > 1 else False,
                        context_pre_only=is_last_stage and (i == num_layer - 1),
                        norm_cls=norm_cls,
                    )
                    for i in range(num_layer)
                ]
            )
            self.transformer_blocks.append(stage_blocks)
        self.transformer_blocks = nn.ModuleList(self.transformer_blocks)

        if self.skip_connection:
            self.skip_norm_linear = nn.ModuleList(self.skip_norm_linear)
            self.skip_norm_linear_enc = nn.ModuleList(self.skip_norm_linear_enc)

        self.norm_final = self.norm_cls(
            hidden_size, eps=norm_eps, sequence_parallel=self.sequence_parallel,
        )

        self.norm_out = AdaNorm(
            embedding_dim=timestep_embed_dim,
            output_dim=hidden_size * 2,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            norm_cls=norm_cls,
        )

        self.proj_out = nn.Linear(
            hidden_size, self.patch_size_t * self.patch_size * self.patch_size * out_channels
        )

        # set label "sequence_parallel", for all_reduce the grad
        modules = [self.norm_final]
        if self.skip_connection:
            modules += [self.skip_norm_linear, self.skip_norm_linear_enc]
        for module in modules:
            for _, param in module.named_parameters():
                setattr(param, "sequence_parallel", self.sequence_parallel)

    def prepare_sparse_mask(self, attention_mask, encoder_attention_mask, sparse_n):
        attention_mask = attention_mask.unsqueeze(1)
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        length = attention_mask.shape[-1]
        if length % (sparse_n * sparse_n) == 0:
            pad_len = 0
        else:
            pad_len = sparse_n * sparse_n - length % (sparse_n * sparse_n)

        attention_mask_sparse = F.pad(attention_mask, (0, pad_len, 0, 0), value=-10000.0)
        attention_mask_sparse_1d = rearrange(
            attention_mask_sparse,
            'b 1 1 (g k) -> (k b) 1 1 g',
            k=sparse_n
        )
        attention_mask_sparse_1d_group = rearrange(
            attention_mask_sparse,
            'b 1 1 (n m k) -> (m b) 1 1 (n k)',
            m=sparse_n,
            k=sparse_n
        )
        encoder_attention_mask_sparse = encoder_attention_mask.repeat(sparse_n, 1, 1, 1)

        # concat mask at sequence dim
        attention_mask_sparse_1d = torch.cat([attention_mask_sparse_1d, encoder_attention_mask_sparse], dim=-1)
        attention_mask_sparse_1d_group = torch.cat([attention_mask_sparse_1d_group, encoder_attention_mask_sparse], dim=-1)

        def get_attention_mask(mask, repeat_num):
            mask = mask.to(torch.bool)
            mask = mask.repeat(1, 1, repeat_num, 1)
            return mask

        attention_mask_sparse_1d = get_attention_mask(
            attention_mask_sparse_1d, attention_mask_sparse_1d.shape[-1]
        )
        attention_mask_sparse_1d_group = get_attention_mask(
            attention_mask_sparse_1d_group, attention_mask_sparse_1d_group.shape[-1]
        )

        return {
            False: attention_mask_sparse_1d,
            True: attention_mask_sparse_1d_group
        }


    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the module (assuming that all the module parameters have the same dtype)."""
        params = tuple(self.parameters())
        if len(params) > 0:
            return params[0].dtype
        else:
            buffers = tuple(self.buffers())
            return buffers[0].dtype


    def _operate_on_enc(
        self, hidden_states, encoder_hidden_states, inputs: BlockForwardInputs
    ):

        skip_connections = []
        for _, stage_block in enumerate(self.transformer_blocks[:len(self.num_layers) // 2]):
            for _, block in enumerate(stage_block):
                attention_mask = self.sparse_mask[block.sparse_n][block.sparse_group]
                hidden_states, encoder_hidden_states = self.block_forward(
                    block=block,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    inputs=inputs
                )
                if self.skip_connection:
                    skip_connections.append([hidden_states, encoder_hidden_states])
        return hidden_states, encoder_hidden_states, skip_connections

    def _operate_on_mid(
        self, hidden_states, encoder_hidden_states, inputs: BlockForwardInputs
    ):
        for _, block in enumerate(self.transformer_blocks[len(self.num_layers) // 2]):
            attention_mask = self.sparse_mask[block.sparse_n][block.sparse_group]
            hidden_states, encoder_hidden_states = self.block_forward(
                block=block,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                inputs=inputs
            )
        return hidden_states, encoder_hidden_states

    def _operate_on_dec(
        self, hidden_states, skip_connections, encoder_hidden_states, inputs: BlockForwardInputs
    ):
        for idx, stage_block in enumerate(self.transformer_blocks[len(self.num_layers) // 2 + 1:]):
            if self.skip_connection:
                skip_hidden_states, skip_encoder_hidden_states = skip_connections.pop()
                hidden_states = torch.cat([hidden_states, skip_hidden_states], dim=-1)
                hidden_states = self.skip_norm_linear[idx](hidden_states)
                encoder_hidden_states = torch.cat([encoder_hidden_states, skip_encoder_hidden_states], dim=-1)
                encoder_hidden_states = self.skip_norm_linear_enc[idx](encoder_hidden_states)

            for _, block in enumerate(stage_block):
                attention_mask = self.sparse_mask[block.sparse_n][block.sparse_group]
                hidden_states, encoder_hidden_states = self.block_forward(
                    block=block,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    inputs=inputs
                )

        return hidden_states, encoder_hidden_states

    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, pooled_projections):

        hidden_states = self.patch_embed(hidden_states.to(self.dtype))
        if pooled_projections.shape[1] != 1:
            raise AssertionError("Pooled projection should have shape (b, 1, 1, d)")
        pooled_projections = pooled_projections.squeeze(1)  # b 1 1 d -> b 1 d
        timesteps_emb = self.time_text_embed(timestep, pooled_projections)  # (N, D)

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # b, 1, l, d
        if encoder_hidden_states.shape[1] != 1:
            raise AssertionError("Encoder hidden states should have shape (b, 1, l, d)")
        encoder_hidden_states = encoder_hidden_states.squeeze(1)

        return hidden_states, encoder_hidden_states, timesteps_emb

    def _get_output_for_patched_inputs(
        self, hidden_states, embedded_timestep, frames, height, width
    ):
        hidden_states = self.norm_final(hidden_states)

        hidden_states = self.norm_out(hidden_states, temb=embedded_timestep)

        if self.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(hidden_states,
                                                                           tensor_parallel_output_grad=False)

        # Change To (b, t*h*w, h) or (b, t//sp*h*w, h)
        hidden_states = rearrange(hidden_states, 's b h -> b s h', b=hidden_states.shape[1]).contiguous()

        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            -1, frames, height, width,
            self.patch_size_t, self.patch_size, self.patch_size, self.out_channels
        )

        hidden_states = torch.einsum("nthwopqc -> nctohpwq", hidden_states)
        output = hidden_states.reshape(
            -1,
            self.out_channels,
            frames * self.patch_size_t,
            height * self.patch_size,
            width * self.patch_size
        )
        return output

    def block_forward(self, block, hidden_states, attention_mask, encoder_hidden_states, inputs: BlockForwardInputs):
        if self.training and self.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                inputs,
                **ckpt_kwargs
            )
        else:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                inputs=inputs
            )
        return hidden_states, encoder_hidden_states


    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        prompt: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        pooled_projections = prompt[1] if len(prompt) > 1 else None
        encoder_hidden_states = prompt[0]
        encoder_attention_mask = prompt_mask[0]
        attention_mask = video_mask

        batch_size, c, frames, height, width = hidden_states.shape

        encoder_attention_mask = encoder_attention_mask.view(batch_size, -1, encoder_attention_mask.shape[-1])
        if self.training and mpu.get_context_parallel_world_size() > 1:
            frames //= mpu.get_context_parallel_world_size()
            hidden_states = split_forward_gather_backward(hidden_states, mpu.get_context_parallel_group(), dim=2,
                                                    grad_scale='down')
            encoder_hidden_states = split_forward_gather_backward(encoder_hidden_states, mpu.get_context_parallel_group(),
                                                   dim=2, grad_scale='down')

        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = attention_mask.to(self.dtype)

            attention_mask = attention_mask.unsqueeze(1)  # b 1 t h w
            attention_mask = F.max_pool3d(
                attention_mask,
                kernel_size=(self.patch_size_t, self.patch_size, self.patch_size),
                stride=(self.patch_size_t, self.patch_size, self.patch_size)
            )
            attention_mask = rearrange(attention_mask, 'b 1 t h w -> (b 1) 1 (t h w)')
            attention_mask = (1 - attention_mask.bool().to(self.dtype)) * -10000.0

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:
            # b, 1, l
            encoder_attention_mask = (1 - encoder_attention_mask.to(self.dtype)) * -10000.0

        # 1. Input
        frames = ((frames - 1) // self.patch_size_t + 1) if frames % 2 == 1 else frames // self.patch_size_t  # patchfy
        height, width = height // self.patch_size, width // self.patch_size

        hidden_states, encoder_hidden_states, embedded_timestep = self._operate_on_patched_inputs(
            hidden_states, encoder_hidden_states, timestep, pooled_projections
        )

        hidden_states = rearrange(hidden_states, 'b s h -> s b h', b=batch_size).contiguous()
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b s h -> s b h', b=batch_size).contiguous()

        self.sparse_mask = {}

        if attention_mask.device != encoder_attention_mask.device:
            encoder_attention_mask = encoder_attention_mask.to(attention_mask.device)
        for sparse_n in list(set(self.sparse_n)):
            self.sparse_mask[sparse_n] = self.prepare_sparse_mask(attention_mask, encoder_attention_mask, sparse_n)

        pos_thw = self.position_getter(
            PositionParams(
                b=batch_size,
                t=frames * mpu.get_context_parallel_world_size(),
                h=height,
                w=width,
                device=hidden_states.device,
                training=self.training
            )
        )

        video_rotary_emb = self.rope(self.head_dim, pos_thw, hidden_states.device, hidden_states.dtype)

        if self.sequence_parallel:
            hidden_states = tensor_parallel.scatter_to_sequence_parallel_region(hidden_states)
            encoder_hidden_states = tensor_parallel.scatter_to_sequence_parallel_region(encoder_hidden_states)

        inputs = BlockForwardInputs(
            embedded_timestep=embedded_timestep,
            frames=frames,
            height=height,
            width=width,
            video_rotary_emb=video_rotary_emb
        )

        hidden_states, encoder_hidden_states, skip_connections = self._operate_on_enc(
            hidden_states, encoder_hidden_states, inputs
        )

        hidden_states, encoder_hidden_states = self._operate_on_mid(
            hidden_states, encoder_hidden_states, inputs
        )

        hidden_states, encoder_hidden_states = self._operate_on_dec(
            hidden_states, skip_connections, encoder_hidden_states, inputs
        )

        # 3. Output
        output = self._get_output_for_patched_inputs(
            hidden_states, embedded_timestep, frames, height, width
        )  # b c t h w

        if self.training and mpu.get_context_parallel_world_size() > 1:
            output = gather_forward_split_backward(output, mpu.get_context_parallel_group(), dim=2,
                                                        grad_scale='up')

        return output


class SparseMMDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        timestep_embed_dim: int, 
        dropout=0.0,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        attention_out_bias: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        context_pre_only: bool = False,
        interpolation_scale_thw: Tuple[int] = (1, 1, 1),
        double_ff: bool = False,
        sparse1d: bool = False,
        sparse_n: int = 2,
        sparse_group: bool = False,
        norm_cls: str = 'rms_norm',
    ):
        super().__init__()

        self.sparse1d = sparse1d
        self.sparse_n = sparse_n
        self.sparse_group = sparse_group
        self.head_dim = head_dim

        if norm_cls == 'rms_norm':
            self.norm_cls = RMSNorm
        elif norm_cls == 'layer_norm':
            self.norm_cls = nn.LayerNorm


        # adanorm-zero1: to introduce timestep and clip condition
        self.norm1 = OpenSoraNormZero(
            timestep_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True, norm_cls=norm_cls
        )

        # 1. MM Attention
        self.attn1 = MultiHeadSparseMMAttentionSBH(
            query_dim=dim,
            key_dim=None,
            num_heads=num_heads,
            head_dim=head_dim,
            added_kv_proj_dim=dim,
            dropout=dropout,
            proj_qkv_bias=attention_bias,
            proj_out_bias=attention_out_bias,
            context_pre_only=context_pre_only,
            qk_norm=norm_cls,
            eps=norm_eps,
            sparse1d=sparse1d,
            sparse_n=sparse_n,
            sparse_group=sparse_group,
            is_cross_attn=False,
        )

        # adanorm-zero2: to introduce timestep and clip condition
        self.norm2 = OpenSoraNormZero(
            timestep_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True, norm_cls=norm_cls
        )

        # 2. Feed-forward
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        self.double_ff = double_ff
        if self.double_ff:
            self.ff_enc = FeedForward(
                dim,
                dropout=dropout,
                activation_fn=activation_fn,
                final_dropout=final_dropout,
                inner_dim=ff_inner_dim,
                bias=ff_bias,
            )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        inputs: Optional[BlockForwardInputs] = None,
    ) -> torch.FloatTensor:

        # 0. Prepare rope embedding
        vis_seq_length, batch_size = hidden_states.shape[:2]

        # 1. norm & scale & shift
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, inputs.embedded_timestep
        )

        # 2. MM Attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            frames=inputs.frames,
            height=inputs.height,
            width=inputs.width,
            attention_mask=attention_mask,
            video_rotary_emb=inputs.video_rotary_emb,
        )

        # 3. residual & gate
        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # 4. norm & scale & shift
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, inputs.embedded_timestep
        )
        if self.double_ff:
            # 5. FFN
            vis_ff_output = self.ff(norm_hidden_states)
            enc_ff_output = self.ff_enc(norm_encoder_hidden_states)
            # 6. residual & gate
            hidden_states = hidden_states + gate_ff * vis_ff_output
            encoder_hidden_states = encoder_hidden_states + enc_gate_ff * enc_ff_output
        else:
            # 5. FFN
            norm_hidden_states = torch.cat([norm_hidden_states, norm_encoder_hidden_states], dim=0)
            ff_output = self.ff(norm_hidden_states)
            # 6. residual & gate
            hidden_states = hidden_states + gate_ff * ff_output[:vis_seq_length]
            encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[vis_seq_length:]

        return hidden_states, encoder_hidden_states



class AdaNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`, *optional*): The size of the embeddings dictionary.
        output_dim (`int`, *optional*):
        norm_elementwise_affine (`bool`, defaults to `False):
        norm_eps (`bool`, defaults to `False`):
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        output_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        norm_cls: str = 'rms_norm',
    ):
        super().__init__()
        output_dim = output_dim or embedding_dim * 2

        if num_embeddings is not None:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        if norm_cls == 'rms_norm':
            self.norm_cls = RMSNorm
        elif norm_cls == 'layer_norm':
            self.norm_cls = nn.LayerNorm
        self.norm = self.norm_cls(
            output_dim // 2, eps=norm_eps
        )

    def forward(
        self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.emb is not None:
            temb = self.emb(timestep)

        temb = self.linear(self.silu(temb))
        # x shape: (S B H), temb shape: (B, H)
        shift, scale = temb.chunk(2, dim=1)
        shift = shift[None, :, :]
        scale = scale[None, :, :]

        x = self.norm(x) * (1 + scale) + shift
        return x


class OpenSoraNormZero(nn.Module):
    def __init__(
        self,
        timestep_embed_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_cls: str = 'rms_norm',
    ) -> None:
        super().__init__()
        args = get_args()
        self.sequence_parallel = args.sequence_parallel
        config = core_transformer_config_from_args(args)

        if norm_cls == 'rms_norm':
            self.norm_cls = RMSNorm
        elif norm_cls == 'layer_norm':
            self.norm_cls = nn.LayerNorm

        self.silu = nn.SiLU()
        self.linear = tensor_parallel.ColumnParallelLinear(
            timestep_embed_dim,
            6 * embedding_dim,
            config=config,
            init_method=config.init_method,
            gather_output=True
        )
        self.norm = self.norm_cls(embedding_dim, eps=eps, sequence_parallel=self.sequence_parallel)
        self.norm_enc = self.norm_cls(embedding_dim, eps=eps, sequence_parallel=self.sequence_parallel)

        # set label "sequence_parallel", for all_reduce the grad
        for module in [self.norm, self.norm_enc]:
            for param in module.parameters():
                setattr(param, "sequence_parallel", self.sequence_parallel)


    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb))[0].chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[None, :, :] + shift[None, :, :] # because hidden_states'shape is (S B H), so we need to add None at the first dimension
        encoder_hidden_states = self.norm_enc(encoder_hidden_states) * (1 + enc_scale)[None, :, :] + enc_shift[None, :, :]
        return hidden_states, encoder_hidden_states, gate[None, :, :], enc_gate[None, :, :]


class RoPE3D(nn.Module):
    
    def __init__(self, freq=10000.0, interpolation_scale=(1, 1, 1)):
        super().__init__()
        self.base = freq
        self.interpolation_scale_t, self.interpolation_scale_h, self.interpolation_scale_w = interpolation_scale
        self.cache = {}

    def get_cos_sin(self, params):
        D, seq_start, seq_end, device, dtype, interpolation_scale = params
        key = (D, seq_start, seq_end, device, dtype)
        if key not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_start, seq_end, device=device, dtype=inv_freq.dtype) / interpolation_scale
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[key] = (cos, sin)
        return self.cache[key]

    def forward(self, dim, positions, device, dtype):
        """
        input:
            * dim: head_dim
            * positions: batch_size x ntokens x 3 (t, y and x position of each token)
        output:
            * tokens after appplying RoPE3D (ntokens x batch_size x nheads x dim)
        """
        if dim % 16 != 0:
            raise ValueError("number of dimensions should be a multiple of 16")
        D_t = dim // 16 * 4
        D = dim // 16 * 6
        poses, min_poses, max_poses = positions
        if len(poses) != 3 or poses[0].ndim != 2:  # [Batch, Seq, 3]
            raise AssertionError("poses shape error")

        params_t = (D_t, min_poses[0], max_poses[0], device, dtype, self.interpolation_scale_t)
        params_y = (D, min_poses[1], max_poses[1], device, dtype, self.interpolation_scale_h)
        params_x = (D, min_poses[2], max_poses[2], device, dtype, self.interpolation_scale_w)

        cos_t, sin_t = self.get_cos_sin(params_t)
        cos_y, sin_y = self.get_cos_sin(params_y)
        cos_x, sin_x = self.get_cos_sin(params_x)

        cos_t, sin_t = compute_rope1d(poses[0], cos_t, sin_t)
        cos_y, sin_y = compute_rope1d(poses[1], cos_y, sin_y)
        cos_x, sin_x = compute_rope1d(poses[2], cos_x, sin_x)
        video_rotary_emb = (cos_t, sin_t, cos_y, sin_y, cos_x, sin_x)
        return video_rotary_emb


def compute_rope1d(pos1d, cos, sin):
    """
        * pos1d: ntokens x batch_size
    """
    if pos1d.ndim != 2:
        raise AssertionError("pos1d.ndim must be 2")
    # for (ntokens x batch_size x nheads x dim)
    cos = F.embedding(pos1d, cos)[:, :, None, :]
    sin = F.embedding(pos1d, sin)[:, :, None, :]

    return cos, sin


class PositionGetter3D:
    """return positions of patches"""

    def __init__(self, max_t, max_h, max_w, explicit_uniform_rope=False, atten_layout="BSH"):
        self.cache_positions = {}
        self.atten_layout = atten_layout
        self.max_t = max_t
        self.max_h = max_h
        self.max_w = max_w
        self.explicit_uniform_rope = explicit_uniform_rope

    @staticmethod
    def check_type(param):
        if isinstance(param, torch.Tensor):
            param = param.item()
        return param

    def get_positions(self, b, t, h, w, device):
        x = torch.arange(w, device=device)
        y = torch.arange(h, device=device)
        z = torch.arange(t, device=device)
        pos = torch.cartesian_prod(z, y, x)
        if self.atten_layout == "SBH":
            pos = pos.reshape(t * h * w, 3).transpose(0, 1).reshape(3, -1, 1).contiguous().expand(3, -1, b).clone()
        elif self.atten_layout == "BSH":
            pos = pos.reshape(t * h * w, 3).transpose(0, 1).reshape(3, 1, -1).contiguous().expand(3, b, -1).clone()
        else:
            raise ValueError(f"Unsupported layout type: {self.atten_layout}")
        return (pos[0].contiguous(), pos[1].contiguous(), pos[2].contiguous())

    def __call__(self, params: PositionParams):
        b = self.check_type(params.b)
        t = self.check_type(params.t)
        h = self.check_type(params.h)
        w = self.check_type(params.w)
        device = params.device
        training = params.training

        # random.randint is [a, b], but torch.randint is [a, b)
        s_t = random.randint(0, self.max_t - t) if self.explicit_uniform_rope and training else 0
        e_t = s_t + t
        s_h = random.randint(0, self.max_h - h) if self.explicit_uniform_rope and training else 0
        e_h = s_h + h
        s_w = random.randint(0, self.max_w - w) if self.explicit_uniform_rope and training else 0
        e_w = s_w + w

        key = (b, s_t, e_t, s_h, e_h, s_w, e_w)
        if key not in self.cache_positions:
            poses = self.get_positions(b, t, h, w, device)
            max_poses = (e_t, e_h, e_w)
            min_poses = (s_t, s_h, s_w)

            self.cache_positions[key] = (poses, min_poses, max_poses)
        pos = self.cache_positions[key]
        return pos
