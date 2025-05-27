#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : qwen2_5_vl_mm_to_hf.py
@Time    : 2025/03/24
@Desc    : qwen2.5vl mindspeed-mm模型转换成huggingface模型
"""
from pathlib import Path
from typing import cast

import torch
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
# 注意mindspeed-mm训练后保存的checkpoint中存储了patch相关信息，在load时需要加下面这行以支持反序列化
import mindspeed.megatron_adaptor  # noqa

from checkpoint.utils import LATEST_TXT, ConvertHFConfig, copy_files_except_suffix, save_by_index_json, \
    split_by_index_json, load_from_mm


def qkv_regroup(value, num_heads, q_size, k_size, v_size):
    qkv_chunks = torch.chunk(value, num_heads, dim=0)
    q_chunks = []
    k_chunks = []
    v_chunks = []
    for chunk in qkv_chunks:
        q_chunk, k_chunk, v_chunk = torch.split(chunk, [q_size, k_size, v_size], dim=0)
        q_chunks.append(q_chunk)
        k_chunks.append(k_chunk)
        v_chunks.append(v_chunk)
    q_res = torch.cat(q_chunks, dim=0)
    k_res = torch.cat(k_chunks, dim=0)
    v_res = torch.cat(v_chunks, dim=0)
    return q_res, k_res, v_res


def merge_by_tp(_state_dicts: list[dict[str, torch.Tensor]], _tp_size: int) -> dict:
    if len(_state_dicts) == 0:
        raise AssertionError(f'_state_dicts is empty.')
    if len(_state_dicts) == 1:
        return _state_dicts[0]
    return_state_dict = {}
    for key, value in _state_dicts[0].items():
        if 'projector' not in key and 'linear_fc1' in key:
            chunks_0 = [torch.chunk(_state_dicts[i][key], 2, dim=0) for i in range(_tp_size)]
            flattened_tensors = [
                pair[i]
                for i in range(2)
                for pair in chunks_0
            ]
            return_state_dict[key] = torch.cat(flattened_tensors, dim=0)
        elif 'linear_qkv' in key or 'linear_fc1' in key or 'output_layer' in key or 'word_embeddings' in key:
            return_state_dict[key] = torch.cat([_state_dicts[i][key] for i in range(_tp_size)], dim=0)
        elif 'linear_proj.weight' in key or 'linear_fc2.weight' in key:
            return_state_dict[key] = torch.cat([_state_dicts[i][key] for i in range(_tp_size)], dim=1)
        else:
            return_state_dict[key] = value
    return return_state_dict


def convert_mm_to_hf(_state_dict: dict[str, torch.Tensor], cfg: Qwen2_5_VLConfig) -> dict:
    vit_head_hidden_size = cfg.vision_config.hidden_size // cfg.vision_config.num_heads
    llm_head_hidden_size = cfg.hidden_size // cfg.num_attention_heads
    q_size = llm_head_hidden_size * cfg.num_attention_heads // cfg.num_key_value_heads
    k_size = llm_head_hidden_size * cfg.num_key_value_heads // cfg.num_key_value_heads
    v_size = llm_head_hidden_size * cfg.num_key_value_heads // cfg.num_key_value_heads

    new_params = {}
    for key, value in _state_dict.items():
        if value is None:
            continue
        new_key = None
        # image_encoder 权重转换部分
        if key.startswith('image_encoder'):
            new_key = key.replace('image_encoder.projector.encoder.linear_fc1', 'visual.merger.mlp.0')
            new_key = new_key.replace('image_encoder.projector.encoder.linear_fc2', 'visual.merger.mlp.2')
            new_key = new_key.replace('image_encoder.projector.layernorm', 'visual.merger.ln_q')
            new_key = new_key.replace('image_encoder.encoder.patch_embed.proj', 'visual.patch_embed.proj')

            new_key = new_key.replace('image_encoder.encoder.blocks.layers', 'visual.blocks')
            new_key = new_key.replace('self_attention.linear_proj', 'attn.proj')
            new_key = new_key.replace('self_attention.linear_qkv', 'attn.qkv')
            new_key = new_key.replace('mlp.linear_fc1', 'mlp.fc1_proj')
            new_key = new_key.replace('mlp.linear_fc2', 'mlp.down_proj')
            new_key = new_key.replace('input_layernorm', 'norm1')
            new_key = new_key.replace('pre_mlp_layernorm', 'norm2')
            if 'attn.qkv' in new_key:
                q_res, k_res, v_res = qkv_regroup(value, cfg.vision_config.num_heads, vit_head_hidden_size,
                                                  vit_head_hidden_size, vit_head_hidden_size)
                new_params[new_key] = torch.cat((q_res, k_res, v_res), dim=0)

            elif 'mlp.fc1_proj' in new_key:
                gate_up_chunks = torch.chunk(value, 2, dim=0)
                layer = key.split('.')[4]
                name = key.split('.')[-1]  # weight或bias
                vit_layer_gate = f'visual.blocks.{layer}.mlp.gate_proj.'
                vit_layer_up = f'visual.blocks.{layer}.mlp.up_proj.'
                new_params[vit_layer_gate + name] = gate_up_chunks[0]
                new_params[vit_layer_up + name] = gate_up_chunks[1]

            else:
                if value is not None:
                    new_params[new_key] = value

        else:
            # self_attention.linear_qkv.weight 和 self_attention.linear_qkv.bias
            if 'self_attention.linear_qkv' in key:
                layer = key.split('.')[3]
                name = key.split('.')[-1]  # weight或bias
                q = f'model.layers.{layer}.self_attn.q_proj.{name}'
                k = f'model.layers.{layer}.self_attn.k_proj.{name}'
                v = f'model.layers.{layer}.self_attn.v_proj.{name}'
                new_params[q], new_params[k], new_params[v] = qkv_regroup(value, cfg.num_key_value_heads, q_size,
                                                                          k_size, v_size)

            elif 'mlp.linear_fc1.weight' in key:
                gate_up_chunks = torch.chunk(value, 2, dim=0)
                layer = key.split('.')[3]
                attention_gate = f'model.layers.{layer}.mlp.gate_proj.weight'
                attention_up = f'model.layers.{layer}.mlp.up_proj.weight'

                new_params[attention_gate] = gate_up_chunks[0]
                new_params[attention_up] = gate_up_chunks[1]

            elif key.startswith('text_decoder.output_layer'):
                new_key = key.replace('text_decoder.output_layer', 'lm_head')
                if value is not None:
                    new_params[new_key] = value
            elif key == 'text_decoder.embedding.word_embeddings.weight':
                new_key = key.replace('text_decoder.embedding.word_embeddings.weight', 'model.embed_tokens.weight')
                if value is not None:
                    new_params[new_key] = value
            elif key == 'text_decoder.decoder.final_layernorm.weight':
                new_key = key.replace('text_decoder.decoder.final_layernorm.weight', 'model.norm.weight')
                if value is not None:
                    new_params[new_key] = value
            elif key.startswith('text_decoder.decoder.layers'):
                new_key = key.replace('text_decoder.decoder.layers', 'model.layers')
                new_key = new_key.replace('self_attention.linear_proj.weight', 'self_attn.o_proj.weight')
                new_key = new_key.replace('pre_mlp_layernorm.weight', 'post_attention_layernorm.weight')
                new_key = new_key.replace('mlp.linear_fc2.weight', 'mlp.down_proj.weight')

                new_params[new_key] = value

    if cfg.tie_word_embeddings and "lm_head.weight" in new_params:
        del new_params["lm_head.weight"]

    return new_params


def main(convert_config: ConvertHFConfig):
    # qwen2vl获取到的config类型是Qn2VLConfig
    config = cast(Qwen2_5_VLConfig, convert_config.hf_config.config)
    parallel_config = convert_config.parallel_config
    state_dicts = load_from_mm(convert_config.mm_dir, parallel_config.vit_pp_layers, parallel_config.llm_pp_layers,
                               parallel_config.tp_size)
    state_dict = merge_by_tp(state_dicts, parallel_config.tp_size)
    state_dict = convert_mm_to_hf(state_dict, config)
    state_dicts = split_by_index_json(state_dict, convert_config.hf_config.hf_dir)
    copy_files_except_suffix(convert_config.hf_config.hf_dir, convert_config.save_hf_dir)
    save_by_index_json(state_dicts, convert_config.save_hf_dir)
