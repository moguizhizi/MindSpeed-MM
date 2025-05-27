#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : internvl2_mm_to_hf.py
@Time    : 2025/02/14
@Desc    : internvl2 mindspeed-mm模型转换成huggingface模型
"""
from pathlib import Path

import torch
import mindspeed.megatron_adaptor # noqa

from checkpoint.utils import ConvertVppHFConfig, copy_files_except_suffix, save_by_index_json, split_by_index_json


def merge_by_pp(pp_ckpt_file, pp_rank: int, _model_config_dict=None):
    _vit_pipeline_num_layers = _model_config_dict.vit_pp_layers[0]
    _llm_pipeline_num_layers = _model_config_dict.llm_pp_layers[0]

    vit_pp_start_index = 0
    llm_pp_start_index = 0
    if pp_rank > 0:
        vit_pp_start_index = sum(_vit_pipeline_num_layers[0: pp_rank])
        llm_pp_start_index = sum(_llm_pipeline_num_layers[0: pp_rank])
    new_dict = {}
    print(f"pp_rank: {pp_rank},  vit_pp_start_index: {vit_pp_start_index}, llm_pp_start_index: {llm_pp_start_index}")
    for parameter_name, tensor in torch.load(pp_ckpt_file)['model'].items():
        if parameter_name.startswith('text_decoder.decoder.layers.'):
            llm_parameter_name_list = parameter_name.split('.')
            llm_layer_id = int(llm_parameter_name_list[3])
            llm_merge_pp_mapp_layer_id = llm_layer_id + llm_pp_start_index
            llm_parameter_name_list[3] = str(llm_merge_pp_mapp_layer_id)
            llm_merge_parameter_name = '.'.join(llm_parameter_name_list)
            new_dict[llm_merge_parameter_name] = tensor
        elif parameter_name.startswith('image_encoder.encoder.encoder.layers.'):
            vit_parameter_name_list = parameter_name.split('.')
            vit_layer_id = int(vit_parameter_name_list[4])
            vit_merge_pp_mapp_layer_id = vit_layer_id + vit_pp_start_index
            vit_parameter_name_list[4] = str(vit_merge_pp_mapp_layer_id)
            vit_merge_parameter_name = '.'.join(vit_parameter_name_list)
            new_dict[vit_merge_parameter_name] = tensor
        else:
            new_dict[parameter_name] = tensor
    return new_dict


def load_from_mm(_load_dir, _model_config_dict=None):
    latest_iter_txt = "latest_checkpointed_iteration.txt"
    mm_load_dir = Path(_load_dir)
    ckpt_iteration_str = mm_load_dir.joinpath(latest_iter_txt).read_text().strip()
    ckpt_iter_dir_str = 'release'
    if ckpt_iteration_str != 'release':
        ckpt_iter_dir_str = 'iter_{:07d}'.format(int(ckpt_iteration_str))
    ckpt_iter_dir = mm_load_dir.joinpath(ckpt_iter_dir_str)
    _merge_state_dict = {}
    for pp_ckpt_dir in ckpt_iter_dir.iterdir():
        if not pp_ckpt_dir.is_dir():
            print(f'mm ckpt: {ckpt_iter_dir} has no pp split subdir, please check.')
            return _merge_state_dict
        for pp_ckpt_file in pp_ckpt_dir.glob('*.pt'):
            print(str(pp_ckpt_file).center(100, '_'))
            pp_ckpt_file_list = str(pp_ckpt_dir).split("_")
            pp_rank = int(pp_ckpt_file_list[-1])
            # if you want to get tp_rank add code: int(pp_ckpt_file_list[-2])
            _merge_state_dict.update(merge_by_pp(pp_ckpt_file, pp_rank, _model_config_dict))

    if not _merge_state_dict:
        print(f'mm ckpt: {_load_dir} load failed, please check.')
    return _merge_state_dict


def split_qkv(wqkv, hn=64, ng=8):
    wq = None
    wk = None
    wv = None
    return wq, wk, wv


def convert_mm_to_hf(_mm_state_dict, _llm_num_layers):
    _hf_state_dict = {}
    # check LlamaForCausalLM or InternLM2ForCausalLM
    architectures_key = "text_decoder.decoder.layers.0.self_attention.linear_qkv.weight"
    is_llama_for_causa_llm = True
    if architectures_key in _mm_state_dict.keys():
        is_llama_for_causa_llm = False
        print(f"-------------internvl2 is InternLM2ForCausalLM-----------")

    for key, value in _mm_state_dict.items():
        new_key = None
        if key.startswith('image_encoder.encoder'):
            new_key = key.replace('image_encoder.encoder', 'vision_model')
            new_key = new_key.replace('self_attention.linear_qkv', 'attn.qkv')
            new_key = new_key.replace('self_attention.q_layernorm', 'attn.q_norm')
            new_key = new_key.replace('self_attention.k_layernorm', 'attn.k_norm')
            new_key = new_key.replace('self_attention.linear_proj', 'attn.proj')
            new_key = new_key.replace('mlp.linear_fc1', 'mlp.fc1')
            new_key = new_key.replace('mlp.linear_fc2', 'mlp.fc2')
            new_key = new_key.replace('input_layernorm', 'norm1')
            new_key = new_key.replace('pre_mlp_layernorm', 'norm2')

        elif key.startswith('text_decoder'):
            if is_llama_for_causa_llm:
                new_key = key.replace('text_decoder', 'language_model')
                new_key = new_key.replace('embedding.word_embeddings', 'model.embed_tokens')
                new_key = new_key.replace('decoder.layers', 'model.layers')
                new_key = new_key.replace('self_attention.wq', 'self_attn.q_proj')
                new_key = new_key.replace('self_attention.wk', 'self_attn.k_proj')
                new_key = new_key.replace('self_attention.wv', 'self_attn.v_proj')
                new_key = new_key.replace('self_attention.linear_proj', 'self_attn.o_proj')
                new_key = new_key.replace('linear_fc1_gate', 'gate_proj')
                new_key = new_key.replace('linear_fc1_up', 'up_proj')
                new_key = new_key.replace('linear_fc2', 'down_proj')
                new_key = new_key.replace('pre_mlp_layernorm', 'post_attention_layernorm')
                new_key = new_key.replace('decoder.final_layernorm', 'model.norm')
                new_key = new_key.replace('output_layer', 'lm_head')
            else:
                new_key = key.replace('text_decoder', 'language_model')
                new_key = new_key.replace('embedding.word_embeddings', 'model.tok_embeddings')
                new_key = new_key.replace('decoder.layers', 'model.layers')
                new_key = new_key.replace('self_attention.linear_qkv', 'attention.wqkv')
                new_key = new_key.replace('self_attention.linear_proj', 'attention.wo')
                new_key = new_key.replace('mlp.linear_fc1', 'feed_forward.w1w3')
                new_key = new_key.replace('mlp.linear_fc2', 'feed_forward.w2')
                new_key = new_key.replace('input_layernorm', 'attention_norm')
                new_key = new_key.replace('pre_mlp_layernorm', 'ffn_norm')
                new_key = new_key.replace('decoder.final_layernorm', 'model.norm')
                new_key = new_key.replace('output_layer', 'output')

        elif key.startswith('image_encoder.projector'):
            new_key = key.replace('image_encoder.projector', 'mlp1')
            new_key = new_key.replace('norm', '0')
            new_key = new_key.replace('linear_fc1', '1')
            new_key = new_key.replace('linear_fc2', '3')

        print(f'mapping {key} to {new_key}')
        _hf_state_dict[new_key] = value

    if is_llama_for_causa_llm:
        for i in range(_llm_num_layers):
            q_name = f'language_model.model.layers.{i}.self_attention.wq.weight'
            k_name = f'language_model.model.layers.{i}.self_attention.wk.weight'
            v_name = f'language_model.model.layers.{i}.self_attention.wv.weight'
            qkv_name = f'language_model.model.layers.{i}.attention.wqkv.weight'

            if qkv_name in _hf_state_dict.keys():
                wqkv = _hf_state_dict[qkv_name]
            else:
                raise AssertionError(f'Missing key {qkv_name}')
            wq, wk, wv = split_qkv(wqkv)
            if not (wq and wk and wv):
                raise ValueError("llama_for_causa_llm split qkv weight error, maybe not support right now.")
            _hf_state_dict[q_name] = wq
            _hf_state_dict[k_name] = wk
            _hf_state_dict[v_name] = wv
            _hf_state_dict.pop(qkv_name)
            print(f'merge {q_name}, {k_name}, {v_name} to {qkv_name}')

    # split w1 and w3 weight
    for i in range(_llm_num_layers):
        gate_and_up_name = f'language_model.model.layers.{i}.feed_forward.w1w3.weight'
        gate_name = f'language_model.model.layers.{i}.feed_forward.w1.weight'
        up_name = f'language_model.model.layers.{i}.feed_forward.w3.weight'
        # split w1 和 w3
        if gate_and_up_name in _hf_state_dict.keys():
            gate_and_up_weight = _hf_state_dict[gate_and_up_name]
            gate_weight, up_weight = torch.split(gate_and_up_weight, gate_and_up_weight.size(0) // 2, dim=0)
            _hf_state_dict[gate_name] = gate_weight
            _hf_state_dict[up_name] = up_weight
        # remove useless weight
        _hf_state_dict.pop(gate_and_up_name)
        print(f'split {gate_and_up_name} to {gate_name} and {up_name}')
    return _hf_state_dict


def main(convert_config: ConvertVppHFConfig):
    config = convert_config.hf_config.config
    raw_hf_dir = convert_config.hf_config.hf_dir

    parallel_config = convert_config.parallel_config

    merge_state_dict = load_from_mm(convert_config.mm_dir, parallel_config)

    hf_state_dict = convert_mm_to_hf(merge_state_dict, config.llm_config.num_hidden_layers)
    state_dicts = split_by_index_json(hf_state_dict, raw_hf_dir)

    copy_files_except_suffix(raw_hf_dir, convert_config.save_hf_dir)

    save_by_index_json(state_dicts, convert_config.save_hf_dir)
