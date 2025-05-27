import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass

import torch
import mindspeed.megatron_adaptor  # noqa
from safetensors.torch import save_file


@dataclass
class Internvl2ModelConfig:
    model_size: str
    pp_size: int
    vit_num_layers: int
    vit_pipeline_num_layers: list
    llm_num_layers: int
    llm_pipeline_num_layers: list


model_config_dict = {
    '2B': Internvl2ModelConfig('2B',
                               1,
                               24, [24, ],
                               24, [24, ]
                               ),
    '8B': Internvl2ModelConfig('2B',
                               4,
                               24, [24, 0, 0, 0],
                               32, [6, 9, 9, 8]
                               ),
    '26B': Internvl2ModelConfig('26B',
                                8,
                                45, [14, 16, 15, 0, 0, 0, 0, 0],
                                48, [0, 0, 0, 9, 10, 10, 10, 10, 9]
                                ),
    '76B': Internvl2ModelConfig('76B',
                                16,
                                45, [11, 12, 12, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                80, [0, 0, 0, 1, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6]
                                ),
}


def get_model_config(model_size) -> Internvl2ModelConfig:
    if model_size not in model_config_dict:
        raise KeyError(f" {model_size} not exist in model config dict.")
    return model_config_dict[model_size]


def check_pp_config(_model_config_dict=None):
    pp_size = _model_config_dict.pp_size
    vit_num_layers = _model_config_dict.vit_num_layers
    vit_pipeline_num_layers = _model_config_dict.vit_pipeline_num_layers
    llm_num_layers = _model_config_dict.llm_num_layers
    llm_pipeline_num_layers = _model_config_dict.llm_pipeline_num_layers
    if len(vit_pipeline_num_layers) != pp_size:
        raise AssertionError(f'length of vit_pipeline_num_layers must be equal to pp_size, '
                             f'but got {len(vit_pipeline_num_layers)} and {pp_size}.')
    if sum(vit_pipeline_num_layers) != vit_num_layers:
        raise AssertionError(f'sum of vit_pipeline_num_layers must be equal to vit_num_layers, '
                             f'but got {sum(vit_pipeline_num_layers)} and {vit_num_layers}.')
    if len(llm_pipeline_num_layers) != pp_size:
        raise AssertionError(f'length of llm_pipeline_num_layers must be equal to pp_size, '
                             f'but got {len(llm_pipeline_num_layers)} and {pp_size}.')
    if sum(llm_pipeline_num_layers) != llm_num_layers:
        raise AssertionError(f'sum of llm_pipeline_num_layers must be equal to llm_num_layers, '
                             f'but got {sum(llm_pipeline_num_layers)} and {llm_num_layers}.')


def merge_by_pp(pp_ckpt_file, pp_rank: int, _model_config_dict=None):
    _vit_pipeline_num_layers = _model_config_dict.vit_pipeline_num_layers
    _llm_pipeline_num_layers = _model_config_dict.llm_pipeline_num_layers

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


def convert_mm_to_hf(_mm_state_dict, _model_config_dict=None):
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
        for i in range(_model_config_dict.llm_num_layers):
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
    for i in range(_model_config_dict.llm_num_layers):
        gate_and_up_name = f'language_model.model.layers.{i}.feed_forward.w1w3.weight'
        gate_name = f'language_model.model.layers.{i}.feed_forward.w1.weight'
        up_name = f'language_model.model.layers.{i}.feed_forward.w3.weight'
        # split w1 和 w3
        if gate_and_up_name in _hf_state_dict.keys():
            gate_and_up_weight = _hf_state_dict[gate_and_up_name]
            # refer to: torch.cat([gate_proj_weight, up_proj_weight], dim=0)
            gate_weight, up_weight = torch.split(gate_and_up_weight, gate_and_up_weight.size(0) // 2, dim=0)
            _hf_state_dict[gate_name] = gate_weight
            _hf_state_dict[up_name] = up_weight
        # remove useless weight
        _hf_state_dict.pop(gate_and_up_name)
        print(f'split {gate_and_up_name} to {gate_name} and {up_name}')
    return _hf_state_dict


def split_by_index_json(_state_dict, _index_json_path):
    return_dicts = []
    with open(_index_json_path, 'r', encoding='utf-8') as file:
        weight_map = json.load(file)['weight_map']
    for key, value in weight_map.items():
        index = int(value.split('-')[1])
        while index > len(return_dicts):
            return_dicts.append({})
        return_dicts[index - 1][key] = _state_dict[key]
    return return_dicts


def save_by_index_json(_state_dicts, _save_dir):
    metadata = {
            'format': 'pt'
            }
    for index, state_dict in enumerate(_state_dicts, start=1):
        name = f'model-{index:05}-of-{len(_state_dicts):05}.safetensors'
        save_file(state_dict, Path(_save_dir).joinpath(name), metadata=metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mm2hf tools checkpoint utility arguments',
                                     allow_abbrev=False,
                                     conflict_handler='resolve')
    parser.add_argument('--model-size', type=str, required=True,
                        help='model size, 2B/8B/26B/76B')
    parser.add_argument('--load-dir', type=str, required=True,
                        help='MindSpeed-MM checkpoint path for loading')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='HuggingFace checkpoint path for saving')
    parser.add_argument('--raw-hf-dir', type=str, required=True,
                        help='original raw huggingface checkpoint path for loading')
    parser.add_argument('--trust-remote-code', type=str, required=True, default=False,
                        help='Whether not to allow HuggingFace API to execute code')
    args, unrecognized_args = parser.parse_known_args()
    if unrecognized_args:
        ValueError(f"please check unrecognized args: {unrecognized_args}")

    index_json_path = os.path.join(args.raw_hf_dir, "model.safetensors.index.json")
    if not os.path.exists(index_json_path):
        raise ValueError(f"safetensors.index.json not in {index_json_path}")
    model_config_ = get_model_config(args.model_size)
    check_pp_config(model_config_)
    merge_state_dict = load_from_mm(args.load_dir, model_config_)

    hf_state_dict = convert_mm_to_hf(merge_state_dict, model_config_)
    state_dicts = split_by_index_json(hf_state_dict, index_json_path)
    save_by_index_json(state_dicts, args.save_dir)

