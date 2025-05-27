import argparse
import os
from copy import deepcopy
from dataclasses import dataclass
import stat
import re

import torch
from transformers import AutoModelForCausalLM, AutoConfig

llm_arch = ''


@dataclass
class InternVLModelConfig:
    model_size: str
    pp_size: int
    vpp_size: int
    vit_num_layers: int
    vit_pipeline_num_layers: list
    llm_num_layers: int
    llm_pipeline_num_layers: list


model_config_dict = {
    '1B': InternVLModelConfig(model_size='1B',
                              pp_size=1,
                              vpp_size=1,
                              vit_num_layers=24,
                              vit_pipeline_num_layers=[24, ],
                              llm_num_layers=24,
                              llm_pipeline_num_layers=[24, ]
                              ),
    '2B': InternVLModelConfig(model_size='2B',
                              pp_size=1,
                              vpp_size=1,
                              vit_num_layers=24,
                              vit_pipeline_num_layers=[24, ],
                              llm_num_layers=24,
                              llm_pipeline_num_layers=[24, ]),
    '4B': InternVLModelConfig(model_size='4B',
                              pp_size=1,
                              vpp_size=1,
                              vit_num_layers=24,
                              vit_pipeline_num_layers=[24, ],
                              llm_num_layers=36,
                              llm_pipeline_num_layers=[36, ]),
    '8B': InternVLModelConfig(model_size='8B',
                              pp_size=1,
                              vpp_size=1,
                              vit_num_layers=24,
                              vit_pipeline_num_layers=[24, ],
                              llm_num_layers=32,
                              llm_pipeline_num_layers=[32, ]),
    '26B': InternVLModelConfig(model_size='26B',
                              pp_size=1,
                              vpp_size=1,
                              vit_num_layers=45,
                              vit_pipeline_num_layers=[45, ],
                              llm_num_layers=48,
                              llm_pipeline_num_layers=[48, ]),
    '38B': InternVLModelConfig(model_size='38B',
                              pp_size=1,
                              vpp_size=1,
                              vit_num_layers=45,
                              vit_pipeline_num_layers=[45, ],
                              llm_num_layers=64,
                              llm_pipeline_num_layers=[64, ]),
    '78B': InternVLModelConfig(model_size='78B',
                              pp_size=16,
                              vpp_size=1,
                              vit_num_layers=45,
                              vit_pipeline_num_layers=[45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              llm_num_layers=80,
                              llm_pipeline_num_layers=[0, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 2])
}


def load_from_hf(load_dir, trust_remote_code):
    # Load Huggingface model.
    hf_model = AutoModelForCausalLM.from_pretrained(
        load_dir, device_map='cpu',
        trust_remote_code=trust_remote_code,
        local_files_only=True)
    print(hf_model)
    config = AutoConfig.from_pretrained(
        load_dir, trust_remote_code=trust_remote_code, local_files_only=True)
    global llm_arch
    llm_arch = config.llm_config.architectures[0]
    return hf_model, config


def get_model_config(model_size, enable_vpp) -> InternVLModelConfig:
    if model_size not in model_config_dict:
        raise KeyError(f" {model_size} not exist in model config dict.")
    if enable_vpp:
        raise ValueError("InternVL2.5 does not support vpp right now.")
    return model_config_dict.get(model_size)


def merge_pp_index(model_config):
    pp_size = model_config.pp_size
    vp_size = model_config.vpp_size
    vit_num_layers = model_config.vit_num_layers
    vit_pipeline_num_layers = model_config.vit_pipeline_num_layers
    llm_num_layers = model_config.llm_num_layers
    llm_pipeline_num_layers = model_config.llm_pipeline_num_layers
    # Flatten the vit and llm layers for VPP
    if vp_size > 1:
        vit_pipeline_num_layers_flat = [item for sublist in vit_pipeline_num_layers for item in sublist]
        llm_pipeline_num_layers_flat = [item for sublist in llm_pipeline_num_layers for item in sublist]
    else:
        vit_pipeline_num_layers_flat = vit_pipeline_num_layers
        llm_pipeline_num_layers_flat = llm_pipeline_num_layers

    # Validation for flattened lists
    expected_length = pp_size * vp_size
    if len(vit_pipeline_num_layers_flat) != expected_length:
        raise AssertionError(f'Length of vit_pipeline_num_layers_flat must be equal to pp_size * vp_size, '
                             f'but got {len(vit_pipeline_num_layers_flat)} and {expected_length}.')
    if sum(vit_pipeline_num_layers_flat) != vit_num_layers:
        raise AssertionError(f'Sum of vit_pipeline_num_layers_flat must be equal to vit_num_layers, '
                             f'but got {sum(vit_pipeline_num_layers_flat)} and {vit_num_layers}.')
    if len(llm_pipeline_num_layers_flat) != expected_length:
        raise AssertionError(f'Length of llm_pipeline_num_layers_flat must be equal to pp_size * vp_size, '
                             f'but got {len(llm_pipeline_num_layers_flat)} and {expected_length}.')
    if sum(llm_pipeline_num_layers_flat) != llm_num_layers:
        raise AssertionError(f'Sum of llm_pipeline_num_layers_flat must be equal to llm_num_layers, '
                             f'but got {sum(llm_pipeline_num_layers_flat)} and {llm_num_layers}.')

    # Generate split method
    split_method = []
    for vit_num, llm_num in zip(vit_pipeline_num_layers_flat, llm_pipeline_num_layers_flat):
        split_method.append((vit_num, llm_num))
    return split_method


def convert_hg_to_mm(_state_dict, model_config, num_key_value_heads):
    _num_layers = model_config.llm_num_layers
    new_dict = {}
    for key, value in _state_dict.items():
        new_key = None
        if key.startswith('vision_model'):
            new_key = key.replace('vision_model', 'image_encoder.encoder')
            new_key = new_key.replace('attn.qkv', 'self_attention.linear_qkv')
            new_key = new_key.replace('attn.q_norm', 'self_attention.q_layernorm')
            new_key = new_key.replace('attn.k_norm', 'self_attention.k_layernorm')
            new_key = new_key.replace('attn.proj', 'self_attention.linear_proj')
            new_key = new_key.replace('mlp.fc1', 'mlp.linear_fc1')
            new_key = new_key.replace('mlp.fc2', 'mlp.linear_fc2')
            new_key = new_key.replace('norm1', 'input_layernorm')
            new_key = new_key.replace('norm2', 'pre_mlp_layernorm')

        elif key.startswith('language_model'):
            if llm_arch == 'LlamaForCausalLM':
                new_key = key.replace('language_model', 'text_decoder')
                new_key = new_key.replace('model.embed_tokens', 'embedding.word_embeddings')
                new_key = new_key.replace('model.layers', 'decoder.layers')
                new_key = new_key.replace('self_attn.q_proj', 'self_attention.wq')
                new_key = new_key.replace('self_attn.k_proj', 'self_attention.wk')
                new_key = new_key.replace('self_attn.v_proj', 'self_attention.wv')
                new_key = new_key.replace('self_attn.o_proj', 'self_attention.linear_proj')
                new_key = new_key.replace('gate_proj', 'linear_fc1_gate')
                new_key = new_key.replace('up_proj', 'linear_fc1_up')
                new_key = new_key.replace('down_proj', 'linear_fc2')
                new_key = new_key.replace('post_attention_layernorm', 'pre_mlp_layernorm')
                new_key = new_key.replace('model.norm', 'decoder.final_layernorm')
                new_key = new_key.replace('lm_head', 'output_layer')
            elif llm_arch == 'InternLM2ForCausalLM':
                new_key = key.replace('language_model', 'text_decoder')
                new_key = new_key.replace('model.tok_embeddings', 'embedding.word_embeddings')
                new_key = new_key.replace('model.layers', 'decoder.layers')
                new_key = new_key.replace('attention.wqkv', 'self_attention.linear_qkv')
                new_key = new_key.replace('attention.wo', 'self_attention.linear_proj')
                new_key = new_key.replace('feed_forward.w1', 'mlp.linear_fc1_gate')
                new_key = new_key.replace('feed_forward.w3', 'mlp.linear_fc1_up')
                new_key = new_key.replace('feed_forward.w2', 'mlp.linear_fc2')
                new_key = new_key.replace('attention_norm', 'input_layernorm')
                new_key = new_key.replace('ffn_norm', 'pre_mlp_layernorm')
                new_key = new_key.replace('model.norm', 'decoder.final_layernorm')
                new_key = new_key.replace('output', 'output_layer')
            elif llm_arch == 'Qwen2ForCausalLM':
                new_key = key.replace('language_model', 'text_decoder')
                new_key = new_key.replace('lm_head', 'output_layer')
                new_key = new_key.replace('model.layers', 'decoder.layers')
                new_key = new_key.replace('self_attn.q_proj', 'self_attention.linear_q')
                new_key = new_key.replace('self_attn.k_proj', 'self_attention.linear_k')
                new_key = new_key.replace('self_attn.v_proj', 'self_attention.linear_v')
                new_key = new_key.replace('self_attn.o_proj', 'self_attention.linear_proj')
                new_key = new_key.replace('post_attention_layernorm', 'pre_mlp_layernorm')
                new_key = new_key.replace('gate_proj', 'linear_fc1_gate')
                new_key = new_key.replace('up_proj', 'linear_fc1_up')
                new_key = new_key.replace('down_proj', 'linear_fc2')
                new_key = new_key.replace('model.norm', 'decoder.final_layernorm')
                new_key = new_key.replace('model.embed_tokens', 'embedding.word_embeddings')

        elif key.startswith('mlp1'):
            new_key = key.replace('mlp1', 'image_encoder.projector')
            new_key = new_key.replace('0', 'norm')
            new_key = new_key.replace('1', 'linear_fc1')
            new_key = new_key.replace('3', 'linear_fc2')


        print(f'mapping {key} to {new_key}')
        new_dict[new_key] = value


    if llm_arch == 'Qwen2ForCausalLM':
        # merge qkv weight
        for i in range(_num_layers):
            q_name = f'text_decoder.decoder.layers.{i}.self_attention.linear_q.weight'
            k_name = f'text_decoder.decoder.layers.{i}.self_attention.linear_k.weight'
            v_name = f'text_decoder.decoder.layers.{i}.self_attention.linear_v.weight'
            qkv_name = f'text_decoder.decoder.layers.{i}.self_attention.linear_qkv.weight'

            if q_name in new_dict.keys():
                wq = new_dict[q_name]
            else:
                raise AssertionError(f'Missing key {q_name}')
            if k_name in new_dict.keys():
                wk = new_dict[k_name]
            else:
                raise AssertionError(f'Missing key {k_name}')
            if v_name in new_dict.keys():
                wv = new_dict[v_name]
            else:
                raise AssertionError(f'Missing key {v_name}')

            q_chunks = torch.chunk(wq, num_key_value_heads, dim=0)
            k_chunks = torch.chunk(wk, num_key_value_heads, dim=0)
            v_chunks = torch.chunk(wv, num_key_value_heads, dim=0)
            all_chunks = []
            for j in range(num_key_value_heads):
                all_chunks.append(q_chunks[j])
                all_chunks.append(k_chunks[j])
                all_chunks.append(v_chunks[j])
            concatenated_tensor = torch.cat(all_chunks, dim=0)
            new_dict[qkv_name] = concatenated_tensor
            if q_name in new_dict:
                new_dict.pop(q_name)
            if k_name in new_dict:
                new_dict.pop(k_name)
            if v_name in new_dict:
                new_dict.pop(v_name)


        # merge qkv bias
        for i in range(_num_layers):
            q_name = f'text_decoder.decoder.layers.{i}.self_attention.linear_q.bias'
            k_name = f'text_decoder.decoder.layers.{i}.self_attention.linear_k.bias'
            v_name = f'text_decoder.decoder.layers.{i}.self_attention.linear_v.bias'
            qkv_name = f'text_decoder.decoder.layers.{i}.self_attention.linear_qkv.bias'

            if q_name in new_dict.keys():
                wq = new_dict[q_name]
            else:
                raise AssertionError(f'Missing key {q_name}')
            if k_name in new_dict.keys():
                wk = new_dict[k_name]
            else:
                raise AssertionError(f'Missing key {k_name}')
            if v_name in new_dict.keys():
                wv = new_dict[v_name]
            else:
                raise AssertionError(f'Missing key {v_name}')

            q_chunks = torch.chunk(wq, num_key_value_heads, dim=0)
            k_chunks = torch.chunk(wk, num_key_value_heads, dim=0)
            v_chunks = torch.chunk(wv, num_key_value_heads, dim=0)
            all_chunks = []
            for j in range(num_key_value_heads):
                all_chunks.append(q_chunks[j])
                all_chunks.append(k_chunks[j])
                all_chunks.append(v_chunks[j])
            concatenated_tensor = torch.cat(all_chunks, dim=0)
            new_dict[qkv_name] = concatenated_tensor
            if q_name in new_dict:
                new_dict.pop(q_name)
            if k_name in new_dict:
                new_dict.pop(k_name)
            if v_name in new_dict:
                new_dict.pop(v_name)


    # 合并mlp的gate和up权重
    for i in range(_num_layers):
        gate_name = f'text_decoder.decoder.layers.{i}.mlp.linear_fc1_gate.weight'
        up_name = f'text_decoder.decoder.layers.{i}.mlp.linear_fc1_up.weight'
        fc1_name = f'text_decoder.decoder.layers.{i}.mlp.linear_fc1.weight'

        # 合并 w1 和 w3
        if gate_name in new_dict.keys():
            gate_proj_weight = new_dict[gate_name]
        if up_name in new_dict.keys():
            up_proj_weight = new_dict[up_name]
        linear_fc1 = torch.cat([gate_proj_weight, up_proj_weight], dim=0)
        new_dict[fc1_name] = linear_fc1

        # 移除合并前的权重
        new_dict.pop(gate_name)
        new_dict.pop(up_name)

        print(f'merge {gate_name} and {up_name} to {fc1_name}')

    return new_dict


def split_model_by_pipeline(state_dict, pp_split):
    if pp_split is None or len(pp_split) <= 1:
        return [state_dict], {}

    pp_size = len(pp_split)
    vit_range = [0, 0]
    llm_range = [pp_size - 1, pp_size - 1]
    for pp_rank, (vit_num, llm_num) in enumerate(pp_split):
        if vit_num > 0 and pp_rank > vit_range[1]:
            vit_range[1] = pp_rank
        if llm_num > 0 and pp_rank < llm_range[0]:
            llm_range[0] = pp_rank
    print(f'vit range: {vit_range[0]}~{vit_range[1]}')
    print(f'llm range: {llm_range[0]}~{llm_range[1]}')

    vit_start_idx = 0
    llm_start_idx = 0
    return_dicts = []
    copy_dict = deepcopy(state_dict)
    for pp_rank, (vit_num, llm_num) in enumerate(pp_split):
        vit_end_idx = vit_start_idx + vit_num
        llm_end_idx = llm_start_idx + llm_num
        new_dict = {}
        for key, value in state_dict.items():
            if key.startswith('image_encoder.encoder.embeddings.'):
                if pp_rank == vit_range[0]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('image_encoder.encoder.encoder.layers.'):
                layer_idx = int(key.split('.')[4])
                if vit_start_idx <= layer_idx < vit_end_idx and vit_range[0] <= pp_rank <= vit_range[1]:
                    new_idx = layer_idx - vit_start_idx
                    new_key = key.replace(f'{layer_idx}', f'{new_idx}', 1)
                    new_dict[new_key] = value
                    copy_dict.pop(key)
            elif key.startswith('image_encoder.projector.'):
                if pp_rank == vit_range[1]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.embedding.'):
                if pp_rank == llm_range[0]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.decoder.layers.'):
                layer_idx = int(key.split('.')[3])
                if llm_start_idx <= layer_idx < llm_end_idx and llm_range[0] <= pp_rank <= llm_range[1]:
                    new_idx = layer_idx - llm_start_idx
                    new_key = key.replace(f'{layer_idx}', f'{new_idx}', 1)
                    new_dict[new_key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.decoder.final_layernorm.'):
                if pp_rank == llm_range[1]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.output_layer.'):
                if pp_rank == llm_range[1]:
                    new_dict[key] = value
                    copy_dict.pop(key)
        vit_start_idx = vit_end_idx
        llm_start_idx = llm_end_idx
        return_dicts.append(new_dict)
    return return_dicts, copy_dict


def save_by_pp(_state_dicts, model_config, _save_dir, _latest_checkpointed_iteration='release', _exists_ok=False):
    pp_size = model_config.pp_size
    vp_size = model_config.vpp_size
    if os.path.exists(_save_dir):
        if not _exists_ok:
            print(f'save dir: {_save_dir} exists, please check.')
            return
    os.makedirs(_save_dir)
    flags = os.O_WRONLY | os.O_CREAT
    mode = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(os.path.join(_save_dir, 'latest_checkpointed_iteration.txt'), flags, mode), 'w') as fout:
        fout.write(_latest_checkpointed_iteration)

    if _latest_checkpointed_iteration == 'release':
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(int(_latest_checkpointed_iteration))

    os.makedirs(os.path.join(_save_dir, directory), exist_ok=True)

    if pp_size > 1:
        for pp_rank in range(pp_size):
            tp_rank = 0
            os.makedirs(os.path.join(_save_dir, directory, f'mp_rank_{tp_rank:02d}_{pp_rank:03d}'))
            save_path = os.path.join(_save_dir, directory, f'mp_rank_{tp_rank:02d}_{pp_rank:03d}', 'model_optim_rng.pt')
            save_dict = {}
            if vp_size > 1:
                # Collect VP state dicts for this PP rank
                save_dict = {f'model{vp_idx}': _state_dicts[vp_idx * pp_size + pp_rank] for vp_idx in range(vp_size)}
                save_dict['checkpoint_version'] = 3.0
            else:
                save_dict = {'model': _state_dicts[pp_rank]}
            torch.save(save_dict, save_path)
        return
    _state_dict = _state_dicts[0]
    tp_rank = 0
    os.makedirs(os.path.join(_save_dir, directory, f'mp_rank_{tp_rank:02d}'))
    save_path = os.path.join(_save_dir, directory, f'mp_rank_{tp_rank:02d}', 'model_optim_rng.pt')
    save_dict = {}
    save_dict['model'] = _state_dict
    torch.save(save_dict, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Megatron Checkpoint Utility Arguments',
                                     allow_abbrev=False,
                                     conflict_handler='resolve')
    parser.add_argument('--model-size', type=str, required=True,
                        help='model size, [4B,]')
    parser.add_argument('--vpp', type=bool, default=False,
                        help='Whether or not to split the weights into VPP weights')
    parser.add_argument('--load-dir', type=str, required=True,
                        help='HuggingFace weight path for loading')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='MindSpeed-MM weight path for saving')
    parser.add_argument('--trust-remote-code', type=str, required=True, default=False,
                        help='Whether or not to allow HuggingFace API to execute code')
    args, unrecognized_args = parser.parse_known_args()
    if unrecognized_args:
        print(f"Unrecognized Args: {unrecognized_args}")

    hf_model, hf_config = load_from_hf(args.load_dir, args.trust_remote_code)
    num_key_value_heads = hf_config.llm_config.num_key_value_heads
    state_dict = hf_model.state_dict()
    model_config = get_model_config(
        args.model_size, args.vpp)
    pp_split = merge_pp_index(model_config)

    for key, value in state_dict.items():
        print(key, value.shape)
    state_dict = convert_hg_to_mm(state_dict, model_config, num_key_value_heads)
    pipeline_state_dicts, remains = split_model_by_pipeline(state_dict, pp_split)

    if len(remains) > 0:
        print(remains)
        raise RuntimeWarning("There are some weights ungrouped.")


    for rank, pipeline_state_dict in enumerate(pipeline_state_dicts):
        print(20 * '#', f'stage {rank}', 20 * '#')
        for key, value in pipeline_state_dict.items():
            print(key, value.shape)
    save_by_pp(pipeline_state_dicts, model_config, args.save_dir, _exists_ok=True)

