import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoConfig


def load_from_hf(load_dir, trust_remote_code):
    # Load Huggingface model.
    hf_model = AutoModelForCausalLM.from_pretrained(
        load_dir, device_map='cpu',
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16, local_files_only=True)
    print(hf_model)
    config = AutoConfig.from_pretrained(
        load_dir, trust_remote_code=trust_remote_code, local_files_only=True)

    return hf_model, config


def merge_qkv(wq, wk, wv, ng=32):
    hq, h = wq.shape
    hkv = wk.shape[0]
    dq = hq // ng
    dkv = hkv // ng
    d = dq + 2 * dkv
    qkv = torch.zeros([hq + hkv * 2, h], dtype=wq.dtype)
    for j in range(ng):
        qkv[j * d : j * d + dq, :] = wq[j * dq : (j + 1) * dq, :]
        qkv[j * d + dq : j * d + dq + dkv, :] = wk[j * dkv : (j + 1) * dkv, :]
        qkv[j * d + dq + dkv : j * d + dq + dkv * 2, :] = wv[j * dkv : (j + 1) * dkv, :]

    return qkv


def convert_hg_to_mm(_state_dict, _model_config):
    _num_layers = _model_config.num_hidden_layers
    _num_query_groups = _model_config.num_key_value_heads
    new_dict = {}
    for key, value in _state_dict.items():
        new_key = key
        # 权重映射
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

        # 打印映射过程
        print(f'mapping {key} to {new_key}')
        new_dict[new_key] = value

    # qkv权重交织合并
    for i in range(_num_layers):
        q_name = f'decoder.layers.{i}.self_attention.wq.weight'
        k_name = f'decoder.layers.{i}.self_attention.wk.weight'
        v_name = f'decoder.layers.{i}.self_attention.wv.weight'
        qkv_name = f'decoder.layers.{i}.self_attention.linear_qkv.weight'

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
        wqkv = merge_qkv(wq, wk, wv, _num_query_groups)
        new_dict[qkv_name] = wqkv
        new_dict.pop(q_name)
        new_dict.pop(k_name)
        new_dict.pop(v_name)

        print(f'merge {q_name}, {k_name}, {v_name} to {qkv_name}')

    # 合并mlp的gate和up权重
    for i in range(_num_layers):
        gate_name = f'decoder.layers.{i}.mlp.linear_fc1_gate.weight'
        up_name = f'decoder.layers.{i}.mlp.linear_fc1_up.weight'
        fc1_name = f'decoder.layers.{i}.mlp.linear_fc1.weight'

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Megatron Checkpoint Utility Arguments',
                                     allow_abbrev=False,
                                     conflict_handler='resolve')
    parser.add_argument('--load-dir', type=str, required=True,
                        help='HuggingFace weight path for loading')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='MindSpeed-MM weight path for saving')
    parser.add_argument('--trust-remote-code', type=str, required=True, default=False,
                        help='Whether or not to allow HuggingFace API to execute code')
    args, unrecognized_args = parser.parse_known_args()
    if unrecognized_args:
        print(f"Unrecognized Args: {unrecognized_args}")

    hf_model, model_config = load_from_hf(args.load_dir, args.trust_remote_code)
    state_dict = hf_model.state_dict()
    print(50 * '*')
    print('origin state_dict:')
    for key, value in state_dict.items():
        print(key, value.shape)
    print(50 * '*')
    new_state_dict = convert_hg_to_mm(state_dict, model_config)
    print('new state_dict:')
    for key, value in new_state_dict.items():
        print(key, value.shape)
    print(50 * '*')
    output_path = os.path.join(args.save_dir, 'converted_vicuna.pt')
    torch.save(new_state_dict, output_path)
    print('all weights have been converted.')