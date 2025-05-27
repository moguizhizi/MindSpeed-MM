import os
import stat
import copy
import argparse
from typing import Any, Dict, List
from pathlib import Path

import torch
from safetensors.torch import load_file


DIT_CONVERT_MAPPING = {
    "pos_embed.proj.bias": "pos_embed.proj.bias",
    "pos_embed.proj.weight": "pos_embed.proj.weight",
    "scale_shift_table": "scale_shift_table",
    "adaln_single.emb.timestep_embedder.linear_1.bias": "adaln_single.emb.timestep_embedder.linear_1.bias",
    "adaln_single.emb.timestep_embedder.linear_1.weight": "adaln_single.emb.timestep_embedder.linear_1.weight",
    "adaln_single.emb.timestep_embedder.linear_2.bias": "adaln_single.emb.timestep_embedder.linear_2.bias",
    "adaln_single.emb.timestep_embedder.linear_2.weight": "adaln_single.emb.timestep_embedder.linear_2.weight",
    "caption_projection.linear_1.bias": "caption_projection.linear_1.bias",
    "caption_projection.linear_1.weight": "caption_projection.linear_1.weight",
    "caption_projection.linear_2.bias": "caption_projection.linear_2.bias",
    "caption_projection.linear_2.weight": "caption_projection.linear_2.weight",
    "clip_projection.bias": "clip_projection.bias",
    "clip_projection.weight": "clip_projection.weight",
    "proj_out.bias": "proj_out.bias",
    "proj_out.weight": "proj_out.weight"
}


def get_layer_mapping(i: int) -> Dict:
    layer_mapping = {}
    layer_mapping[f"transformer_blocks.{i}.attn1.k_norm.weight"] = f"transformer_blocks.{i}.attn1.k_norm.weight"
    layer_mapping[f"transformer_blocks.{i}.attn1.q_norm.weight"] = f"transformer_blocks.{i}.attn1.q_norm.weight"
    layer_mapping[f"transformer_blocks.{i}.attn1.wo.weight"] = f"transformer_blocks.{i}.attn1.proj_out.weight"
    layer_mapping[f"transformer_blocks.{i}.attn1.wqkv.weight"] = f"transformer_blocks.{i}.attn1.proj_qkv.weight"
    layer_mapping[f"transformer_blocks.{i}.attn2.k_norm.weight"] = f"transformer_blocks.{i}.attn2.k_norm.weight"
    layer_mapping[f"transformer_blocks.{i}.attn2.q_norm.weight"] = f"transformer_blocks.{i}.attn2.q_norm.weight"
    layer_mapping[f"transformer_blocks.{i}.attn2.wkv.weight"] = f"transformer_blocks.{i}.attn2.proj_kv.weight"
    layer_mapping[f"transformer_blocks.{i}.attn2.wo.weight"] = f"transformer_blocks.{i}.attn2.proj_out.weight"
    layer_mapping[f"transformer_blocks.{i}.attn2.wq.weight"] = f"transformer_blocks.{i}.attn2.proj_q.weight"
    layer_mapping[f"transformer_blocks.{i}.ff.net.0.proj.weight"] = f"transformer_blocks.{i}.ff.net.0.proj.weight"
    layer_mapping[f"transformer_blocks.{i}.ff.net.2.weight"] = f"transformer_blocks.{i}.ff.net.2.weight"
    layer_mapping[f"transformer_blocks.{i}.norm1.bias"] = f"transformer_blocks.{i}.norm1.bias"
    layer_mapping[f"transformer_blocks.{i}.norm1.weight"] = f"transformer_blocks.{i}.norm1.weight"
    layer_mapping[f"transformer_blocks.{i}.norm2.bias"] = f"transformer_blocks.{i}.norm2.bias"
    layer_mapping[f"transformer_blocks.{i}.norm2.weight"] = f"transformer_blocks.{i}.norm2.weight"
    layer_mapping[f"transformer_blocks.{i}.scale_shift_table"] = f"transformer_blocks.{i}.scale_shift_table"

    return layer_mapping


def load_from_hf(_load_dir):
    # Load Huggingface model 。
    load_dir = Path(_load_dir)
    safetensors_files = list(load_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No *.safetensors files found in {load_dir}")
    state_dict = {}
    for safe_path in safetensors_files:
        state_dict.update(load_file(str(safe_path), device='cpu'))
    return state_dict


def update_state_dict_inplace(
    state_dict: Dict[str, Any],
    convert_mapping: Dict
):
    for old_key in convert_mapping:
        new_key = convert_mapping[old_key]
        if old_key in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)
        else:
            print(f"Warning: missing update key {old_key}")


def split_by_tp(state_dict: Dict[str, Any], tp_size: int = 2, num_layers: int = 48, num_heads: int = 48) -> List[Dict]:
    if tp_size <= 1:
        return [state_dict, ]
    
    num_heads_per_tp = num_heads // tp_size
    new_state_dicts = []
    for tp_rank in range(tp_size):
        new_state_dict = copy.deepcopy(state_dict)

        for index in range(num_layers):
            # Common ColumnParallelLinear
            suffixed_common_col = [
                f"transformer_blocks.{index}.ff.net.0.proj.weight",
                f"transformer_blocks.{index}.attn2.proj_q.weight",
            ]
            # Common RowParallelLinear
            suffixed_common_row = [
                f"transformer_blocks.{index}.attn1.proj_out.weight",
                f"transformer_blocks.{index}.attn2.proj_out.weight",
                f"transformer_blocks.{index}.ff.net.2.weight",
            ]
            # Self Attention qkv ColumnParallelLinear
            suffixed_col_qkv_concat = [
                f"transformer_blocks.{index}.attn1.proj_qkv.weight",
            ]
            # Cross Attention kv ColumnParallelLinear
            suffixed_col_kv_concat = [
                f"transformer_blocks.{index}.attn2.proj_kv.weight",
            ]

            for split_name in suffixed_common_col:
                new_state_dict[split_name] = torch.chunk(state_dict[split_name], tp_size, dim=0)[tp_rank].clone()
            for split_name in suffixed_common_row:
                new_state_dict[split_name] = torch.chunk(state_dict[split_name], tp_size, dim=1)[tp_rank].clone()
            for split_name in suffixed_col_qkv_concat:
                weight = torch.chunk(state_dict[split_name], tp_size, dim=0)[tp_rank]
                qkvs_per_head = torch.chunk(weight, num_heads_per_tp, dim=0)
                qs_per_head = [torch.chunk(w, 3, dim=0)[0] for w in qkvs_per_head]
                ks_per_head = [torch.chunk(w, 3, dim=0)[1] for w in qkvs_per_head]
                vs_per_head = [torch.chunk(w, 3, dim=0)[2] for w in qkvs_per_head]
                
                weight = torch.cat(qs_per_head + ks_per_head + vs_per_head, dim=0)
                new_state_dict[split_name] = weight.clone()
            for split_name in suffixed_col_kv_concat:
                weight = torch.chunk(state_dict[split_name], tp_size, dim=0)[tp_rank]
                weight = torch.chunk(state_dict[split_name], tp_size, dim=0)[tp_rank]
                kvs_per_head = torch.chunk(weight, num_heads_per_tp, dim=0)
                ks_per_head = [torch.chunk(w, 2, dim=0)[0] for w in kvs_per_head]
                vs_per_head = [torch.chunk(w, 2, dim=0)[1] for w in kvs_per_head]
                
                weight = torch.cat(ks_per_head + vs_per_head, dim=0)
                new_state_dict[split_name] = weight.clone()
        # adaLN modulation
        col_split_names = [
            "adaln_single.linear.bias",
            "adaln_single.linear.weight"
        ]
        for split_name in col_split_names:
            new_state_dict[split_name] = torch.chunk(state_dict[split_name], tp_size, dim=0)[tp_rank].clone()
        
        new_state_dicts.append(new_state_dict)
    
    res_state_dict = {}
    for tp_rank, state_dict in enumerate(new_state_dicts):
        res_state_dict[(0, tp_rank)] = state_dict

    return res_state_dict


def merge_by_tp(state_dicts: Dict[str, Any], num_layers: int, tp_size: int, is_last_pp_stage: bool):
    if tp_size == 1:
        return state_dicts[0]
    
    merged_state_dict = copy.deepcopy(state_dicts[0])
    for index in range(num_layers):
        # Common ColumnParallelLinear
        suffixed_common_col = [
            f"transformer_blocks.{index}.ff.net.0.proj.weight",
            f"transformer_blocks.{index}.attn2.proj_q.weight",
        ]
        # Common RowParallelLinear
        suffixed_common_row = [
            f"transformer_blocks.{index}.attn1.proj_out.weight",
            f"transformer_blocks.{index}.attn2.proj_out.weight",
            f"transformer_blocks.{index}.ff.net.2.weight",
        ]
        # Self Attention qkv ColumnParallelLinear
        suffixed_col_qkv_concat = [
            f"transformer_blocks.{index}.attn1.proj_qkv.weight",
        ]
        # Cross Attention kv ColumnParallelLinear
        suffixed_col_kv_concat = [
            f"transformer_blocks.{index}.attn2.proj_kv.weight",
        ]

        for name in suffixed_common_col:
            parameters = [state_dicts[tp_rank][name] for tp_rank in range(tp_size)]
            parameters = torch.cat(parameters, dim=0)
            merged_state_dict[name] = parameters
        for name in suffixed_common_row:
            parameters = [state_dicts[tp_rank][name] for tp_rank in range(tp_size)]
            parameters = torch.cat(parameters, dim=1)
            merged_state_dict[name] = parameters
        for name in suffixed_col_qkv_concat:
            wq = [torch.chunk(state_dicts[tp_rank][name], 3, dim=0)[0] for tp_rank in range(tp_size)]
            wk = [torch.chunk(state_dicts[tp_rank][name], 3, dim=0)[1] for tp_rank in range(tp_size)]
            wv = [torch.chunk(state_dicts[tp_rank][name], 3, dim=0)[2] for tp_rank in range(tp_size)]
            wq = torch.cat(wq, dim=0)
            wk = torch.cat(wk, dim=0)
            wv = torch.cat(wv, dim=0)
            wqkv = torch.cat([wq, wk, wv], dim=0)
            merged_state_dict[name] = wqkv
        for name in suffixed_col_kv_concat:
            wk = [torch.chunk(state_dicts[tp_rank][name], 2, dim=0)[0] for tp_rank in range(tp_size)]
            wv = [torch.chunk(state_dicts[tp_rank][name], 2, dim=0)[1] for tp_rank in range(tp_size)]
            wk = torch.cat(wk, dim=0)
            wv = torch.cat(wv, dim=0)
            wkv = torch.cat([wk, wv], dim=0)
            merged_state_dict[name] = wkv

    return merged_state_dict


def merge_by_tp_pp(train_save_dir: str, save_path: str, tp_size: int, pp_sizes: list):
    flags = os.O_RDONLY
    mode = stat.S_IRUSR
    with os.fdopen(os.open(os.path.join(train_save_dir, "latest_checkpointed_iteration.txt"), flags, mode)) as f:
        latest_checkpointed_iteration = f.readline()

    if latest_checkpointed_iteration == 'release':
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(int(latest_checkpointed_iteration))

    _pp_state_dicts = []
    for pp_rank, _ in enumerate(pp_sizes):
        _tp_state_dicts = []
        for tp_rank in range(tp_size):
            if len(pp_sizes) > 1:
                state_dict_path = os.path.join(train_save_dir, directory, f"mp_rank_{tp_rank:02d}_{pp_rank:03d}", "model_optim_rng.pt")
            else:
                state_dict_path = os.path.join(train_save_dir, directory, f"mp_rank_{tp_rank:02d}", "model_optim_rng.pt")
            _tp_state_dicts.append(torch.load(state_dict_path, map_location=torch.device("cpu"))['model'])
        is_last_pp_stage = pp_rank == len(pp_sizes) - 1
        merged_tp_state_dict = merge_by_tp(_tp_state_dicts, num_layers=pp_sizes[pp_rank], tp_size=tp_size, is_last_pp_stage=is_last_pp_stage)
        _pp_state_dicts.append(merged_tp_state_dict)

    save_by_tp_pp(_pp_state_dicts[0], save_path, len(pp_sizes) > 1)
    return 


def save_by_tp_pp(state_dicts: Dict[tuple, Dict], save_dir: str, enable_pp: bool, latest_checkpointed_iteration='release'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    flags = os.O_WRONLY | os.O_CREAT
    mode = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(os.path.join(save_dir, 'latest_checkpointed_iteration.txt'), flags, mode), 'w') as fout:
        fout.write(latest_checkpointed_iteration)
    if latest_checkpointed_iteration == 'release':
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(latest_checkpointed_iteration)

    for (pp_rank, tp_rank), state_dict in state_dicts.items():
        if enable_pp:
            os.makedirs(os.path.join(save_dir, directory, f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"))
            save_path = os.path.join(save_dir, directory, f"mp_rank_{tp_rank:02d}_{pp_rank:03d}", "model_optim_rng.pt")
        else:
            os.makedirs(os.path.join(save_dir, directory, f"mp_rank_{tp_rank:02d}"))
            save_path = os.path.join(save_dir, directory, f"mp_rank_{tp_rank:02d}", "model_optim_rng.pt")
        save_dict = {}
        save_dict['model'] = state_dict
        torch.save(save_dict, save_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp_size", type=int, default=2, help="Tensor model parallel world size")
    parser.add_argument("--pp_sizes", type=int, nargs='+', help="Pipeline parallel model split sizes")
    parser.add_argument("--source_path", type=str, default="./transformers/mp_rank_00/model_states.pt", help="Source path of checkpoint")
    parser.add_argument("--target_path", type=str, default="./ckpt/stepvideo/", help="Save path of MM checkpoint")
    parser.add_argument("--num_layers", type=int, default=48, help="Layer numbers of video_dit")
    parser.add_argument("--num_heads", type=int, default=48, help="Head numbers of video_dit")
    parser.add_argument("--mode", type=str, default="split", choices=["split", "merge"], 
        help="Split mode is used to split the pretrained weights according to tp_size before training, \
        and Merge mode is used to merge weights based on tp_size after training is completed")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    if len(args.pp_sizes) > 1:
        raise ValueError("Pipeline parallel for StepVideo is coming soon!")

    if args.mode == "split":
        source_state_dict = load_from_hf(args.source_path)
        
        # inplace state dict
        for i in range(args.num_layers):
            DIT_CONVERT_MAPPING.update(get_layer_mapping(i))
        update_state_dict_inplace(source_state_dict, DIT_CONVERT_MAPPING)

        state_dicts = split_by_tp(source_state_dict, tp_size=args.tp_size, num_layers=args.num_layers, num_heads=args.num_heads)
        save_by_tp_pp(state_dicts, args.target_path, enable_pp=len(args.pp_sizes) > 1)

    elif args.mode == "merge":
        merge_by_tp_pp(args.source_path, args.target_path, tp_size=args.tp_size, pp_sizes=args.pp_sizes)