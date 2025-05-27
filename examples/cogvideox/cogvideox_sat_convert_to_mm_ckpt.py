import os
import stat
import copy
import argparse
from typing import Any, Dict, List
import torch


CONVERT_MAPPING = {
    "time_embed.0.bias": "time_embed.time_embed.0.bias",
    "time_embed.0.weight": "time_embed.time_embed.0.weight",
    "time_embed.2.bias": "time_embed.time_embed.2.bias",
    "time_embed.2.weight": "time_embed.time_embed.2.weight",
    "mixins.patch_embed.proj.bias": "patch_embed.proj.bias",
    "mixins.patch_embed.proj.weight": "patch_embed.proj.weight",
    "mixins.patch_embed.text_proj.bias": "caption_projection.bias",
    "mixins.patch_embed.text_proj.weight": "caption_projection.weight",
    "mixins.pos_embed.freqs_cos": "pos_embed.freqs_cos",
    "mixins.pos_embed.freqs_sin": "pos_embed.freqs_sin",
    "transformer.final_layernorm.weight": "norm_final.weight",
    "transformer.final_layernorm.bias": "norm_final.bias",
    "mixins.final_layer.norm_final.weight": "norm_out.weight",
    "mixins.final_layer.norm_final.bias": "norm_out.bias",
    "mixins.final_layer.linear.weight": "proj_out_linear.weight",
    "mixins.final_layer.linear.bias": "proj_out_linear.bias",
    "mixins.final_layer.adaLN_modulation.1.weight": "adaLN_modulation.1.weight",
    "mixins.final_layer.adaLN_modulation.1.bias": "adaLN_modulation.1.bias"
}

first_pipeline_stage_keys = ["time_embed.time_embed.0.bias", "time_embed.time_embed.0.weight",
                    "time_embed.time_embed.2.bias", "time_embed.time_embed.2.weight",
                    "patch_embed.proj.bias", "patch_embed.proj.weight",
                    "caption_projection.bias", "caption_projection.weight"]

last_pipeline_stage_keys = ["norm_final.weight", "norm_final.bias",
                            "norm_out.weight", "norm_out.bias",
                            "proj_out_linear.weight", "proj_out_linear.bias",
                            "adaLN_modulation.1.weight", "adaLN_modulation.1.bias"]


def update_state_dict_inplace(
    state_dict: Dict[str, Any],
    convert_mapping: Dict,
    prefix: str = "model.diffusion_model."
):
    for old_key in convert_mapping:
        new_key = convert_mapping[old_key]
        if prefix + old_key in state_dict:
            state_dict[new_key] = state_dict.pop(prefix + old_key)
        else:
            print(f"Warning: missing update key {prefix + old_key}")


def get_layer_mapping(i: int) -> Dict:
    layer_mapping = {}
    layer_mapping[f"mixins.adaln_layer.adaLN_modulations.{i}.1.bias"] = f"videodit_blocks.{i}.scale_shift_table.1.bias"
    layer_mapping[f"mixins.adaln_layer.adaLN_modulations.{i}.1.weight"] = f"videodit_blocks.{i}.scale_shift_table.1.weight"
    layer_mapping[f"mixins.adaln_layer.query_layernorm_list.{i}.bias"] = f"videodit_blocks.{i}.self_atten.q_norm.bias"
    layer_mapping[f"mixins.adaln_layer.query_layernorm_list.{i}.weight"] = f"videodit_blocks.{i}.self_atten.q_norm.weight"
    layer_mapping[f"mixins.adaln_layer.key_layernorm_list.{i}.bias"] = f"videodit_blocks.{i}.self_atten.k_norm.bias"
    layer_mapping[f"mixins.adaln_layer.key_layernorm_list.{i}.weight"] = f"videodit_blocks.{i}.self_atten.k_norm.weight"
    layer_mapping[f"transformer.layers.{i}.input_layernorm.bias"] = f"videodit_blocks.{i}.norm1.bias"
    layer_mapping[f"transformer.layers.{i}.input_layernorm.weight"] = f"videodit_blocks.{i}.norm1.weight"
    layer_mapping[f"transformer.layers.{i}.attention.dense.bias"] = f"videodit_blocks.{i}.self_atten.proj_out.bias"
    layer_mapping[f"transformer.layers.{i}.attention.dense.weight"] = f"videodit_blocks.{i}.self_atten.proj_out.weight"
    layer_mapping[f"transformer.layers.{i}.post_attention_layernorm.bias"] = f"videodit_blocks.{i}.norm2.bias"
    layer_mapping[f"transformer.layers.{i}.post_attention_layernorm.weight"] = f"videodit_blocks.{i}.norm2.weight"
    layer_mapping[f"transformer.layers.{i}.mlp.dense_h_to_4h.bias"] = f"videodit_blocks.{i}.ff.net.0.proj.bias"
    layer_mapping[f"transformer.layers.{i}.mlp.dense_h_to_4h.weight"] = f"videodit_blocks.{i}.ff.net.0.proj.weight"
    layer_mapping[f"transformer.layers.{i}.mlp.dense_4h_to_h.bias"] = f"videodit_blocks.{i}.ff.net.2.bias"
    layer_mapping[f"transformer.layers.{i}.mlp.dense_4h_to_h.weight"] = f"videodit_blocks.{i}.ff.net.2.weight"
    layer_mapping[f"transformer.layers.{i}.attention.query_key_value.weight"] = f"videodit_blocks.{i}.self_atten.proj_qkv.weight"
    layer_mapping[f"transformer.layers.{i}.attention.query_key_value.bias"] = f"videodit_blocks.{i}.self_atten.proj_qkv.bias"
    return layer_mapping


def remove_layers(
    state_dict: Dict[str, Any],
    remove_keys: List
):
    for remove_key in remove_keys:
        if remove_key in state_dict:
            state_dict.pop(remove_key)
        else:
            print(f"Warning: missing remove key {remove_key}")


def split_by_tp(state_dict: Dict[str, Any], tp_size: int = 2, num_layers: int = 42) -> List[Dict]:
    if tp_size <= 1:
        return [state_dict, ]
    
    new_state_dicts = []
    for tp_rank in range(tp_size):
        new_state_dict = copy.deepcopy(state_dict)

        for index in range(num_layers):
            # ColumnParallelLinear
            suffixed_0 = [
                f"videodit_blocks.{index}.ff.net.0.proj.weight",
                f"videodit_blocks.{index}.ff.net.0.proj.bias",
                f"videodit_blocks.{index}.scale_shift_table.1.weight",
                f"videodit_blocks.{index}.scale_shift_table.1.bias"
            ]
            # RowParallelLinear
            suffixed_1 = [
                f"videodit_blocks.{index}.self_atten.proj_out.weight",
                f"videodit_blocks.{index}.ff.net.2.weight",
            ]
            # self_atten.proj_qkv
            suffixed_special = [
                f"videodit_blocks.{index}.self_atten.proj_qkv.weight",
                f"videodit_blocks.{index}.self_atten.proj_qkv.bias"
            ]

            for split_name in suffixed_0:
                new_state_dict[split_name] = torch.chunk(state_dict[split_name], tp_size, dim=0)[tp_rank]
            for split_name in suffixed_1:
                new_state_dict[split_name] = torch.chunk(state_dict[split_name], tp_size, dim=1)[tp_rank]
            for split_name in suffixed_special:
                wq, wk, wv = torch.chunk(state_dict[split_name], 3, dim=0)
                wq = torch.chunk(wq, tp_size, dim=0)[tp_rank]
                wk = torch.chunk(wk, tp_size, dim=0)[tp_rank]
                wv = torch.chunk(wv, tp_size, dim=0)[tp_rank]
                weight = torch.cat([wq, wk, wv], dim=0)
                new_state_dict[split_name] = weight
        # adaLN modulation
        col_split_names = [
            "adaLN_modulation.1.weight",
            "adaLN_modulation.1.bias",
        ]
        for split_name in col_split_names:
            new_state_dict[split_name] = torch.chunk(state_dict[split_name], tp_size, dim=0)[tp_rank]
        new_state_dicts.append(new_state_dict)
    
    return new_state_dicts


def split_by_pp(state_dicts: List[Dict[str, Any]], pp_sizes: List, remove_pos_emb: bool = False) -> Dict[tuple, Dict]:
    if len(pp_sizes) == 1:
        new_state_dicts = {}
        for tp_rank, state_dict in enumerate(state_dicts):
            new_state_dicts[(0, tp_rank)] = state_dict
        return new_state_dicts

    new_state_dicts = {}
    for pp_rank, _ in enumerate(pp_sizes):
        start_layer_index, end_layer_index = sum(pp_sizes[:pp_rank]), sum(pp_sizes[:pp_rank + 1])
        is_pipeline_first_stage = pp_rank == 0
        is_pipeline_last_stage = pp_rank == len(pp_sizes) - 1

        for tp_rank, state_dict in enumerate(state_dicts):
            pp_tp_param = dict()

            for i in range(start_layer_index, end_layer_index):
                layer_names = get_layer_mapping(i).values()
                pp_layer_names = get_layer_mapping(i - start_layer_index).values()

                for pp_layer_name, layer_name in zip(pp_layer_names, layer_names):
                    if layer_name in state_dict:
                        pp_tp_param[pp_layer_name] = state_dict[layer_name]
                    else:
                        print(f"Warning: missing param key {layer_name}")
                
            if is_pipeline_first_stage:
                for layer_name in first_pipeline_stage_keys:
                    if layer_name in state_dict:
                        pp_tp_param[layer_name] = state_dict[layer_name]
                    else:
                        print(f"Warning: missing pp first stage key {layer_name}")
            if is_pipeline_last_stage:
                for layer_name in last_pipeline_stage_keys:
                    if layer_name in state_dict:
                        pp_tp_param[layer_name] = state_dict[layer_name]
                    else:
                        print(f"Warning: missing pp last stage key {layer_name}")
            new_state_dicts[(pp_rank, tp_rank)] = pp_tp_param

    return new_state_dicts


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


def merge_by_tp(state_dicts: Dict[str, Any], num_layers: int, tp_size: int, is_last_pp_stage: bool):
    if tp_size == 1:
        return state_dicts[0]

    merged_state_dict = copy.deepcopy(state_dicts[0])
    for index in range(num_layers):
        # ColumnParallelLinear
        suffixed_0 = [
            f"videodit_blocks.{index}.ff.net.0.proj.weight",
            f"videodit_blocks.{index}.ff.net.0.proj.bias",
            f"videodit_blocks.{index}.scale_shift_table.1.weight",
            f"videodit_blocks.{index}.scale_shift_table.1.bias"
        ]
        # RowParallelLinear
        suffixed_1 = [
            f"videodit_blocks.{index}.self_atten.proj_out.weight",
            f"videodit_blocks.{index}.ff.net.2.weight",
        ]
        # self_atten.proj_qkv
        suffixed_special = [
            f"videodit_blocks.{index}.self_atten.proj_qkv.weight",
            f"videodit_blocks.{index}.self_atten.proj_qkv.bias"
        ]
        for name in suffixed_0:
            parameters = [state_dicts[tp_rank][name] for tp_rank in range(tp_size)]
            parameters = torch.cat(parameters, dim=0)
            merged_state_dict[name] = parameters
        for name in suffixed_1:
            parameters = [state_dicts[tp_rank][name] for tp_rank in range(tp_size)]
            parameters = torch.cat(parameters, dim=1)
            merged_state_dict[name] = parameters
        for name in suffixed_special:
            wq = [torch.chunk(state_dicts[tp_rank][name], 3, dim=0)[0] for tp_rank in range(tp_size)]
            wk = [torch.chunk(state_dicts[tp_rank][name], 3, dim=0)[1] for tp_rank in range(tp_size)]
            wv = [torch.chunk(state_dicts[tp_rank][name], 3, dim=0)[2] for tp_rank in range(tp_size)]
            wq = torch.cat(wq, dim=0)
            wk = torch.cat(wk, dim=0)
            wv = torch.cat(wv, dim=0)
            wqkv = torch.cat([wq, wk, wv], dim=0)
            merged_state_dict[name] = wqkv

        if is_last_pp_stage:
            # adaLN modulation
            col_split_names = [
                "adaLN_modulation.1.weight",
                "adaLN_modulation.1.bias",
            ]
            for split_name in col_split_names:
                merged_state_dict[split_name] = torch.cat([state_dicts[tp_rank][split_name] for tp_rank in range(tp_size)])
    return merged_state_dict


def merge_by_pp(state_dicts: Dict[str, Any], pp_sizes: list):
    if len(pp_sizes) == 1:
        return state_dicts[0]

    merged_state_dict = {}
    for key in first_pipeline_stage_keys:
        if key in state_dicts[0]:
            merged_state_dict[key] = state_dicts[0][key]
        else:
            print(f"Warning: missing pp first stage key {key}")
    for i, pp_size in enumerate(pp_sizes):
        for layer_index in range(pp_size):
            pp_layer_names = get_layer_mapping(layer_index).values()
            layer_names = get_layer_mapping(layer_index + sum(pp_sizes[:i])).values()
            for pp_layer_name, layer_name in zip(pp_layer_names, layer_names):
                if pp_layer_name in state_dicts[i]:
                    merged_state_dict[layer_name] = state_dicts[i][pp_layer_name]
                else:
                    print(f"Warning: missing pp layer key {pp_layer_name}")
    for key in last_pipeline_stage_keys:
        if key in state_dicts[-1]:
            merged_state_dict[key] = state_dicts[-1][key]
        else:
            print(f"Warning: missing pp last stage key {key}")
    return merged_state_dict


def merge_by_tp_pp(train_save_dir: str, save_path: str, tp_size: int, pp_sizes: list):
    flags = os.O_RDONLY
    mode = stat.S_IRUSR
    with os.fdopen(os.open(os.path.join(train_save_dir, "latest_checkpointed_iteration.txt"), flags, mode)) as f:
        latest_checkpointed_iteration = f.readline()

    if latest_checkpointed_iteration == 'release':
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(latest_checkpointed_iteration)    

    _pp_state_dicts = []
    for pp_rank, _ in enumerate(pp_sizes):
        _tp_state_dicts = []
        for tp_rank in range(tp_size):
            if len(pp_sizes) > 1:
                state_dict_path = os.path.join(train_save_dir, directory, f"mp_rank_{tp_rank:02d}_{pp_rank:03d}", "model_optim_rng.pt")
            else:
                state_dict_path = os.path.join(train_save_dir, directory, f"mp_rank_{tp_rank:02d}", "model_optim_rng.pt")
            _tp_state_dicts.append(torch.load(state_dict_path)['model'])
        is_last_pp_stage = pp_rank == len(pp_sizes) - 1
        merged_tp_state_dict = merge_by_tp(_tp_state_dicts, num_layers=pp_sizes[pp_rank], tp_size=tp_size, is_last_pp_stage=is_last_pp_stage)
        _pp_state_dicts.append(merged_tp_state_dict)
    merged_state_dict = merge_by_pp(_pp_state_dicts, pp_sizes=pp_sizes)

    torch.save(merged_state_dict, save_path)
    return 


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp_size", type=int, default=2, help="Tensor model parallel world size")
    parser.add_argument("--pp_sizes", type=int, nargs='+', help="Pipeline parallel model split sizes")
    parser.add_argument("--num_layers", type=int, default=42, help="Layer numbers of video_dit")
    parser.add_argument("--source_path", type=str, default="./transformer/1/mp_rank_00_model_states.pt", help="Source path of checkpoint")
    parser.add_argument("--target_path", type=str, default="./ckpt/sat_dit/", help="Save path of MM checkpoint")
    parser.add_argument("--task", type=str, default="t2v", choices=["t2v", "i2v"], help="Task type")
    parser.add_argument("--remove_pos_emb", action="store_true", help="remove_pos_emb")
    parser.add_argument("--mode", type=str, default="split", choices=["split", "merge"], 
        help="Split mode is used to split the pretrained weights according to tp_size before training, \
        and Merge mode is used to merge weights based on tp_size after training is completed")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == 'split':
        source_state_dict = torch.load(args.source_path, map_location='cpu')['module']

        if args.task == "i2v":
            CONVERT_MAPPING.update({"mixins.pos_embed.pos_embedding": "pos_embed.pos_embedding"})
            CONVERT_MAPPING.update({"ofs_embed.0.bias": "ofs_embed.0.bias"})
            CONVERT_MAPPING.update({"ofs_embed.0.weight": "ofs_embed.0.weight"})
            CONVERT_MAPPING.update({"ofs_embed.2.bias": "ofs_embed.2.bias"})
            CONVERT_MAPPING.update({"ofs_embed.2.weight": "ofs_embed.2.weight"})

        # inplace state dict
        for i in range(args.num_layers):
            CONVERT_MAPPING.update(get_layer_mapping(i))
        update_state_dict_inplace(source_state_dict, CONVERT_MAPPING)

        # remove dummy layers
        remove_keys = set(source_state_dict.keys()) - set(CONVERT_MAPPING.values())
        remove_layers(source_state_dict, remove_keys)

        if args.remove_pos_emb:
            remove_layers(source_state_dict, ["pos_embed.freqs_cos", "pos_embed.freqs_sin"])
            if args.task == "i2v":
                remove_layers(source_state_dict, ["pos_embed.pos_embedding"])
        else: 
            first_pipeline_stage_keys.append("pos_embed.freqs_cos")
            first_pipeline_stage_keys.append("pos_embed.freqs_sin")
            if args.task == "i2v":
                first_pipeline_stage_keys.append("pos_embed.pos_embedding")
        
        if sum(args.pp_sizes) != args.num_layers:
            raise ValueError(f"The sum of args.pp_sizes {args.pp_sizes} must be equal to args.num_layers {args.num_layers}")

        state_dicts = split_by_tp(source_state_dict, tp_size=args.tp_size, num_layers=args.num_layers)
        state_dicts = split_by_pp(state_dicts, pp_sizes=args.pp_sizes, remove_pos_emb=args.remove_pos_emb)
        save_by_tp_pp(state_dicts, args.target_path, enable_pp=len(args.pp_sizes) > 1)
    
    elif args.mode == 'merge':
        first_pipeline_stage_keys.append("pos_embed.freqs_cos")
        first_pipeline_stage_keys.append("pos_embed.freqs_sin")
        if args.task == "i2v":
            first_pipeline_stage_keys.append("pos_embed.pos_embedding")
        merge_by_tp_pp(args.source_path, args.target_path, tp_size=args.tp_size, pp_sizes=args.pp_sizes)