import os
from copy import deepcopy
import stat
import argparse
from typing import Any, Dict, List

import torch
import torch_npu

from torch_npu.contrib import transfer_to_npu
from safetensors.torch import load_file


def load_weight(weight_path):
    if weight_path.endswith(".safetensors"):
        return load_file(weight_path)
    else:
        return torch.load(weight_path, map_location="cpu")


def load_from_hf(load_path):
    ckpt_dict = {}
    if os.path.isdir(load_path):
        for filename in os.listdir(load_path):
            file_path = os.path.join(load_path, filename)
            if os.path.isfile(file_path) and os.path.splitext(file_path)[1] in [".pt", ".pth", ".ckpt", ".safetensors"]:
                ckpt_dict_part = load_weight(file_path)
                ckpt_dict.update(ckpt_dict_part)
    elif os.path.isfile(load_path):
        ckpt_dict = load_weight(load_path)
    else:
        raise ValueError(f"Invalid path: {load_path}")
    return ckpt_dict


def convert_hg_to_mm(state_dict: Dict[str, Any],) -> Dict[str, Any]:
    new_checkpoint = {}
    state_dict = state_dict.get("ema_state_dict", state_dict)

    for key, value in state_dict.items():
        model_key = key.replace("transformer_blocks", "videodit_blocks")
        model_key = model_key.replace("attn1", "self_atten")
        model_key = model_key.replace("attn2", "cross_atten")
        model_key = model_key.replace("to_q", "proj_q")
        model_key = model_key.replace("to_k", "proj_k")
        model_key = model_key.replace("to_v", "proj_v")
        model_key = model_key.replace("to_out.0", "proj_out")
        model_key = model_key.replace("to_out.1", "dropout")
        new_checkpoint[model_key] = value

    return new_checkpoint


def split_by_tp(
        state_dicts: Dict[str, Any], 
        tp_size: int
) -> List[Dict]:
    if tp_size == 0 or tp_size == 1:
        return [state_dicts, ]

    copy_dict = deepcopy(state_dicts)
    print("total keys: %s" % len(copy_dict))
    total_params = sum(param.numel() for param in copy_dict.values())
    print(f"Total number of parameters before convert : {total_params / 1e9} B.")

    return_dicts = []
    # column_parallel_linears
    suffixes_0 = ["atten.proj_q.weight", "atten.proj_q.bias", 
                "atten.proj_k.weight", "atten.proj_k.bias",
                "atten.proj_v.weight", "atten.proj_v.bias", 
                "ff.net.0.proj.weight", "ff.net.0.proj.bias"]
    # row_parallel_linears
    suffixes_1 = ["atten.proj_out.weight", "ff.net.2.weight"]

    for tp_rank in range(tp_size):
        new_dict = {}
        for key, value in copy_dict.items():
            if isinstance(value, torch.Tensor):
                if any(key.endswith(suffix) for suffix in suffixes_0):
                    value_copy = deepcopy(value)
                    value_copy = torch.chunk(value_copy, tp_size, dim=0)[tp_rank]
                    new_dict[key] = value_copy.clone()
                    del value_copy
                elif any(key.endswith(suffix) for suffix in suffixes_1):
                    value_copy = deepcopy(value)
                    value_copy = torch.chunk(value_copy, tp_size, dim=1)[tp_rank]
                    new_dict[key] = value_copy.clone()
                    del value_copy
                else:
                    new_dict[key] = value
            else:
                new_dict[key] = value
                print(f"key: {key}, Type:{type(value)}")

        total_params = sum(param.numel() for param in new_dict.values())
        print(f"Total number of parameters after convert on TP rank:{tp_rank} : {total_params / 1e9} B.")
        return_dicts.append(new_dict)
    del copy_dict
    return return_dicts


def save_by_tp(
    state_dicts: Dict[str, Any], 
    save_dir: str, 
    latest_checkpointed_iteration: str = "release",
    exists_ok: bool = False
):
    if os.path.exists(save_dir) and not exists_ok:
        print(f"save dir: {save_dir} exists, please check.")
        return
    else:
        os.makedirs(save_dir)

    flags = os.O_WRONLY | os.O_CREAT
    stat_mode = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(os.path.join(save_dir, 'latest_checkpointed_iteration.txt'), flags, stat_mode), 'w') as fout:
        fout.write(latest_checkpointed_iteration)
    if latest_checkpointed_iteration == 'release':
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(latest_checkpointed_iteration)

    for tp_rank, state_dict in enumerate(state_dicts):
        os.makedirs(os.path.join(save_dir, directory, f"mp_rank_{tp_rank:02d}"))
        save_path = os.path.join(save_dir, directory, f"mp_rank_{tp_rank:02d}", "model_optim_rng.pt")
        save_dict = {}
        save_dict["model"] = state_dict
        torch.save(save_dict, save_path)


def save_vae(
    state_dict: Dict[str, Any], 
    save_dir: str,
    exists_ok: bool = False
):
    if os.path.exists(save_dir):
        if not exists_ok:
            print(f"save dir: {save_dir} exists, please check.")
            return
    else:
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "casualvae_mm.pt")
    torch.save(state_dict, save_path)


def load_state_dicts_by_tp(
    load_dir: str, 
    tp_size: int = 2
) -> List[Dict[str, Any]]:
    flags = os.O_RDONLY
    stat_mode = stat.S_IRUSR
    with os.fdopen(os.open(os.path.join(load_dir, "latest_checkpointed_iteration.txt"), flags, stat_mode)) as f:
        latest_checkpointed_iteration = f.readline()

    if latest_checkpointed_iteration == 'release':
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(latest_checkpointed_iteration)

    tp_state_dicts = []
    for tp_rank in range(tp_size):
        state_dict_path = os.path.join(load_dir, directory, f"mp_rank_{tp_rank:02d}", "model_optim_rng.pt")
        if os.path.isfile(state_dict_path):
            tp_state_dicts.append(torch.load(state_dict_path)['model'])
        else:
            raise ValueError(f"Invalid path: {state_dict_path}")
    return tp_state_dicts


def get_tp_split_layer_names(state_dicts: List[Dict[str, Any]]):
    # column_parallel_linears
    suffixes_0 = ["atten.proj_q.weight", "atten.proj_q.bias", 
                "atten.proj_k.weight", "atten.proj_k.bias",
                "atten.proj_v.weight", "atten.proj_v.bias", 
                "ff.net.0.proj.weight", "ff.net.0.proj.bias"]
    # row_parallel_linears
    suffixes_1 = ["atten.proj_out.weight", "ff.net.2.weight"]

    column_parallel_linears = []
    row_parallel_linears = []
    for key, value in state_dicts.items():
        if isinstance(value, torch.Tensor):
            if any(key.endswith(suffix) for suffix in suffixes_0):
                column_parallel_linears.append(key)
            elif any(key.endswith(suffix) for suffix in suffixes_1):
                row_parallel_linears.append(key)
        else:
            print(f"key:{key}, Type:{type(value)}")
    
    return column_parallel_linears, row_parallel_linears


def merge_by_tp(
    state_dicts: List[Dict[str, Any]], 
    tp_size: int = 1
) -> Dict[str, Any]:
    if tp_size == 0 or tp_size == 1:
        return state_dicts
    
    merged_state_dict = deepcopy(state_dicts[0])
    column_parallel_linears, row_parallel_linears = get_tp_split_layer_names(merged_state_dict)
    
    for name in column_parallel_linears:
        merged_state_dict[name] = torch.cat(
            [state_dicts[tp_rank][name] for tp_rank in range(tp_size)],
            dim=0
            )

    for name in row_parallel_linears:
        merged_state_dict[name] = torch.cat(
            [state_dicts[tp_rank][name] for tp_rank in range(tp_size)],
            dim=1
            )
    
    return merged_state_dict


def get_args():
    parser = argparse.ArgumentParser()
    # 参数配置
    parser.add_argument('--tp-size', type=int, default=1,
                        help='Tensor model parallel world size')

    parser.add_argument('--dit-hg-weight-path', type=str, 
                        required=False, default=None,
                        help='Local downloaded open sora plan weight path')
    parser.add_argument('--dit-mm-save-path', type=str, 
                        required=False, default=None,
                        help='Dir to save dit weights after transfer to MindSpeed-MM')

    parser.add_argument('--vae-convert', type=bool, 
                        required=False, default=True,
                        help='Transfer the weight of vae to MindSpeed-MM with split mode.')
    parser.add_argument('--vae-hg-weight-path', type=str, 
                        required=False, default=None,
                        help='Local downloaded vae weight path')
    parser.add_argument('--vae-mm-save-path', type=str, 
                        required=False, default=None,
                        help='Dir to save vae weights after transfer to MindSpeed-MM')
    
    parser.add_argument('--dit-mm-weight-path', type=str, 
                        required=False, default=None,
                        help='The path of splited open sora plan weights after training is completed')
    parser.add_argument('--dit-merge-save-path', type=str, 
                        required=False, default=None,
                        help='Dir to save the merged dit weights based on tp_size')

    parser.add_argument("--mode", type=str, default="split", choices=["split", "merge"],
        help="Split mode is used to split the pretrained weights according to tp_size before training, \
        and Merge mode is used to merge weights based on tp_size after training is completed")

    args, unrecognized_args = parser.parse_known_args()
    if unrecognized_args:
        print(f"Unrecognized Args: {unrecognized_args}")
    
    return args


if __name__ == "__main__":
    args = get_args()

    if args.mode == "split":

        if args.dit_hg_weight_path is None:
            args.dit_hg_weight_path = "./raw_ckpt/open-sora-plan/93x480p/diffusion_pytorch_model.safetensors"
            print(f"No dit_hg_weight_path, the default path is {args.dit_hg_weight_path}")
        if args.dit_mm_save_path is None:
            args.dit_mm_save_path = "./ckpt/open-sora-plan-12/93x480p"
            print(f"No dit_mm_save_path, the default path is {args.dit_mm_save_path}")

        # 转换dit权重
        dit_state_dict = load_from_hf(args.dit_hg_weight_path)
        dit_state_dict = convert_hg_to_mm(dit_state_dict)
        dit_state_dicts = split_by_tp(dit_state_dict, args.tp_size)

        for tp_rank, state_dict in enumerate(dit_state_dicts):
            print(f"\n\n tp_{tp_rank}")
            for param_k in state_dict:
                print(f"{param_k}: {state_dict[param_k].shape}")

        save_by_tp(dit_state_dicts, args.dit_mm_save_path, exists_ok=False)

        # 转换VAE权重
        if args.vae_convert:

            if args.vae_hg_weight_path is None:
                args.vae_hg_weight_path = "./raw_ckpt/open-sora-plan/vae/checkpoint.ckpt"
                print(f"No vae_hg_weight_path, the default path is {args.vae_hg_weight_path}")
            if args.vae_mm_save_path is None:
                args.vae_mm_save_path = "./ckpt/vae"
                print(f"No vae_mm_save_path, the default path is {args.vae_mm_save_path}")

            vae_state_dict = load_from_hf(args.vae_hg_weight_path)
            vae_state_dict = convert_hg_to_mm(vae_state_dict)
            save_vae(vae_state_dict, args.vae_mm_save_path, exists_ok=False)
    # 合并dit权重
    elif args.mode == "merge":

        if args.dit_mm_weight_path is None:
            args.dit_mm_weight_path = "./ckpt/open-sora-plan-12/93x480p"
            print(f"No dit_mm_weight_path, the default path is {args.dit_mm_weight_path}")
        if args.dit_merge_save_path is None:
            args.dit_merge_save_path = "./ckpt/open-sora-plan-12/merge"
            print(f"No dit_merge_save_path, the default path is {args.dit_merge_save_path}")

        tp_state_dicts = load_state_dicts_by_tp(args.dit_mm_weight_path, tp_size=args.tp_size)
        merged_state_dict = merge_by_tp(tp_state_dicts, tp_size=args.tp_size)
        save_by_tp([merged_state_dict], args.dit_merge_save_path, exists_ok=False)
