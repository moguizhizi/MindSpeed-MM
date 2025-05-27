import os
from copy import deepcopy
import stat

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from safetensors.torch import load_file as safe_load


def load_weight(weight_path):
    if weight_path.endswith(".safetensors"):
        return safe_load(weight_path)
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


def convert_hg_to_mm(state_dict):
    new_checkpoint = {}
    state_dict = state_dict.get("ema_state_dict", state_dict)

    for key, value in state_dict.items():
        new_key = key.replace("transformer_blocks", "videodit_sparse_blocks")
        new_key = new_key.replace("attn1", "self_atten")
        new_key = new_key.replace("attn2", "cross_atten")
        new_key = new_key.replace("to_q", "proj_q")
        new_key = new_key.replace("to_k", "proj_k")
        new_key = new_key.replace("to_v", "proj_v")
        new_key = new_key.replace("to_out.0", "proj_out")
        new_key = new_key.replace("to_out.1", "dropout")
        new_key = new_key.replace("module.", "")
        new_checkpoint[new_key] = value

    return new_checkpoint


def split_by_tp(state_dicts, tp_size):
    if tp_size == 0:
        return [state_dicts, ]

    copy_dict = deepcopy(state_dicts)
    print("total keys: %s" % len(copy_dict))
    total_params = sum(param.numel() for param in copy_dict.values())
    print(f"Total number of parameters before convert : {total_params / 1e9} B.")

    return_dicts = []
    suffixes_0 = ["atten.proj_q.weight", "atten.proj_q.bias", "atten.proj_k.weight", "atten.proj_k.bias",
                  "atten.proj_v.weight", "atten.proj_v.bias", "ff.net.0.proj.weight", "ff.net.0.proj.bias"]
    suffixes_1 = ["atten.proj_out.weight", "ff.net.2.weight"]

    for tp_rank in range(tp_size):
        new_dict = {}
        for key, value in copy_dict.items():
            if isinstance(value, torch.Tensor):
                if any(key.endswith(suffix) for suffix in suffixes_0):
                    value_copy = deepcopy(value)
                    value_copy = torch.chunk(value_copy, tp_size, dim=0)[tp_rank]
                    new_dict[key] = value_copy
                    del value_copy
                elif any(key.endswith(suffix) for suffix in suffixes_1):
                    value_copy = deepcopy(value)
                    value_copy = torch.chunk(value_copy, tp_size, dim=1)[tp_rank]
                    new_dict[key] = value_copy
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


def split_by_pp_vpp(state_dicts, pp_sizes):
    return_dict = []
    if len(pp_sizes) == 0:
        for tp_rank, state_dict in enumerate(state_dicts):
            return_dict.append((tp_rank, state_dict))
        return return_dict

    enable_vpp = isinstance(pp_sizes[0], list)
    if enable_vpp:
        pp_sizes_flat = [layers for vpp_layer in pp_sizes for layers in vpp_layer]
    else:
        pp_sizes_flat = pp_sizes

    print(f"pp_sizes_flat: {pp_sizes_flat}")

    postprocess_weight_names = ['scale_shift_table', 'proj_out.weight', 'proj_out.bias']
    for pp_rank, layers in enumerate(pp_sizes_flat):
        is_first = pp_rank == 0
        is_last = pp_rank == len(pp_sizes_flat) - 1
        start_layer, end_layer = sum(pp_sizes_flat[:pp_rank]), sum(pp_sizes_flat[:pp_rank + 1])
        for tp_rank, state_dict in enumerate(state_dicts):
            pp_tp_param = dict()
            for k in state_dict.keys():
                if k.startswith("videodit_sparse_blocks"):
                    idx = int(k.split('.')[1])
                    if start_layer <= idx < end_layer:
                        cur_idx, tmps = str(idx - start_layer), k.split('.')
                        new_k = '.'.join(tmps[:1] + [cur_idx] + tmps[2:])
                        pp_tp_param[new_k] = state_dict[k]
                elif k in postprocess_weight_names:
                    # for pp rank -1
                    if is_last:
                        pp_tp_param[k] = state_dict[k]
                else:
                    # for pp rank 0
                    if is_first:
                        pp_tp_param[k] = state_dict[k]
            return_dict.append((tp_rank, pp_tp_param))

    return return_dict


def save_by_pp_vpp_tp(pp_sizes, state_dicts, save_dir, mode="train", latest_checkpointed_iteration="release"):
    if os.path.exists(save_dir):
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

    if len(pp_sizes) > 0:
        enable_vpp = isinstance(pp_sizes[0], list)
    else:
        for tp_rank, state_dict in state_dicts:
            save_dict = {}
            filename = f"mp_rank_{tp_rank:02d}"
            os.makedirs(os.path.join(save_dir, directory, filename))
            save_path = os.path.join(save_dir, directory, filename, "model_optim_rng.pt")
            save_dict["model"] = state_dict
            torch.save(save_dict, save_path)
        return

    if enable_vpp:
        vpp_size = len(pp_sizes)
        pp_size = len(pp_sizes[0])
    else:
        pp_size = len(pp_sizes)

    for pp_rank in range(pp_size):
        tp_rank, state_dict = state_dicts[pp_rank]
        os.makedirs(os.path.join(save_dir, directory, f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"))
        save_path = os.path.join(save_dir, directory, f"mp_rank_{tp_rank:02d}_{pp_rank:03d}", "model_optim_rng.pt")

        save_dict = {}
        if mode == "train":
            if enable_vpp:
                save_dict = {f"model{vpp_rank}": state_dicts[vpp_rank * pp_size + pp_rank][1] for vpp_rank in range(vpp_size)}
                save_dict['checkpoint_version'] = 3.0
            else:
                save_dict["model"] = state_dict
        elif mode == "inference":
            save_dict = state_dict
        else:
            raise ValueError(f"unsupported mode: {mode}")
        torch.save(save_dict, save_path)


def save_by_tp(state_dicts, save_dir, latest_checkpointed_iteration="release", exists_ok=False):
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


def save_vae(_state_dict, save_dir, exists_ok=False):
    if os.path.exists(save_dir):
        if not exists_ok:
            print(f"save dir: {save_dir} exists, please check.")
            return
    else:
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "wfvae_mm.pt")
    torch.save(_state_dict, save_path)


if __name__ == "__main__":
    # 参数配置
    TP_SIZE = 1
    # The layers of each pp_rank. For example, [] means disable pp (pp_size is 1),
    # [8, 8, 8, 8] means pp_size is 4 and allocate 8 layers for each pp stage.
    # [[4, 4, 4, 4], [4, 4, 4, 4]] means vp_size is 2, pp_size is 4 and allocatte 4 layers in each vp stage.
    PP_SIZE = []

    dit_hg_weight_path = "local downloaded open sora plan weight path"
    dit_mm_save_dir = "dir to save dit weights after transfer to MindSpeed-MM"

    vae_hg_weight_path = "local downloaded vae weight path"
    vae_mm_save_dir = "dir to save vae weights after transfer to MindSpeed-MM"

    # 转换dit权重
    dit_state_dict = load_from_hf(dit_hg_weight_path)
    dit_state_dict = convert_hg_to_mm(dit_state_dict)
    dit_state_dicts = split_by_tp(dit_state_dict, TP_SIZE)
    dit_state_dicts = split_by_pp_vpp(dit_state_dicts, PP_SIZE)

    for pp_rank, (tp_rank, state_dict) in enumerate(dit_state_dicts):
        print(f"\n\npp_{pp_rank} tp_{tp_rank}")
        for param_k in state_dict:
            print(f"{param_k}: {state_dict[param_k].shape}")

    save_by_pp_vpp_tp(PP_SIZE, dit_state_dicts, dit_mm_save_dir)

    # 转换VAE权重
    vae_state_dict = load_from_hf(vae_hg_weight_path)
    vae_state_dict = convert_hg_to_mm(vae_state_dict)
    save_vae(vae_state_dict, vae_mm_save_dir, exists_ok=False)
