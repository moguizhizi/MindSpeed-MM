import os
import stat
import copy
import argparse
from typing import Any, Dict, List

import torch
from safetensors.torch import load_file
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration
)


DIT_CONVERT_MAPPING = {
    "txt_in.t_embedder.mlp.0.weight": "txt_in.t_embedder.time_embed.0.weight",
    "txt_in.t_embedder.mlp.0.bias": "txt_in.t_embedder.time_embed.0.bias",
    "txt_in.t_embedder.mlp.2.weight": "txt_in.t_embedder.time_embed.2.weight",
    "txt_in.t_embedder.mlp.2.bias": "txt_in.t_embedder.time_embed.2.bias",
    "time_in.mlp.0.weight": "time_in.time_embed.0.weight",
    "time_in.mlp.0.bias": "time_in.time_embed.0.bias",
    "time_in.mlp.2.weight": "time_in.time_embed.2.weight",
    "time_in.mlp.2.bias": "time_in.time_embed.2.bias",
    "vector_in.in_layer.weight": "vector_in.fc1.weight",
    "vector_in.in_layer.bias": "vector_in.fc1.bias",
    "vector_in.out_layer.weight": "vector_in.fc2.weight",
    "vector_in.out_layer.bias": "vector_in.fc2.bias",
    "guidance_in.mlp.0.weight": "guidance_in.time_embed.0.weight",
    "guidance_in.mlp.0.bias": "guidance_in.time_embed.0.bias",
    "guidance_in.mlp.2.weight": "guidance_in.time_embed.2.weight",
    "guidance_in.mlp.2.bias": "guidance_in.time_embed.2.bias",
    "final_layer.linear.weight": "proj_out.weight",
    "final_layer.linear.bias": "proj_out.bias",
    "final_layer.adaLN_modulation.1.weight": "adaLN_modulation.1.weight",
    "final_layer.adaLN_modulation.1.bias": "adaLN_modulation.1.bias"
}


def preprocess_text_encoder_tokenizer(source_dir, save_dir):
    processor = AutoProcessor.from_pretrained(source_dir)
    model = LlavaForConditionalGeneration.from_pretrained(
        source_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    model.language_model.save_pretrained(save_dir)
    processor.tokenizer.save_pretrained(save_dir)


def replace_state_dict(
        state_dict: Dict[str, Any],
        convert_mapping: Dict,
):
    for ori_key, mm_key in convert_mapping.items():
        state_dict[mm_key] = state_dict.pop(ori_key)
    return state_dict


def split_qkv_mlp_fused_column_linear(
        state_dict: Dict[str, Any],
        single_stream_layers: int,
        hidden_size: int = 3072,
):
    """
    Split qkv_mlp fused linear in single stream blocks into qkv part and mlp part
    """
    for index in range(single_stream_layers):
        weight1 = state_dict.pop(f"single_blocks.{index}.linear1.weight")
        bias1 = state_dict.pop(f"single_blocks.{index}.linear1.bias")
        state_dict[f"single_blocks.{index}.linear1_qkv.weight"] = weight1[:hidden_size * 3]
        state_dict[f"single_blocks.{index}.linear1_mlp.weight"] = weight1[hidden_size * 3:]
        state_dict[f"single_blocks.{index}.linear1_qkv.bias"] = bias1[:hidden_size * 3]
        state_dict[f"single_blocks.{index}.linear1_mlp.bias"] = bias1[hidden_size * 3:]
    return state_dict


def get_tp_split_layer_names(
        double_stream_layers: int = 20,
        single_stream_layers: int = 40,
) -> List[str]:
    column_parallel_linears = [
        "vector_in.fc1.weight",
        "vector_in.fc1.bias",
        "proj_out.weight",
        "proj_out.bias",
        "adaLN_modulation.1.weight",
        "adaLN_modulation.1.bias",
    ]

    row_parallel_linears = [
        "vector_in.fc2.weight"
    ]

    qkv_fused_projs = []
    qkv_mlp_fused_projs = []
    x_mlp_fused_row_parallel_linear = []

    for index in range(double_stream_layers):
        # ColumnParallelLinear
        column_parallel_linears += [
            f"double_blocks.{index}.img_mod.linear.weight",
            f"double_blocks.{index}.img_mod.linear.bias",
            f"double_blocks.{index}.img_mlp.fc1.weight",
            f"double_blocks.{index}.img_mlp.fc1.bias",
            f"double_blocks.{index}.txt_mod.linear.weight",
            f"double_blocks.{index}.txt_mod.linear.bias",
            f"double_blocks.{index}.txt_mlp.fc1.weight",
            f"double_blocks.{index}.txt_mlp.fc1.bias",
        ]

        # RowParallelLinear
        row_parallel_linears += [
            f"double_blocks.{index}.img_attn_proj.weight",
            f"double_blocks.{index}.img_mlp.fc2.weight",
            f"double_blocks.{index}.txt_attn_proj.weight",
            f"double_blocks.{index}.txt_mlp.fc2.weight",
        ]

        # qkv_fused_proj
        qkv_fused_projs += [
            f"double_blocks.{index}.img_attn_qkv.weight",
            f"double_blocks.{index}.img_attn_qkv.bias",
            f"double_blocks.{index}.txt_attn_qkv.weight",
            f"double_blocks.{index}.txt_attn_qkv.bias",
        ]

    for index in range(single_stream_layers):
        # ColumnParallelLinear
        column_parallel_linears += [
            f"single_blocks.{index}.modulation.linear.weight",
            f"single_blocks.{index}.modulation.linear.bias",
            f"single_blocks.{index}.linear1_mlp.weight",
            f"single_blocks.{index}.linear1_mlp.bias"
        ]

        # RowParallelLinear
        x_mlp_fused_row_parallel_linear += [
            f"single_blocks.{index}.linear2.weight"
        ]

        # qkv_fused_proj
        qkv_fused_projs += [
            f"single_blocks.{index}.linear1_qkv.weight",
            f"single_blocks.{index}.linear1_qkv.bias"
        ]

    return (
        column_parallel_linears,
        row_parallel_linears,
        qkv_fused_projs,
        qkv_mlp_fused_projs,
        x_mlp_fused_row_parallel_linear
    )


def split_by_tp(
        state_dict: Dict[str, Any],
        tp_size: int = 2,
        double_stream_layers: int = 20,
        single_stream_layers: int = 40,
        num_heads: int = 24,
        head_dim: int = 128
) -> List[Dict]:
    hidden_size = num_heads * head_dim

    if tp_size <= 1:
        return [state_dict]

    new_state_dicts = []

    (
        column_parallel_linears,
        row_parallel_linears,
        qkv_fused_projs,
        qkv_mlp_fused_projs,
        x_mlp_fused_row_parallel_linear
    ) = get_tp_split_layer_names(
        double_stream_layers=double_stream_layers,
        single_stream_layers=single_stream_layers
    )

    for tp_rank in range(tp_size):
        new_state_dict = copy.deepcopy(state_dict)

        for name in column_parallel_linears:
            new_state_dict[name] = torch.chunk(state_dict[name], tp_size, dim=0)[tp_rank]
        for name in row_parallel_linears:
            new_state_dict[name] = torch.chunk(state_dict[name], tp_size, dim=1)[tp_rank]

        for name in qkv_fused_projs:
            wq, wk, wv = torch.chunk(state_dict[name], 3, dim=0)
            wq = torch.chunk(wq, tp_size, dim=0)[tp_rank]
            wk = torch.chunk(wk, tp_size, dim=0)[tp_rank]
            wv = torch.chunk(wv, tp_size, dim=0)[tp_rank]
            weight = torch.cat([wq, wk, wv], dim=0)
            new_state_dict[name] = weight

        for name in qkv_mlp_fused_projs:
            wqkv = state_dict[name][:hidden_size * 3]
            wmlp = state_dict[name][hidden_size * 3:]
            wq, wk, wv = torch.chunk(wqkv, 3, dim=0)
            wq = torch.chunk(wq, tp_size, dim=0)[tp_rank]
            wk = torch.chunk(wk, tp_size, dim=0)[tp_rank]
            wv = torch.chunk(wv, tp_size, dim=0)[tp_rank]
            wmlp = torch.chunk(wmlp, tp_size, dim=0)[tp_rank]
            weight = torch.cat([wq, wk, wv, wmlp], dim=0)
            new_state_dict[name] = weight

        for name in x_mlp_fused_row_parallel_linear:
            wx = state_dict[name][:, :hidden_size]
            wmlp = state_dict[name][:, hidden_size:]
            wx = torch.chunk(wx, tp_size, dim=1)[tp_rank]
            wmlp = torch.chunk(wmlp, tp_size, dim=1)[tp_rank]
            weight = torch.cat([wx, wmlp], dim=1)
            new_state_dict[name] = weight

        new_state_dicts.append(new_state_dict)

    return new_state_dicts


def merge_by_tp(
        state_dicts: List[Dict[str, Any]],
        tp_size: int = 2,
        double_stream_layers: int = 20,
        single_stream_layers: int = 40,
        num_heads: int = 24,
        head_dim: int = 128
) -> Dict[str, Any]:
    hidden_size = num_heads * head_dim

    if tp_size == 1:
        return state_dicts

    merged_state_dict = copy.deepcopy(state_dicts[0])
    (
        column_parallel_linears,
        row_parallel_linears,
        qkv_fused_projs,
        qkv_mlp_fused_projs,
        x_mlp_fused_row_parallel_linear
    ) = get_tp_split_layer_names(
        double_stream_layers=double_stream_layers,
        single_stream_layers=single_stream_layers
    )

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

    for name in qkv_fused_projs:
        wq = torch.cat(
            [torch.chunk(state_dicts[tp_rank][name], 3, dim=0)[0] for tp_rank in range(tp_size)],
            dim=0
        )
        wk = torch.cat(
            [torch.chunk(state_dicts[tp_rank][name], 3, dim=0)[1] for tp_rank in range(tp_size)],
            dim=0
        )
        wv = torch.cat(
            [torch.chunk(state_dicts[tp_rank][name], 3, dim=0)[2] for tp_rank in range(tp_size)],
            dim=0
        )
        wqkv = torch.cat([wq, wk, wv], dim=0)
        merged_state_dict[name] = wqkv

    for name in qkv_mlp_fused_projs:
        wqkvs = [state_dicts[tp_rank][name][:hidden_size // tp_size * 3] for tp_rank in range(tp_size)]
        wq = torch.cat(
            [torch.chunk(wqkvs[tp_rank], 3, dim=0)[0] for tp_rank in range(tp_size)],
            dim=0
        )
        wk = torch.cat(
            [torch.chunk(wqkvs[tp_rank], 3, dim=0)[1] for tp_rank in range(tp_size)],
            dim=0
        )
        wv = torch.cat(
            [torch.chunk(wqkvs[tp_rank], 3, dim=0)[2] for tp_rank in range(tp_size)],
            dim=0
        )
        wmlp = torch.cat(
            [state_dicts[tp_rank][name][hidden_size // tp_size * 3:] for tp_rank in range(tp_size)],
            dim=0
        )
        merged_state_dict[name] = torch.cat([wq, wk, wv, wmlp], dim=0)

    for name in x_mlp_fused_row_parallel_linear:
        wx = torch.cat(
            [state_dicts[tp_rank][name][:, :hidden_size // tp_size] for tp_rank in range(tp_size)],
            dim=1,
        )
        wmlp = torch.cat(
            [state_dicts[tp_rank][name][:, hidden_size // tp_size:] for tp_rank in range(tp_size)],
            dim=1,
        )
        merged_state_dict[name] = torch.cat([wx, wmlp], dim=1)

    return merged_state_dict


def load_state_dicts_by_tp(load_dir: str, tp_size: int = 2) -> List[Dict[str, Any]]:
    flags = os.O_RDONLY
    mode = stat.S_IRUSR
    with os.fdopen(os.open(os.path.join(load_dir, "latest_checkpointed_iteration.txt"), flags, mode)) as f:
        latest_checkpointed_iteration = f.readline()

    if latest_checkpointed_iteration == 'release':
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(latest_checkpointed_iteration)

    tp_state_dicts = []
    for tp_rank in range(tp_size):
        state_dict_path = os.path.join(load_dir, directory, f"mp_rank_{tp_rank:02d}", "model_optim_rng.pt")
        tp_state_dicts.append(torch.load(state_dict_path)['model'])

    return tp_state_dicts


def save(state_dicts: List[Dict], save_dir: str, latest_checkpointed_iteration="release"):
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

    for tp_rank, state_dict in enumerate(state_dicts):
        os.makedirs(os.path.join(save_dir, directory, f"mp_rank_{tp_rank:02d}"))
        save_path = os.path.join(save_dir, directory, f"mp_rank_{tp_rank:02d}", "model_optim_rng.pt")
        save_dict = {}
        save_dict['model'] = state_dict
        torch.save(save_dict, save_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, choices=["dit", "text_encoder", "lora"], default="dit", help="The module to convert")
    parser.add_argument("--source_path", type=str, default="./transformers/mp_rank_00/model_states.pt", help="Source path of checkpoint")
    parser.add_argument("--target_path", type=str, default="./ckpt/hunyuanvideo/", help="Save path of MM checkpoint")
    parser.add_argument("--tp_size", type=int, default=2, help="Tensor model parallel world size")
    parser.add_argument("--double_stream_layers", type=int, default=20)
    parser.add_argument("--single_stream_layers", type=int, default=40)
    parser.add_argument("--num_heads", type=int, default=24)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--mode", type=str, default="split", choices=["split", "merge"], 
        help="Split mode is used to split the pretrained weights according to tp_size before training, \
        and Merge mode is used to merge weights based on tp_size after training is completed")

    args = parser.parse_args()
    return args


def convert_kohya_to_peft_keys(
    kohya_dict: dict,
    kohya_prefix="",
    peft_prefix: str = "base_model.model",
    device="cpu",
    num_heads: int = 24,
    head_dim: int = 128
) -> dict:
    hidden_size = num_heads * head_dim

    peft_dict = {}
    for k, v in kohya_dict.items():
        if ".alpha" in k:
            continue
        new_key = k.replace(f"{kohya_prefix}_lora_", "")
        new_key = new_key.replace("single_blocks_", "single_blocks.")
        new_key = new_key.replace("double_blocks_", "double_blocks.")
        new_key = new_key.replace("_img_attn_proj", ".img_attn_proj")
        new_key = new_key.replace("_img_attn_qkv", ".img_attn_qkv")
        new_key = new_key.replace("_img_mlp_fc", ".img_mlp.fc")
        new_key = new_key.replace("_txt_mlp_fc", ".txt_mlp.fc")
        new_key = new_key.replace("_img_mod", ".img_mod")
        new_key = new_key.replace("_txt", ".txt")
        new_key = new_key.replace("_modulation", ".modulation")
        new_key = new_key.replace("_linear", ".linear")
        new_key = new_key.replace("lora_down", "lora_A.default")
        new_key = new_key.replace("lora_up", "lora_B.default")
        new_key = new_key.replace(
            "_individual_token_refiner_blocks_", ".individual_token_refiner.blocks."
        )
        new_key = new_key.replace("_mlp_fc", ".mlp.fc")

        new_key = new_key.replace("vector_in_in_layer", "vector_in.fc1")
        new_key = new_key.replace("vector_in_out_layer", "vector_in.fc2")
        new_key = new_key.replace("final_layer.linear", "proj_out")

        if "linear1" in new_key:
            # split linear1 to linear1_qkv and linear1_mlp
            if ".lora_A" in new_key:
                peft_dict[new_key.replace("linear1", "linear1_qkv")] = v.to(device)
                peft_dict[new_key.replace("linear1", "linear1_mlp")] = v.to(device)
            else:
                w_qkv = v[: hidden_size * 3]
                w_mlp = v[hidden_size * 3:]
                peft_dict[new_key.replace("linear1", "linear1_qkv")] = w_qkv.to(device)
                peft_dict[new_key.replace("linear1", "linear1_mlp")] = w_mlp.to(device)
        else:
            peft_dict[new_key] = v.to(device)
    return peft_dict


if __name__ == "__main__":
    args = get_args()

    if args.module == "text_encoder":
        preprocess_text_encoder_tokenizer(args.source_path, args.target_path)
    elif args.module == "lora":
        kohya_weights = load_file(args.source_path)
        state_dict = convert_kohya_to_peft_keys(
            kohya_weights, kohya_prefix="Hunyuan_video_I2V", device="cpu"
        )
        save([state_dict], args.target_path)
    else:
        if args.mode == "split":
            source_state_dict = torch.load(args.source_path, map_location='cpu')['module']
            state_dict = replace_state_dict(source_state_dict, convert_mapping=DIT_CONVERT_MAPPING)
            state_dict = split_qkv_mlp_fused_column_linear(
                state_dict, 
                single_stream_layers=args.single_stream_layers, 
                hidden_size=args.num_heads * args.head_dim
            )
            state_dicts = split_by_tp(
                state_dict,
                tp_size=args.tp_size,
                double_stream_layers=args.double_stream_layers,
                single_stream_layers=args.single_stream_layers,
                num_heads=args.num_heads,
                head_dim=args.head_dim
            )
            save(state_dicts, args.target_path)
        elif args.mode == "merge":
            tp_state_dicts = load_state_dicts_by_tp(args.source_path, tp_size=args.tp_size)
            merged_state_dict = merge_by_tp(
                tp_state_dicts,
                tp_size=args.tp_size,
                double_stream_layers=args.double_stream_layers,
                single_stream_layers=args.single_stream_layers,
                num_heads=args.num_heads,
                head_dim=args.head_dim
            )
            save([merged_state_dict], args.target_path)