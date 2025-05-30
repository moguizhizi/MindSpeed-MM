# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import argparse
import os

import clip
import torch


def convert(download_root, output_path, tensor_parallel_size, use_te_layernorm_linear):
    device = "cpu"

    model, _ = clip.load(download_root, device=device)

    state_dict = model.state_dict()
    new_state_dicts = [{"model": dict()} for _ in range(tensor_parallel_size)]

    # Indices from mapping pytorch multihead attention to megatron.
    kv_channels = 64
    hidden_dim = 1024
    num_heads = 16
    indices = []
    for i in range(num_heads):
        lb = i * kv_channels
        ub = (i + 1) * kv_channels
        indices.append(torch.arange(lb, ub, dtype=torch.int))
        indices.append(torch.arange(hidden_dim + lb, hidden_dim + ub, dtype=torch.int))
        indices.append(torch.arange(2 * hidden_dim + lb, 2 * hidden_dim + ub, dtype=torch.int))

    indices = torch.cat(indices)

    for name, tensor in state_dict.items():
        # Skip text model.
        if "visual" not in name:
            continue

        # Skip final layers not used in our model.
        if name == "visual.proj" or "ln_post" in name:
            continue

        # Map parameter names to ones used in megatron.
        new_name = ""
        new_tensor = tensor
        if new_tensor.dtype == torch.float16:
            new_tensor = new_tensor.to(torch.float32)

        # This is used for chunking some tensors to target tensor parallel size.
        chunk_dim = None

        if "class_embedding" in name:
            new_name = "class_token"
            # Our model uses class token that is expanded to input dimensions already.
            new_tensor = new_tensor.expand(1, 1, -1)
        elif "positional_embedding" in name:
            new_name = "position_embeddings.weight"
        elif "conv1" in name:
            new_name = "conv1.weight"
        elif "ln_pre.weight" in name:
            new_name = "ln_pre.weight"
        elif "ln_pre.bias" in name:
            new_name = "ln_pre.bias"
        elif "transformer.resblocks" in name:
            layer_idx = name.split(".")[3]
            base = f"decoder.layers.{layer_idx}"

            if "attn.in_proj_weight" in name:
                new_name = f"{base}.self_attention.linear_qkv.weight"
                new_tensor = new_tensor[indices]
                chunk_dim = 0
            elif "attn.in_proj_bias" in name:
                new_name = f"{base}.self_attention.linear_qkv.bias"
                new_tensor = new_tensor[indices]
                chunk_dim = 0
            elif "attn.out_proj.weight" in name:
                new_name = f"{base}.self_attention.linear_proj.weight"
                chunk_dim = 1
            elif "attn.out_proj.bias" in name:
                new_name = f"{base}.self_attention.linear_proj.bias"
            elif "ln_1.weight" in name:
                new_name = f"{base}.input_layernorm.weight"
                if use_te_layernorm_linear:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_weight"
            elif "ln_1.bias" in name:
                new_name = f"{base}.input_layernorm.bias"
                if use_te_layernorm_linear:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_bias"
            elif "mlp.c_fc.weight" in name:
                new_name = f"{base}.mlp.linear_fc1.weight"
                chunk_dim = 0
            elif "mlp.c_fc.bias" in name:
                new_name = f"{base}.mlp.linear_fc1.bias"
                chunk_dim = 0
            elif "mlp.c_proj.weight" in name:
                new_name = f"{base}.mlp.linear_fc2.weight"
                chunk_dim = 1
            elif "mlp.c_proj.bias" in name:
                new_name = f"{base}.mlp.linear_fc2.bias"
            elif "ln_2.weight" in name:
                new_name = f"{base}.pre_mlp_layernorm.weight"
                if use_te_layernorm_linear:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_weight"
            elif "ln_2.bias" in name:
                new_name = f"{base}.pre_mlp_layernorm.bias"
                if use_te_layernorm_linear:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_bias"

        if new_name == "":
            raise AssertionError(f"unexpected layer name {name}")

        if chunk_dim is None:
            new_tensors = [new_tensor for _ in range(tensor_parallel_size)]
        else:
            new_tensors = torch.chunk(new_tensor, tensor_parallel_size, dim=chunk_dim)

        for i in range(tensor_parallel_size):
            # chunk() creates a view of a bigger tensor. clone() is used here to avoid excessive storage.
            new_state_dicts[i]["model"][new_name] = new_tensors[i].clone()

    output_path_pt = os.path.join(output_path, "converted_clip.pt")
    torch.save(new_state_dicts[0]["model"], output_path_pt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Convert OpenAI CLIP VIT weights to megatron format.


Example usage:
python examples/llava1.5/clip_converter.py \
    --download-root {dir_to_model}/ViT-L-14-336px.pt \
    --output {target_dir}
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--download-root", type=str, required=True, help="Download folder for OpenAI CLIP weights",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="output directory for megatron state dict file(s)"
    )

    args = parser.parse_args()

    convert(args.download_root, args.output, 1, False)

    print("all weights have been converted.")