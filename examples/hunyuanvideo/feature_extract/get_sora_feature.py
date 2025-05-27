import copy
import json
import os
import time

import mindspeed.megatron_adaptor
import numpy as np
import PIL.Image
import torch
import torch.distributed
from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.training.initialize import initialize_megatron, set_jit_fusion_options
from numpy import save

from mindspeed_mm.configs.config import merge_mm_args, mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.constants import (
    FILE_INFO,
    PROMPT_IDS,
    PROMPT_MASK,
    VIDEO,
    VIDEO_MASK,
)
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.text_encoder import TextEncoder
from mindspeed_mm.tools.profiler import Profiler
from mindspeed_mm.utils.utils import get_device, get_dtype, is_npu_available

if is_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False


def prepare_model(args, extract_video_feature, extract_text_feature, device):
    if extract_video_feature:
        vae = AEModel(args.mm.model.ae).to(device, args.mm.model.ae.dtype).eval()
    else:
        vae = None
    if extract_text_feature:
        text_encoder = TextEncoder(args.mm.model.text_encoder).to(device).eval()
    else:
        text_encoder = None
    return vae, text_encoder


def get_pt_name(file_name):
    pt_name = os.path.basename(file_name).replace(".", "_") + ".pt"
    return pt_name


def extract_feature():
    
    initialize_megatron(extra_args_provider=mm_extra_args_provider, args_defaults={})
    args = get_args()
    merge_mm_args(args)

    extract_video_feature = args.mm.tool.sorafeature.extract_video_feature
    extract_text_feature = args.mm.tool.sorafeature.extract_text_feature
    data_storage_mode = args.mm.tool.sorafeature.data_storage_mode
    
    save_path = args.mm.tool.sorafeature.save_path
    
    if torch.distributed.get_rank() == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if data_storage_mode == 'standard':
            if not os.path.exists(os.path.join(save_path, 'videos')):
                os.makedirs(os.path.join(save_path, 'videos'))
            if not os.path.exists(os.path.join(save_path, 'labels')):
                os.makedirs(os.path.join(save_path, 'labels'))
            if not os.path.exists(os.path.join(save_path, 'images')):
                os.makedirs(os.path.join(save_path, 'images'))
        elif data_storage_mode == "sorafeatured":
            if not os.path.exists(os.path.join(save_path, 'features')):
                os.makedirs(os.path.join(save_path, 'features'))
        else:
            raise NotImplementedError(f"Data storage mode {data_storage_mode} is not implemented! ")

    
    set_jit_fusion_options()
    torch.set_grad_enabled(False)
    dtype = get_dtype(args.mm.model.ae.dtype)
    device = get_device("npu")

    train_dataset = build_mm_dataset(args.mm.data.dataset_param)
    train_dataloader = build_mm_dataloader(
        train_dataset,
        args.mm.data.dataloader_param,
        process_group=mpu.get_data_parallel_group(),
        dataset_param=args.mm.data.dataset_param,
    )

    # master rank, write data info jsonl
    if torch.distributed.get_rank() == 0:
        with open(os.path.join(save_path, 'data.jsonl'), 'w', encoding="utf-8") as json_file:
            for data_sample in train_dataset.data_samples:
                source_data_storage_mode = args.mm.data.dataset_param.basic_parameters.data_storage_mode
                if source_data_storage_mode == "combine":
                    source_file_key = "path"
                elif source_data_storage_mode == "standard":
                    source_file_key = FILE_INFO
                else: 
                    raise NotImplementedError(f"Extract features from data storage mode {source_data_storage_mode} is not implemented")
                
                file_name = data_sample[source_file_key]
                pt_name = get_pt_name(file_name)
                data_info = copy.deepcopy(data_sample)
                if data_storage_mode == "standard":
                    data_info.update({
                        "file": os.path.join('videos', pt_name),
                        "captions": os.path.join('labels', pt_name),
                        "image_latent": os.path.join('images', pt_name)
                    })
                elif data_storage_mode == "sorafeatured":
                    data_info.update({
                        'path': f"features/{pt_name}"
                    })
                json_file.write(json.dumps(data_info) + '\n')
    
    vae, text_encoder = prepare_model(args, extract_video_feature, extract_text_feature, device)
    
    start_time = time.time()
    print_rank_0(f"Features extraction begins. {len(train_dataloader)} data in total.")
    counter = 0

    if hasattr(args.mm.tool, "profile"):
        prof = Profiler(args.mm.tool.profile)
        prof.start()

    for batch in train_dataloader:
        if batch:      
            video = batch.pop(VIDEO).to(device, dtype)
            prompt_ids = batch.pop(PROMPT_IDS)
            prompt_mask = batch.pop(PROMPT_MASK)
            file_names = batch.pop(FILE_INFO)
        else:
            raise ValueError("Batch is None!")

        bs = video.shape[0]
        counter += bs
        kwargs = {}
        
        if extract_video_feature:
            latents, latents_dict = vae.encode(video, **kwargs)
            if latents_dict is not None:
                kwargs.update(latents_dict)
        else:
            latents = video
        
        if extract_text_feature:
            prompt, prompt_mask = text_encoder.encode(prompt_ids, prompt_mask, **kwargs)
        else:
            prompt = prompt_ids
        
        if data_storage_mode == 'standard':
            for i in range(bs):
                pt_name = get_pt_name(file_names[i])
                latent = latents[i].cpu()
                torch.save(latent, os.path.join(save_path, 'videos', pt_name))
                if isinstance(prompt, list) or isinstance(prompt, tuple):
                    prompt = [_prompt[i].cpu() for _prompt in prompt]
                else:
                    prompt = prompt[i].cpu()
                torch.save(prompt, os.path.join(save_path, "labels", pt_name))
                data_to_save = {
                    "file": os.path.join('videos', pt_name),
                    "captions": os.path.join('labels', pt_name)
                }
                if latents_dict is not None:
                    for k in latents_dict:
                        latents_dict[k] = latents_dict[k][i]
                    torch.save(latents_dict, os.path.join(save_path, 'images', pt_name))
            print_rank_0(f"consumed sample {counter} | elapsed time {(time.time() - start_time):.2f} | file {pt_name}")
        
        elif data_storage_mode == 'sorafeatured':
            for i in range(bs):
                latent_i = latents[i].cpu()
                if isinstance(prompt_ids, list) or isinstance(prompt_ids, tuple):
                    prompts_i = [_prompt[i].cpu() for _prompt in prompt]
                    prompt_masks_i = [_prompt_mask[i].cpu() for _prompt_mask in prompt_mask]
                else:
                    prompts_i = prompt[i].cpu()
                    prompt_masks_i = prompt_mask[i]
                
                data_to_save = {
                    "latents": latent_i,
                    "prompt": prompts_i,
                    "prompt_mask": prompt_masks_i
                }
                if latents_dict:
                    for key in latents_dict.keys():
                        if isinstance(latents_dict[key][i], torch.Tensor):
                            data_to_save[key] = latents_dict[key][i].cpu()
                        elif isinstance(latents_dict[key][i], PIL.Image.Image):
                            data_to_save[key] = np.array(latents_dict[key][i])
                        else:
                            data_to_save[key] = latents_dict[key][i]
                pt_name = get_pt_name(file_names[i])
                torch.save(data_to_save, os.path.join(save_path, "features", pt_name))
            print_rank_0(f"consumed sample {counter} | elapsed time {(time.time() - start_time):.2f} | file {pt_name}")
        
        if hasattr(args.mm.tool, "profile"):
            prof.step()
    
    if hasattr(args.mm.tool, "profile"):
        prof.stop()

    duration = time.time() - start_time
    print_rank_0(f"{counter} feature vectors extracted in {duration:.2f} seconds.")


if __name__ == "__main__":
    extract_feature()
