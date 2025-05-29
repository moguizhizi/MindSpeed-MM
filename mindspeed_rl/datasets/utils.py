# coding=utf-8
# Copyright (c) 2020; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.

import math
import os
import random
from io import BytesIO
from typing import Literal, Optional, List

import av
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from PIL.Image import Image as ImageObject
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizer, ProcessorMixin
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def get_train_valid_test_num_samples(
        train_samples, train_iters, global_batch_size, eval_interval, eval_iters):
    """Train/valid/test num samples."""

    # Number of train/valid/test samples.
    if train_samples:
        train_samples = train_samples
    else:
        train_samples = train_iters * global_batch_size
    eval_iters = (train_iters // eval_interval + 1) * eval_iters
    test_iters = eval_iters

    return (
        train_samples,
        eval_iters * global_batch_size,
        test_iters * global_batch_size,
    )


def build_data_iter(dataloader, dataloader_type):
    # Build iterators.
    dl_type = dataloader_type
    if dl_type not in ['single', 'cyclic', 'external']:
        raise ValueError('dl_type should be one of (single, cyclic, external)')

    def _get_iterator(dataloader_type, dataloader):
        """Return dataset iterator."""
        if dataloader_type == "single":
            return iter(dataloader)
        elif dataloader_type == "cyclic":
            return iter(cyclic_iter(dataloader))
        elif dataloader_type == "external":
            return dataloader
        else:
            raise RuntimeError("unexpected dataloader type")

    if dataloader is not None:
        data_iterator = _get_iterator(dl_type, dataloader)
    else:
        data_iterator = None

    return data_iterator


def get_prompt_index(labels, ignored_label):
    prompt_begin_list = []
    prompt_end_list = []
    in_group = False
    for idx, label in enumerate(labels):
        if label == ignored_label:
            if not in_group:
                prompt_begin_list.append(idx)
                in_group = True
        elif in_group:
            prompt_end_list.append(idx)
            in_group = False

    return prompt_begin_list, prompt_end_list


def _infer_seqlen(source_len: int, target_len: int, cutoff_len: int):
    r"""
    Computes the real sequence length after truncation by the cutoff_len.
    """
    # truncate source
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    # truncate target
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


def _build_index_mappings(
        name,
        data_prefix,
        start_index,
        nb_documents,
        num_samples: int,
        seed,
        full_shuffle_instruction_dataset,
        parallel_state,
        no_shuffle=False
):
    """
    - `shuffle_index` is [num_epoch * len(self.mtf)]
    - `sample_index` is [num_sample, 2] (storing the start and end of the sample). We query the sample via `self.shuffle_index[start:end]`
    """

    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += '_{}_indexmap'.format(name)
    _filename += '_{}ns'.format(num_samples)
    _filename += '_{}s'.format(seed)
    shuffle_idx_filename = _filename + f'_nb{nb_documents}' + '_decoder_packed_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    if (not torch.distributed.is_initialized()) or (torch.distributed.get_rank()
                                                    % torch.cuda.device_count()) == 0:
        if not os.path.isfile(shuffle_idx_filename):

            # iteratively add the entire dataset for every epoch and see if it's enough given current packing strategy
            epoch = 0
            shuffle_idx = []
            while len(shuffle_idx) < num_samples:
                new_document_ids = _build_shuffle_idx(
                    nb_documents=nb_documents,
                    start_index=start_index,
                    np_rng=np_rng,
                    no_shuffle=no_shuffle
                )
                shuffle_idx.extend(new_document_ids.tolist())
                epoch += 1

            if full_shuffle_instruction_dataset:
                random.shuffle(shuffle_idx)

            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        counts = torch.cuda.LongTensor([1])
        torch.distributed.all_reduce(counts, group=parallel_state.get_data_parallel_group())
        torch.distributed.all_reduce(counts, group=parallel_state.get_pipeline_model_parallel_group())
        torch.distributed.all_reduce(counts, group=parallel_state.get_context_parallel_group())

    # Load mappings.
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r+')

    return shuffle_idx


def _build_sequential_idx(nb_documents: int, start_index):
    """Build the range [0, dataset_size)."""
    dtype_ = np.int64
    stop = start_index + nb_documents
    result = np.arange(start=start_index, stop=stop, step=1, dtype=dtype_)
    return result


def _build_shuffle_idx(nb_documents: int, start_index, np_rng, no_shuffle):
    """Build the range [0, dataset_size) and shuffle."""
    result = _build_sequential_idx(nb_documents, start_index)
    # in-place shuffling
    if not no_shuffle:
        np_rng.shuffle(result)
    return result


def get_processor(model_path, **kwargs) -> Optional[ProcessorMixin]:
    try:
        processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    except Exception:
        processor = None
        
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return processor


def process_image(image: ImageObject, max_pixels: int, min_pixels: int) -> ImageObject:
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def get_video_sample_frames(video_stream: "Stream", **kwargs) -> int:
    r"""
    Computes video sample frames according to fps.
    """
    video_fps: float = kwargs.get("video_fps")
    video_maxlen: int = kwargs.get("video_maxlen")
    total_frames = video_stream.frames
    sample_frames = float(video_stream.duration * video_stream.time_base) * video_fps
    sample_frames = min(total_frames, video_maxlen, sample_frames)
    return max(math.floor(sample_frames), 1)


def process_videos(video, **kwargs):
    with av.open(video, "r") as container:
        video_stream = next(stream for stream in container.streams if stream.type == "video")
        total_frames = video_stream.frames
        sample_frames = get_video_sample_frames(video_stream, **kwargs)
        sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
        frames: List["ImageObject"] = []
        container.seek(0)
        for frame_idx, frame in enumerate(container.decode(video_stream)):
            if frame_idx in sample_indices:
                frames.append(process_image(frame.to_image(), 2048 * 2048, 512 * 512))
        return frames


def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors

    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return F.pad(tensors, pad_tuple, "constant", pad_token_id)


def get_rope_index(
    processor: Qwen2_5_VLProcessor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Gets the position ids for Qwen2-VL, it should be generated before sharding the sequence.
    The batch dim has been removed and the input_ids should be a 1D tensor representing a single example.
    https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1546
    """
    spatial_merge_size = processor.image_processor.merge_size
    tokens_per_second = 2
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(3, input_ids.size(0), dtype=input_ids.dtype, device=input_ids.device)  # (3, seqlen)
        image_index, video_index = 0, 0
        input_ids = input_ids[attention_mask == 1]
        image_nums, video_nums = 0, 0
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                second_per_grid_t = 0
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                if second_per_grid_ts is not None:
                    second_per_grid_t = second_per_grid_ts[video_index]
                else:
                    second_per_grid_t = 1.0

                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w)
            t_index = (t_index * second_per_grid_t * tokens_per_second).long().flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(input_ids.device)
        else:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1).expand(3, -1)

    return position_ids


def postprocess_data(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        max_length: int,
        pad_token_id: int,
        left_pad: bool = True,
        truncation: Literal["left", "right", "error"] = "error",
    ):
    """
    Pad or truncate data.
    """
    assert truncation in ["left", "right", "error"]
    seq_length = len(input_ids)
    if seq_length < max_length:
        input_ids = pad_sequence_to_length(
            input_ids, max_seq_len=max_length, pad_token_id=pad_token_id, left_pad=left_pad
        )
        attention_mask = pad_sequence_to_length(
            attention_mask, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad
        )
        position_ids = pad_sequence_to_length(position_ids, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad)
    elif seq_length > max_length:
        if truncation == "left":  # actually, left truncation may not be reasonable
            input_ids = input_ids[..., -max_length:]
            attention_mask = attention_mask[..., -max_length:]
            position_ids = position_ids[..., -max_length:]
        elif truncation == "right":
            input_ids = input_ids[..., :max_length]
            attention_mask = attention_mask[..., :max_length]
            position_ids = position_ids[..., :max_length]
        elif truncation == "error":
            raise NotImplementedError(f"{seq_length} is larger than {max_length}.")
        else:
            raise NotImplementedError(f"Unknown truncation method {truncation}.")

    return input_ids, attention_mask, position_ids