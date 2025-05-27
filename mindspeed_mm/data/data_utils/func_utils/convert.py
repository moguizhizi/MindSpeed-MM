# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, unique
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

from transformers import PreTrainedTokenizer, ProcessorMixin, AutoProcessor, AutoConfig, AutoTokenizer

from mindspeed_mm.data.data_utils.func_utils.log import get_logger

logger = get_logger(__file__)

IGNORE_INDEX = -100


def convert_sharegpt(
        example: Dict[str, Any],
        dataset_attr: "DatasetAttr",
        dataset_dir: str,
) -> Dict[str, Any]:
    r"""
    Converts sharegpt format dataset to the standard format.
    """
    tag_mapping = {
        dataset_attr.user_tag: Role.USER.value,
        dataset_attr.assistant_tag: Role.ASSISTANT.value,
        dataset_attr.observation_tag: Role.OBSERVATION.value,
        dataset_attr.function_tag: Role.FUNCTION.value,
        dataset_attr.system_tag: Role.SYSTEM.value,
    }
    odd_tags = (dataset_attr.user_tag, dataset_attr.observation_tag)
    even_tags = (dataset_attr.assistant_tag, dataset_attr.function_tag)
    accept_tags = (odd_tags, even_tags)
    messages = example[dataset_attr.messages]
    if (
            dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][dataset_attr.role_tag] == dataset_attr.system_tag
    ):
        system = messages[0][dataset_attr.content_tag]
        messages = messages[1:]
    else:
        system = example[dataset_attr.system] if dataset_attr.system else ""

    aligned_messages = []
    broken_data = False
    for turn_idx, message in enumerate(messages):
        if message[dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
            logger.warning("Invalid role tag in %s.", messages)
            broken_data = True

        aligned_messages.append(
            {"role": tag_mapping.get(message[dataset_attr.role_tag]),
             "content": message[dataset_attr.content_tag]}
        )

    is_valid_message_count = (not dataset_attr.ranking and len(aligned_messages) % 2 != 0) or \
                             (dataset_attr.ranking and len(aligned_messages) % 2 == 0)
    if is_valid_message_count:
        logger.warning_rank0(f"Invalid message count in {messages}.")
        broken_data = True

    if (
        dataset_attr.ranking
        and isinstance(example[dataset_attr.chosen], dict)
        and isinstance(example[dataset_attr.rejected], dict)
    ):  # pairwise example
        chosen = example[dataset_attr.chosen]
        rejected = example[dataset_attr.rejected]
        chosen_role = chosen.get(dataset_attr.role_tag, None)
        rejected_role = rejected.get(dataset_attr.role_tag, None)
        chosen_content = chosen.get(dataset_attr.content_tag, None)
        rejected_content = rejected.get(dataset_attr.content_tag, None)
        chosen_role_tag = tag_mapping.get(chosen_role, None)
        rejected_role_tag = tag_mapping.get(rejected_role, None)

        if chosen_role is None or rejected_role is None:
            logger.warning_rank0(f"Missing role tag in {[chosen, rejected]}.")
            broken_data = True
        elif chosen_role not in accept_tags[-1] or rejected_role not in accept_tags[-1]:
            logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
            broken_data = True

        if chosen_content is None or rejected_content is None:
            logger.warning_rank0(f"Missing content tag in {[chosen, rejected]}.")
            broken_data = True

        if chosen_role_tag is None or rejected_role_tag is None:
            logger.warning_rank0(f"Invalid role tag(tag_mapping) in {[chosen, rejected]}.")
            broken_data = True

        prompt = aligned_messages
        response = [
            {"role": chosen_role_tag, "content": chosen_content},
            {"role": rejected_role_tag, "content": rejected_content},
        ]
    else:  # normal example
        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]

    if broken_data:
        logger.warning("Skipping this abnormal example.")
        prompt, response = [], []

    convert_images = partial(_convert_images, dataset_dir=dataset_dir)
    convert_videos = partial(_convert_videos, dataset_dir=dataset_dir)
    output = {
        "_prompt": prompt,
        "_response": response,
        "_system": system,
        "_images": convert_images(example[dataset_attr.images]) if dataset_attr.images else None,
        "_videos": convert_videos(example[dataset_attr.videos]) if dataset_attr.videos else None,
    }
    return output


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


def _convert_images(
        images: Sequence["ImageInput"],
        dataset_dir: str,
) -> Optional[List["ImageInput"]]:
    r"""
    Optionally concatenates image path to dataset dir when loading from local disk.
    """
    if len(images) == 0:
        return None

    images = images[:]
    for i, image in enumerate(images):
        if isinstance(image, str) and os.path.isfile(os.path.join(dataset_dir, image)):
            images[i] = os.path.join(dataset_dir, image)

    return images


def _convert_videos(
        videos: List["VideoInput"],
        dataset_dir: str,
) -> Optional[List["VideoInput"]]:
    r"""
    Optionally concatenates video path to dataset dir when loading from local disk.
    """
    if len(videos) == 0:
        return None

    videos = videos[:]
    for i, video in enumerate(videos):
        if isinstance(video, str) and os.path.isfile(os.path.join(dataset_dir, video)):
            videos[i] = os.path.join(dataset_dir, video)

    return videos


@dataclass
class DatasetAttr:
    r"""
    Dataset attributes.
    """

    # basic configs
    ranking: bool = False
    # common columns
    system: Optional[str] = None
    images: Optional[str] = None
    videos: Optional[str] = None
    # sharegpt columns
    messages: Optional[str] = "conversations"
    # sharegpt tags
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"
    # rlhf columns
    chosen: Optional[str] = None
    rejected: Optional[str] = None


@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory to read/write data. Defaults to `~/.cache/huggingface/datasets`(env:HF_DATASETS_CACHE)"},
    )
    template: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which template to use for constructing prompts in training and inference."},
    )
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    cutoff_len: int = field(
        default=1024,
        metadata={
            "help": "The cutoff length of the tokenized inputs in the dataset."},
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    mask_history: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to mask the history and train on the last turn only."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The number of examples in one group in pre-processing."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes, truncate the number of examples for each dataset."},
    )
    tool_format: Optional[str] = field(
        default=None,
        metadata={
            "help": "Tool format to use for constructing function calling examples."},
    )
    val_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the validation dataset."},
    )
    val_max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes, truncate the number of examples for each validation dataset."},
    )
    val_rate: Optional[float] = field(
        default=None,
        metadata={"help": "The proportion of the dataset to be used for validation."},
    )

    def __post_init__(self):
        self.dataset = self.dataset.split(",")


def preprocess_supervised_dataset(
        examples: Dict[str, List[Any]],
        template: "Template",
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning("Dropped invalid example: %s", examples["_prompt"][i] + examples["_response"][i])
            continue
        params = EncodeSupervisedParams(images=examples["_images"][i] or [],
                                        videos=examples["_videos"][i] or [],
                                        template=template,
                                        tokenizer=tokenizer,
                                        processor=processor,
                                        cutoff_len=data_args.cutoff_len,
                                        train_on_prompt=data_args.train_on_prompt,
                                        mask_history=data_args.mask_history
                                        )
        input_ids, labels = _encode_supervised_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            params=params
        )
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs


@dataclass
class EncodeSupervisedParams:
    images: Sequence["ImageInput"]
    videos: Sequence["VideoInput"]
    template: "Template"
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]
    cutoff_len: int
    train_on_prompt: bool
    mask_history: bool


def _encode_supervised_example(
        prompt: Sequence[Dict[str, str]],
        response: Sequence[Dict[str, str]],
        system: Optional[str],
        params: EncodeSupervisedParams
) -> Tuple[List[int], List[int]]:
    messages = params.template.mm_plugin.process_messages(
        prompt + response, params.images, params.videos, params.processor)
    input_ids, labels = params.template.mm_plugin.process_token_ids(
        [], [], params.images, params.videos)
    encoded_pairs = params.template.encode_multiturn(
        params.tokenizer, messages, system)
    total_length = len(input_ids) + (1 if params.template.efficient_eos else 0)
    if params.mask_history:
        encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        if total_length >= params.cutoff_len:
            break

        source_len, target_len = infer_seqlen(
            len(source_ids), len(target_ids), params.cutoff_len - total_length)
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        total_length += source_len + target_len

        if params.train_on_prompt:
            source_label = source_ids
        elif params.template.efficient_eos:
            source_label = [params.tokenizer.eos_token_id] + \
                           [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len

        if params.mask_history and turn_idx != 0:  # train on the last turn only
            target_label = [IGNORE_INDEX] * target_len
        else:
            target_label = target_ids

        if params.mask_history:  # reversed sequences
            input_ids = source_ids + target_ids + input_ids
            labels = source_label + target_label + labels
        else:
            input_ids += source_ids + target_ids
            labels += source_label + target_label

    if params.template.efficient_eos:
        input_ids += [params.tokenizer.eos_token_id]
        labels += [params.tokenizer.eos_token_id]

    return input_ids, labels


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> Tuple[int, int]:
    r"""
    Computes the real sequence length after truncation by the cutoff_len.
    """
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(
            cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


def get_image_seqlen(config: "PretrainedConfig") -> int:
    r"""
    Computes the number of special tokens per image.
    """
    model_type = getattr(config, "model_type", None)
    if model_type == "llava":
        image_seqlen = (config.vision_config.image_size // config.vision_config.patch_size) ** 2
        if getattr(config, "vision_feature_select_strategy", "default") == "full":  # add [CLS] token
            image_seqlen += 1
    elif model_type == "paligemma":
        image_seqlen = config.vision_config.num_image_tokens
    else:
        image_seqlen = -1

    return image_seqlen


def get_patch_size(config: "PretrainedConfig") -> int:
    r"""
    Computes the patch size of the vit.
    """
    patch_size = getattr(config.vision_config, "patch_size", -1)
    return patch_size


def get_vision_feature_select_strategy(config: "PretrainedConfig") -> int:
    r"""
    Get the vision_feature_select_strategy.
    """
    vision_feature_select_strategy = getattr(config, "vision_feature_select_strategy", "default")
    return vision_feature_select_strategy


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""
    Loads pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        split_special_tokens=model_args.split_special_tokens,
        padding_side="right", local_files_only=True
    )

    try:
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, local_files_only=True)
        setattr(processor, "tokenizer", tokenizer)
        setattr(processor, "image_seqlen", get_image_seqlen(config))
        setattr(processor, "image_resolution", model_args.image_resolution)
        setattr(processor, "patch_size", get_patch_size(config))
        setattr(processor, "video_resolution", model_args.video_resolution)
        setattr(processor, "video_fps", model_args.video_fps)
        setattr(processor, "video_maxlen", model_args.video_maxlen)
        setattr(processor, "vision_feature_select_strategy", get_vision_feature_select_strategy(config))
    except Exception as e:
        logger.warning("Processor was not found: %s.", e)
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return {"tokenizer": tokenizer, "processor": processor}


@dataclass
class EncodePairwiseParams:
    images: Sequence["ImageInput"]
    videos: Sequence["VideoInput"]
    template: "Template"
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]
    cutoff_len: int


def _encode_pairwise_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    params: EncodePairwiseParams
) -> Tuple[List[int], List[int], List[int], List[int]]:
    chosen_messages = params.template.mm_plugin.process_messages(prompt + [response[0]], params.images, params.videos, params.processor)
    rejected_messages = params.template.mm_plugin.process_messages(prompt + [response[1]], params.images, params.videos, params.processor)
    prompt_ids, chosen_ids = params.template.encode_oneturn(params.tokenizer, chosen_messages, system)
    _, rejected_ids = params.template.encode_oneturn(params.tokenizer, rejected_messages, system)

    if params.template.efficient_eos:
        chosen_ids += [params.tokenizer.eos_token_id]
        rejected_ids += [params.tokenizer.eos_token_id]

    prompt_ids, _ = params.template.mm_plugin.process_token_ids(prompt_ids, None, params.images, params.videos)
    # consider the response is more important
    source_len, target_len = infer_seqlen(len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), params.cutoff_len)
    prompt_ids = prompt_ids[:source_len]
    chosen_ids = chosen_ids[:target_len]
    rejected_ids = rejected_ids[:target_len]

    chosen_input_ids = prompt_ids + chosen_ids
    chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
    rejected_input_ids = prompt_ids + rejected_ids
    rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
    return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels


def preprocess_pairwise_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        params = EncodePairwiseParams(images=examples["_images"][i] or [],
                                      videos=examples["_videos"][i] or [],
                                      template=template,
                                      tokenizer=tokenizer,
                                      processor=processor,
                                      cutoff_len=data_args.cutoff_len,
                                      )
        chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = _encode_pairwise_example(
                                                                                prompt=examples["_prompt"][i],
                                                                                response=examples["_response"][i],
                                                                                system=examples["_system"][i],
                                                                                params=params
                                                                                )
        model_inputs["chosen_input_ids"].append(chosen_input_ids)
        model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
        model_inputs["chosen_labels"].append(chosen_labels)
        model_inputs["rejected_input_ids"].append(rejected_input_ids)
        model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
        model_inputs["rejected_labels"].append(rejected_labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs
