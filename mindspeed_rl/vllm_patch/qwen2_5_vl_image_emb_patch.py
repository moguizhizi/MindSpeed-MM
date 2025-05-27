# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import List, Mapping, Optional, Tuple, Union, cast, TYPE_CHECKING
from typing_extensions import assert_never

import vllm
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalEncDecInputs,
                                    MultiModalInputs)
from vllm.inputs.data import (DecoderOnlyInputs, EncoderDecoderInputs, ProcessorInputs,
                   PromptType, SingletonInputs, SingletonPrompt, token_inputs)

if TYPE_CHECKING:
    from vllm.sequence import SequenceGroupMetadata

from vllm.multimodal.inputs import MultiModalKwargs
from vllm.multimodal.base import MultiModalPlaceholderMap
# logger = init_logger(__name__)


def _process_multimodal(
    self,
    prompt: Union[str, List[int]],
    mm_data: MultiModalDataDict,
    mm_processor_kwargs: Optional[Mapping[str, object]],
    lora_request: Optional[LoRARequest],
) -> MultiModalInputs:
    """
    Apply the model's multi-modal processor to a multi-modal prompt,
    returning the corresponding token IDs and metadata.
    """
    # At the moment on model (PrithviGeoSpatialMAE) requires to be
    # initialized without a tokenizer while using also multi-modal
    # input.
    if "image_embeds" in mm_data and "image_grid_thw" in mm_data:
        return token_inputs(
            prompt_token_ids=prompt,
            multi_modal_data=mm_data,
            mm_processor_kwargs=mm_processor_kwargs,
        )

    if not self.tokenizer:
        tokenizer = None
    else:
        tokenizer_group = self.get_tokenizer_group()
        tokenizer = tokenizer_group.get_lora_tokenizer(lora_request)

    mm_processor = self.mm_registry.create_processor(
        self.model_config, tokenizer)

    if mm_processor_kwargs is None:
        mm_processor_kwargs = {}

    return mm_processor.apply(prompt, mm_data, mm_processor_kwargs)

async def _process_multimodal_async(
    self,
    prompt: Union[str, List[int]],
    mm_data: MultiModalDataDict,
    mm_processor_kwargs: Optional[Mapping[str, object]],
    lora_request: Optional[LoRARequest],
) -> MultiModalInputs:
    """Async version of :meth:`_process_multimodal`."""
    # At the moment on model (PrithviGeoSpatialMAE) requires to be
    # initialized without a tokenizer while using also multi-modal
    # input.
    if "image_embeds" in mm_data and "image_grid_thw" in mm_data:
        return token_inputs(
            prompt_token_ids=prompt,
            multi_modal_data=mm_data,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        
    if not self.tokenizer:
        tokenizer = None
    else:
        tokenizer_group = self.get_tokenizer_group()
        tokenizer = await tokenizer_group.get_lora_tokenizer_async(
            lora_request)

    mm_processor = self.mm_registry.create_processor(
        self.model_config, tokenizer)
    if mm_processor_kwargs is None:
        mm_processor_kwargs = {}

    return mm_processor.apply(prompt, mm_data, mm_processor_kwargs)

def _process_multimodal_vllm085(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        mm_processor_kwargs: Optional[Mapping[str, object]],
        lora_request: Optional[LoRARequest],
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:
        """
        Apply the model's multi-modal processor to a multi-modal prompt,
        returning the corresponding token IDs and metadata.
        """
        # At the moment on model (PrithviGeoSpatialMAE) requires to be
        if not self.tokenizer:
            tokenizer = object()  # Dummy
        else:
            tokenizer_group = self.get_tokenizer_group()
            tokenizer = tokenizer_group.get_lora_tokenizer(lora_request)

        if "image_embeds" in mm_data and "image_grid_thw" in mm_data:
            return MultiModalInputs(
                type="multimodal",
                prompt=tokenizer.decode(prompt),
                prompt_token_ids=prompt,
                mm_kwargs=mm_data,
                mm_hashes=None,
                mm_placeholders=None,
            )
        
        mm_processor = self.mm_registry.create_processor(self.model_config,
                                                         tokenizer=tokenizer)

        if mm_processor_kwargs is None:
            mm_processor_kwargs = {}

        return mm_processor.apply(prompt, mm_data, mm_processor_kwargs,
                                  return_mm_hashes)

async def _process_multimodal_async_vllm085(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        mm_processor_kwargs: Optional[Mapping[str, object]],
        lora_request: Optional[LoRARequest],
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:
        """Async version of :meth:`_process_multimodal`."""
        # At the moment on model (PrithviGeoSpatialMAE) requires to be
        # initialized without a tokenizer while using also multi-modal input
        
        if not self.tokenizer:
            tokenizer = object()  # Dummy
        else:
            tokenizer_group = self.get_tokenizer_group()
            tokenizer = await tokenizer_group.get_lora_tokenizer_async(
                lora_request)

        if "image_embeds" in mm_data and "image_grid_thw" in mm_data:
            return MultiModalInputs(
                type="multimodal",
                prompt=tokenizer.decode(prompt),
                prompt_token_ids=prompt,
                mm_kwargs=mm_data,
                mm_hashes=None,
                mm_placeholders=None,
            )

        mm_processor = self.mm_registry.create_processor(self.model_config,
                                                         tokenizer=tokenizer)
        if mm_processor_kwargs is None:
            mm_processor_kwargs = {}

        return mm_processor.apply(prompt, mm_data, mm_processor_kwargs,
                                  return_mm_hashes)

@classmethod
def from_seq_group(
    cls, seq_group: "SequenceGroupMetadata", positions: range
) -> tuple[MultiModalKwargs, dict[str, "MultiModalPlaceholderMap"]]:

    seq_mm_data = seq_group.multi_modal_data
    seq_mm_placeholders = seq_group.multi_modal_placeholders

    if seq_mm_data and not seq_mm_placeholders:
        return seq_mm_data, {}
    
    if not seq_mm_data or not seq_mm_placeholders:
        return MultiModalKwargs({}), {}

    placeholder_maps = dict[str, MultiModalPlaceholderMap]()

    for modality, placeholders in seq_mm_placeholders.items():
        placeholder_map = MultiModalPlaceholderMap()

        if positions:
            placeholder_map.append_items_from_seq_group(
                positions,
                # Dummy, since we don't care about intersecting items
                [None] * len(placeholders),
                placeholders,
            )

        placeholder_maps[modality] = placeholder_map

    return seq_mm_data, placeholder_maps

def replace_with_npu_qwen2_5_image_emb():
    vllm.inputs.preprocess.InputPreprocessor._process_multimodal = _process_multimodal
    vllm.inputs.preprocess.InputPreprocessor._process_multimodal_async = _process_multimodal_async

def image_emb_reuse():
    vllm.inputs.preprocess.InputPreprocessor._process_multimodal = _process_multimodal_vllm085
    vllm.inputs.preprocess.InputPreprocessor._process_multimodal_async = _process_multimodal_async_vllm085
    vllm.multimodal.base.MultiModalPlaceholderMap.from_seq_group = from_seq_group