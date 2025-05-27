# Copyright 2024 OpenAccess AI Collective and the LlamaFactory team.
#
# This code is inspired by the OpenAccess AI Collective's axolotl library.
# https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/monkeypatch/utils.py
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

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, Literal

import torch
import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq, ProcessorMixin

from mindspeed_mm.data.data_utils.func_utils.mm_plugin import LengthParams
from mindspeed_mm.data.data_utils.func_utils.template import Template


@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    r"""
    Data collator that supports VLMs.

    Features should contain input_ids, attention_mask, labels and images.
    """

    template: Optional["Template"] = None
    processor: Optional["ProcessorMixin"] = None

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        batch_images, batch_videos, batch_imglens, batch_vidlens, batch_seqlens = [], [], [], [], []
        for feature in features:
            images = feature.pop("images", None) or []
            videos = feature.pop("videos", None) or []
            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_imglens.append(len(images))
            batch_vidlens.append(len(videos))
            batch_seqlens.append(len(feature["input_ids"]))

        mm_inputs = self.template.mm_plugin.get_mm_inputs(
            batch_images, batch_videos, LengthParams(batch_imglens, batch_vidlens, batch_seqlens), self.processor
        )
        if "token_type_ids" in mm_inputs:
            token_type_ids = mm_inputs.pop("token_type_ids")
            for i, feature in enumerate(features):
                feature["token_type_ids"] = token_type_ids[i]

        features: Dict[str, "torch.Tensor"] = super().__call__(features)
        if self.model is not None and hasattr(self.model, "get_rope_index"):  # for qwen2vl mrope
            rope_index_kwargs = {
                "input_ids": features["input_ids"],
                "image_grid_thw": mm_inputs.get("image_grid_thw"),
                "video_grid_thw": mm_inputs.get("video_grid_thw"),
                "attention_mask": features["attention_mask"],
            }
            if "second_per_grid_ts" in mm_inputs:
                rope_index_kwargs["second_per_grid_ts"] = mm_inputs.get("second_per_grid_ts")

            features["position_ids"], features["rope_deltas"] = self.model.get_rope_index(**rope_index_kwargs)
        features.update(mm_inputs)
        return features


@dataclass
class PairwiseDataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature[f"{key}_input_ids"],
                    "attention_mask": feature[f"{key}_attention_mask"],
                    "labels": feature[f"{key}_labels"],
                    "images": feature["images"],
                    "videos": feature["videos"],
                }
                concatenated_features.append(target_feature)

        return super().__call__(concatenated_features)
