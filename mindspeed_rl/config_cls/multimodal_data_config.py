# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
from mindspeed_rl.config_cls.base_config import BaseConfig


class MultiModalDataConfig(BaseConfig):
    def __init__(self, config_dict):
        # Multimodal dataset prompt key
        self.prompt_key: str = None

        # Multimodal dataset answer key
        self.answer_key: str = None

        # Multimodal dataset image key
        self.image_key: str = None

        # 
        self.truncation: str = None

        # Shuffle dataset
        self.shuffle: bool = False

        # Multimodal dataset maximum pixels
        self.max_pixels: int = None

        # Multimodal dataset minimum pixels
        self.min_pixels: int = None

        # Return raw chat or not
        self.return_raw_chat: bool = False

        # Filter overlong prompts or not
        self.filter_overlong_prompts: bool = True

        if config_dict is not None:
            self.update(config_dict)
