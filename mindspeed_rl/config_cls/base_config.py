# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import json
from pathlib import Path

from mindspeed_rl.utils import Loggers

logger = Loggers("config class")


class BaseConfig:
    '''
    Base configuration class.
    '''

    def update(self, config_dict, model_config_dict=None):
        '''
        Method to update parameters from a config dictionary
        '''
        if 'model' in config_dict:
            # if str, parsed as file path, used for multi-modal
            if isinstance(model_config_dict, str):
                self._process_multi_modal(model_config_dict)
            else:
                self.update(model_config_dict[config_dict['model']])
        for key, value in config_dict.items():
            if key == 'model':
                continue

            key = key.replace('-', '_')
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"The key: {key} is missing, causing the setup to fail. Please check."
                               f" If necessary, register it in the config file.")

    def _process_multi_modal(self, model_config_dict):
        config_path = Path(model_config_dict)
        if not config_path.is_file():
            raise FileNotFoundError(f'model json {str(model_config_dict)} not found!')
        else:
            config = json.loads(config_path.read_text())
            # used for actor_hybrid_worker initialize megatron resharding manager
            setattr(self, 'img_pp_layers',
                    config.get('image_encoder', {}).get('vision_encoder', {}).get('pipeline_num_layers', ''))
            setattr(self, 'llm_pp_layers',
                    config.get('text_decoder', {}).get('pipeline_num_layers', ''))
            # used for mindspeed-mm
            setattr(self, "mm_model", model_config_dict)

    def __repr__(self):
        '''Represent the model config as a string for easy reading'''
        return f"<{self.__class__.__name__} {vars(self)}>"

    def items(self):
        return self.__dict__.items()

    def dict(self):
        return self.__dict__
