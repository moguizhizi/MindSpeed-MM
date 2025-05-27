# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import os
import json
import torch.distributed

_ALL_CONFIG = {}  # {name: DetachedConfig()}
# model_idx:  0           1
#       vae rank0 ↘
#                    vit rank2,3
#       t5  rank1 ↗
_RANK_NUMBER_TO_MODEL_INDEX = []  # rank index (list index) -- model index -- [0, 0, 1, 1]
_RANK_NUMBER_TO_MODEL_NAME = []  # rank index (list index) -- model name -- ['vae', 't5', 'vit', 'vit']
_NUMBER_OF_MODELS = 0
_USE_MULTIPARAM_SEND_RECV = False
_ALL_DIST_MODEL_INDEX = []
_ALL_DIST_MODEL_NAME = []
_ALL_DIST_MODEL_CONFIG = []
_SUPPORT_MODEL_NAME = {"internvl2": ["vit", "gpt"], "opensoraplan1.3": ["vae", "dit"]}


class ContextKey:
    DIST_CONFIG = 'dist_config'
    # global config keys
    MODEL_CONFIG = 'model_config'
    USE_MULTIPARAM_SEND_RECV = 'use_multiparam_send_recv'
    MODEL_NAME = 'model_name'
    # model config keys
    NAME = 'name'
    MODEL_INDEX = 'model_index'
    WORLD_SIZE = 'world_size'
    TENSOR_MODEL_PARALLEL_SIZE = 'tensor_model_parallel_size'
    PIPELINE_MODEL_PARALLEL_SIZE = 'pipeline_model_parallel_size'
    CONTEXT_PARALLEL_SIZE = 'context_parallel_size'
    MAIN_DP = 'main_dp'
    FORWARD_ONLY = 'forward_only'


CK = ContextKey()


class ModelConfig:
    def __init__(self, config_dict: dict, start_rank):
        self._keys = {CK.NAME, CK.MODEL_INDEX, CK.WORLD_SIZE, CK.TENSOR_MODEL_PARALLEL_SIZE,
                      CK.PIPELINE_MODEL_PARALLEL_SIZE, CK.CONTEXT_PARALLEL_SIZE, CK.FORWARD_ONLY, CK.MAIN_DP}
        self._base_validate(config_dict)

        setattr(self, CK.NAME, None)
        setattr(self, CK.MODEL_INDEX, None)
        setattr(self, CK.WORLD_SIZE, None)
        setattr(self, CK.TENSOR_MODEL_PARALLEL_SIZE, 1)
        setattr(self, CK.PIPELINE_MODEL_PARALLEL_SIZE, 1)
        setattr(self, CK.CONTEXT_PARALLEL_SIZE, 1)
        setattr(self, CK.FORWARD_ONLY, False)
        setattr(self, CK.MAIN_DP, False)
        self._set_single_model_config(config_dict)

        # Additional generated attributes.
        self.start_rank = start_rank
        self.ranks = list(range(self.start_rank, self.start_rank + getattr(self, CK.WORLD_SIZE)))

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        delattr(self, key)

    def __repr__(self):
        repr_str = '('
        for k in self._keys:
            repr_str += f'{k}: {getattr(self, k)}, '
        repr_str = repr_str.rstrip(', ') + ')'
        return repr_str

    def _set_single_model_config(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)
            self._keys.add(k)

    def _base_validate(self, ori_cfg):
        # startswith
        if any(key.startswith('_') for key in ori_cfg.keys()):
            raise ValueError('The configuration item field cannot start with an underscore (_) '
                             'to prevent unexpected overwriting.')
        # check valid key
        valid_keys = list(self._keys)
        invalid_keys = [key for key in ori_cfg if key not in valid_keys]
        if invalid_keys:
            raise KeyError(f"The following keys in DistTrain config are not valid: {invalid_keys}")
        # world_size
        world_size = ori_cfg.get(CK.WORLD_SIZE)
        if not (isinstance(world_size, int) and world_size > 0):
            raise ValueError(f'`{CK.WORLD_SIZE}` ({world_size}) should be greater than or equal to 0')
        # parallel
        tp_size = ori_cfg.get(CK.TENSOR_MODEL_PARALLEL_SIZE, 1)
        pp_size = ori_cfg.get(CK.PIPELINE_MODEL_PARALLEL_SIZE, 1)
        cp_size = ori_cfg.get(CK.CONTEXT_PARALLEL_SIZE, 1)
        if not (isinstance(tp_size, int) and tp_size > 0):
            raise ValueError(f'`{CK.TENSOR_MODEL_PARALLEL_SIZE}` ({tp_size}) should be greater than 0')
        if not (isinstance(pp_size, int) and pp_size > 0):
            raise ValueError(f'`{CK.PIPELINE_MODEL_PARALLEL_SIZE}` ({pp_size}) should be greater than 0')
        if not (isinstance(cp_size, int) and cp_size > 0):
            raise ValueError(f'`{CK.CONTEXT_PARALLEL_SIZE}` ({cp_size}) should be greater than 0')
        if world_size % (tp_size * pp_size * cp_size):
            raise ValueError((f'`{CK.WORLD_SIZE}` ({world_size}) should be divisible by the product of '
                              f'`{CK.TENSOR_MODEL_PARALLEL_SIZE}` ({tp_size}), `{CK.PIPELINE_MODEL_PARALLEL_SIZE}` '
                              f'({pp_size}), and `{CK.CONTEXT_PARALLEL_SIZE}` ({cp_size})'))
        if CK.FORWARD_ONLY in ori_cfg and not isinstance(ori_cfg.get(CK.FORWARD_ONLY), bool):
            raise TypeError(f"The `{CK.FORWARD_ONLY}` value type must be bool.")


def validate_configs_world_size(args):
    world_size = 0
    for cfg in _ALL_CONFIG.values():
        world_size += cfg[CK.WORLD_SIZE]
    if world_size != args.world_size:
        raise ValueError('The sum of `world_size` in config must be equal to the actual `world_size`.')


def get_all_config():
    return _ALL_CONFIG


def get_all_config_size():
    return len(_ALL_CONFIG)


def get_rank_number_to_model_index():
    return _RANK_NUMBER_TO_MODEL_INDEX


def get_rank_number_to_model_name():
    return _RANK_NUMBER_TO_MODEL_NAME


def get_dist_model_name(rank: int = None, global_index: int = None) -> str:
    if global_index is not None:
        if not (0 - _NUMBER_OF_MODELS <= global_index < _NUMBER_OF_MODELS):
            raise ValueError(f'`global_index` must between `0 - _NUMBER_OF_MODELS` ({0 - _NUMBER_OF_MODELS}) '
                             f'and `_NUMBER_OF_MODELS` ({_NUMBER_OF_MODELS})')
        key = list(_ALL_CONFIG.keys())[global_index]
        index_name = _ALL_CONFIG[key][CK.NAME]
        if rank is None:
            return index_name
        else:
            if not (0 <= rank < len(_RANK_NUMBER_TO_MODEL_NAME)):
                raise IndexError(f'{rank=} should between 0 and {len(_RANK_NUMBER_TO_MODEL_NAME)=}, '
                                 f'check the config file and launch params')
            name = _RANK_NUMBER_TO_MODEL_NAME[rank]
            if index_name != name:
                raise RuntimeError(f'{rank=}, `{index_name}` should equals `{name}`')
            return name

    if rank is None:
        rank = torch.distributed.get_rank()
    if not (0 <= rank < len(_RANK_NUMBER_TO_MODEL_NAME)):
        raise IndexError(f'{rank=} should between 0 and {len(_RANK_NUMBER_TO_MODEL_NAME)=}, '
                         f'check the config file and launch params')

    name = _RANK_NUMBER_TO_MODEL_NAME[rank]
    return name


def get_dist_model_config(name: str = None, rank: int = None, global_index: int = None):
    if global_index is not None:
        if not (0 - _NUMBER_OF_MODELS <= global_index < _NUMBER_OF_MODELS):
            raise ValueError(f'`global_index` must between `0 - _NUMBER_OF_MODELS` ({0 - _NUMBER_OF_MODELS}) '
                             f'and `_NUMBER_OF_MODELS` ({_NUMBER_OF_MODELS})')
    if name is not None:
        if rank is not None or global_index is not None:
            if name != get_dist_model_name(rank, global_index):
                raise RuntimeError(f'{rank=}, `{name}` should equals `{get_dist_model_name(rank, global_index)}`')
    else:
        name = get_dist_model_name(rank, global_index)
    if name not in _ALL_CONFIG.keys():
        raise KeyError(f'{name=} not in {_ALL_CONFIG.keys()=}')
    return _ALL_CONFIG[name]


def get_dist_model_index(rank: int = None) -> int:
    if rank is None:
        rank = torch.distributed.get_rank()
    if not (0 - len(_RANK_NUMBER_TO_MODEL_INDEX) <= rank < len(_RANK_NUMBER_TO_MODEL_INDEX)):
        raise IndexError(f'{0 - len(_RANK_NUMBER_TO_MODEL_INDEX)=} <= {rank=} < {len(_RANK_NUMBER_TO_MODEL_INDEX)=}, '
                         f'check the config file and launch params')
    return _RANK_NUMBER_TO_MODEL_INDEX[rank]


def get_dist_global_model_index(rank: int = None) -> int:
    name = get_dist_model_name(rank)
    keys = _ALL_CONFIG.keys()
    return list(keys).index(name)


def is_use_multiparam_send_recv():
    return _USE_MULTIPARAM_SEND_RECV


def _read_json(json_path):
    try:
        with open(json_path, mode="r") as f:
            json_file = f.read()
        configs_list = json.loads(json_file)
        return configs_list
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The file {json_path} does not exist.") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"The file {json_path} is not a valid JSON file.") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}") from e


def _check_config(config_dict):
    if CK.MODEL_CONFIG not in config_dict.keys():
        raise KeyError(f"The `{CK.MODEL_CONFIG}` key does not exist in DistTrain config.")
    if CK.USE_MULTIPARAM_SEND_RECV in config_dict.keys() and not isinstance(config_dict[CK.USE_MULTIPARAM_SEND_RECV], bool):
        raise TypeError(f"The `{CK.USE_MULTIPARAM_SEND_RECV}` value type must be bool.")
    if CK.MODEL_NAME not in config_dict.keys():
        raise KeyError(f"The `{CK.MODEL_NAME}` key does not exist in DistTrain config.")
    if not isinstance(config_dict[CK.MODEL_NAME], str):
        raise TypeError(f"The `{CK.MODEL_NAME}` value type must be string.")
    global _SUPPORT_MODEL_NAME
    if config_dict[CK.MODEL_NAME] not in _SUPPORT_MODEL_NAME:
        raise ValueError(f"The `{CK.MODEL_NAME}` current not support.")
    valid_keys = [CK.MODEL_CONFIG, CK.USE_MULTIPARAM_SEND_RECV, CK.MODEL_NAME]
    invalid_keys = [key for key in config_dict.keys() if key not in valid_keys]
    if invalid_keys:
        raise KeyError(f"Get unexpected keywords: {invalid_keys}")
    if not isinstance(config_dict[CK.MODEL_CONFIG], list):
        raise TypeError(f"The `{CK.MODEL_CONFIG}` type must be list.")
    if not config_dict[CK.MODEL_CONFIG]:
        raise ValueError(f"The `{CK.MODEL_CONFIG}` must not be empty.")
    global _ALL_DIST_MODEL_INDEX, _ALL_DIST_MODEL_NAME, _ALL_DIST_MODEL_CONFIG
    _ALL_DIST_MODEL_INDEX = [config.get(CK.MODEL_INDEX) for config in config_dict[CK.MODEL_CONFIG]]
    _ALL_DIST_MODEL_NAME = [config.get(CK.NAME) for config in config_dict[CK.MODEL_CONFIG]]
    _ALL_DIST_MODEL_CONFIG = config_dict[CK.MODEL_CONFIG]
    if not all(key in config.keys() for config in _ALL_DIST_MODEL_CONFIG for key in [CK.NAME, CK.WORLD_SIZE, CK.MODEL_INDEX]):
        raise ValueError(f"At least three items must be configured: `{CK.NAME}`, `{CK.WORLD_SIZE}`, and `{CK.MODEL_INDEX}`.")
    if not all(isinstance(name, str) for name in _ALL_DIST_MODEL_NAME):
        raise TypeError(f"The `{CK.NAME}` value type must be str.")
    if len(_ALL_DIST_MODEL_NAME) != len(set(_ALL_DIST_MODEL_NAME)):
        raise ValueError(f"`{CK.NAME}` is duplicate in DistTrain config.")
    if not all(name.isidentifier() for name in _ALL_DIST_MODEL_NAME):
        raise ValueError(f"`{CK.NAME}` is not a valid string.")
    valid_names = _SUPPORT_MODEL_NAME.get(config_dict[CK.MODEL_NAME])
    if len(_ALL_DIST_MODEL_NAME) != len(valid_names):
        raise ValueError(f"`{config_dict[CK.MODEL_NAME]}` model current only support {valid_names}.")
    if not all(isinstance(index, int) for index in _ALL_DIST_MODEL_INDEX):
        raise TypeError(f"The `{CK.MODEL_INDEX}` value type must be int.")
    _ALL_DIST_MODEL_INDEX.sort()
    if not all(_ALL_DIST_MODEL_INDEX[i] - _ALL_DIST_MODEL_INDEX[i - 1] == 1 for i in range(1, len(_ALL_DIST_MODEL_INDEX))):
        raise ValueError(f"`{CK.MODEL_INDEX}` must be continuous.")

    # 把model_index升序的name保存
    combined = list(zip(_ALL_DIST_MODEL_INDEX, _ALL_DIST_MODEL_CONFIG))
    combined.sort(key=lambda x: x[0])
    _, _ALL_DIST_MODEL_CONFIG = list(zip(*combined))
    if _ALL_DIST_MODEL_CONFIG[0][CK.MODEL_INDEX] < 0:
        raise ValueError(f"`{CK.MODEL_INDEX}` must start from 0.")
    if not all(name == valid for name, valid in zip(_ALL_DIST_MODEL_NAME, valid_names)):
        raise ValueError(f"`{CK.NAME}` sequence is incorrect, {config_dict[CK.MODEL_NAME]} "
                         f"model name list strictly follow the sequence [{valid_names}].")
    if not all(
        isinstance(config.get(CK.MAIN_DP), bool)
        for config in _ALL_DIST_MODEL_CONFIG
        if CK.MAIN_DP in config
    ):
        raise TypeError(f"The `{CK.MAIN_DP}` value type must be bool.")
    if sum(1 for config in _ALL_DIST_MODEL_CONFIG if config.get(CK.MAIN_DP, False)) > 1:
        raise ValueError(f"Only one `{CK.MAIN_DP}` can be true.")


def _set_config(config_dict):
    _check_config(config_dict)
    global _NUMBER_OF_MODELS, _ALL_DIST_MODEL_CONFIG
    _NUMBER_OF_MODELS = len(_ALL_DIST_MODEL_CONFIG)
    config_dict[CK.MODEL_CONFIG] = _ALL_DIST_MODEL_CONFIG
    # Save the config in ascending order by name.
    for k, v in config_dict.items():
        if k == CK.USE_MULTIPARAM_SEND_RECV:
            global _USE_MULTIPARAM_SEND_RECV
            _USE_MULTIPARAM_SEND_RECV = v
        elif k == CK.MODEL_CONFIG:
            global _ALL_CONFIG, _RANK_NUMBER_TO_MODEL_NAME, _RANK_NUMBER_TO_MODEL_INDEX
            for model_config in v:  # v == [{}, {}, {}, ...]
                _ALL_CONFIG[model_config.get(CK.NAME)] = ModelConfig(model_config, len(_RANK_NUMBER_TO_MODEL_INDEX))
                _RANK_NUMBER_TO_MODEL_INDEX.extend([model_config.get(CK.MODEL_INDEX)] * model_config.get(CK.WORLD_SIZE))
                _RANK_NUMBER_TO_MODEL_NAME.extend([model_config.get(CK.NAME)] * model_config.get(CK.WORLD_SIZE))
            print(f"{_ALL_CONFIG=}\n{_RANK_NUMBER_TO_MODEL_NAME=}\n{_RANK_NUMBER_TO_MODEL_INDEX=}")


def _clear_dist_config():
    global _ALL_CONFIG, _RANK_NUMBER_TO_MODEL_NAME, _RANK_NUMBER_TO_MODEL_INDEX, _NUMBER_OF_MODELS, \
        _USE_MULTIPARAM_SEND_RECV, _ALL_DIST_MODEL_INDEX, _ALL_DIST_MODEL_NAME, _ALL_DIST_MODEL_CONFIG
    _ALL_CONFIG = {}
    _RANK_NUMBER_TO_MODEL_NAME = []
    _RANK_NUMBER_TO_MODEL_INDEX = []
    _NUMBER_OF_MODELS = 0
    _USE_MULTIPARAM_SEND_RECV = False
    _ALL_DIST_MODEL_INDEX = []
    _ALL_DIST_MODEL_NAME = []
    _ALL_DIST_MODEL_CONFIG = []


def merge_dist_train_args(path):
    real_path = os.path.realpath(path)
    if real_path.endswith(".json"):  # MindSpeed-MM use json config
        config = _read_json(real_path)
        if isinstance(config, dict):
            config = config.get(CK.DIST_CONFIG, {})
        else:
            raise ValueError('Unexpected json file, not contain dist_config dict data.')
    else:
        raise TypeError("Unexpected file type.")
    _clear_dist_config()
    _set_config(config)


def is_forward_only_model(name: str = None, rank: int = None, global_index: int = None):
    return get_dist_model_config(name, rank, global_index)[CK.FORWARD_ONLY]
