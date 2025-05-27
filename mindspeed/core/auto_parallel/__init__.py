# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import operator
from functools import reduce
import datetime
import threading

import torch
import numpy as np
import pandas as pd

KV_STORE = None
ITERATION_LOOP_TIME = 5
BAND_WIDTH_UNIDIRECTIONAL = 25 # GB/s
SEARCH_CACHE_PATH = None
MODULE_PATTERN = 'PP{}_TP{}_DP{}_CP{}_UP{}_MBS{}_MODULE.json'
OPERATOR_PATTERN = 'PP{}_TP{}_DP{}_CP{}_UP{}_MBS{}_OPERATOR'


# Operator dims after merging
ARD_NUM_DIMS = {
    'MatMul': 3,
    'BatchMatMul': 4,
    'Softmax': 4,
    'SoftmaxGrad': 4,
    'RmsNorm': 3,
    'RmsNormGrad': 3,
    'LayerNorm': 3,
    'LayerNormGrad': 3,
    'FlashAttentionScore': 3,
    'FlashAttentionScoreGrad': 3
}


# profiling data filed
class KeyField:
    OpType = 'Type'
    InputShapes = 'Input Shapes'
    OutputShapes = 'Output Shapes'
    Duration = 'Duration(us)'
    FwdTime = 'fwd_time'
    BwdTime = 'bwd_time'


class GlobalMemoryBuffer:
    buffers_length = [0, 0, 0]
    buffers = [None, None, None]

    @staticmethod
    def get_tensor(shape: list, index):
        if index not in (0, 1, 2):
            raise AssertionError('index must be 0, 1, 2')
        data_type = torch.float16
        required_len = reduce(operator.mul, shape, 1)
        if GlobalMemoryBuffer.buffers_length[index] < required_len:
            GlobalMemoryBuffer.buffers[index] = torch.empty(
                required_len, dtype=data_type, requires_grad=False, device=torch.cuda.current_device()
            )
            GlobalMemoryBuffer.buffers_length[index] = required_len
        return GlobalMemoryBuffer.buffers[index][0:required_len].view(*shape).uniform_()


class SingletonType(type):
    single_lock = threading.RLock()

    def __call__(cls, *args, **kwargs):
        with SingletonType.single_lock:
            if not hasattr(cls, "_instance"):
                cls._instance = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instance


class SampleCache:
    def __init__(self):
        self.MatMul = {}
        self.RmsNorm = {}
        self.RmsNormGrad = {}
        self.BatchMatMul = {}
        self.Add = {}
        self.LayerNorm = {}
        self.LayerNormGrad = {}
        self.ScaledMaskedSoftmax = {}
        self.ScaledMaskedSoftmaxGrad = {}
        self.FastGeluGrad = {}
        self.FastGelu = {}
        self.Mul = {}
        self.Softmax = {}
        self.SoftmaxGrad = {}
        self.FlashAttentionScore = {}
        self.FlashAttentionScoreGrad = {}

    def clear_cache(self):
        for attr in self.__dict__:
            setattr(self, attr, {})


class ModelManager:
    def __init__(self, npu_type='910B'):
        self.models = {}
        self.npu_type = npu_type

    def cache_model(self, model, op):
        self.models[op] = model

    def get_cached_model(self, model_name: str):
        return self.models.get(model_name, None)

    def load_model(self, model, op, model_dir):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Can't find '{model_dir}'.")
        path = os.path.join(model_dir, f"{op}_{self.npu_type}.pth")
        weight = torch.load(path)
        model.set_model_info(weight.popitem()[1])
        model.load_state_dict(weight)
        # if use model to predict,need to set training=False,otherwise require inputs dims==model_train_inputs dims
        # during fit,after clear model cache(self.train()),training's value will be reset True
        model.training = False
        self.models[op] = model

    def save_model(self, model, op, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=False)
        weight = model.state_dict()
        weight['model_info'] = model.get_model_info()
        torch.save(weight, f'{model_dir}/{op}_{self.npu_type}.pth')

    def save_models(self, model_dir):
        for op, op_model in self.models.items():
            self.save_model(op_model, op, model_dir)


class OperateProfileCache(metaclass=SingletonType):
    def __init__(self):
        self.data_frame = pd.DataFrame(
            columns=[KeyField.OpType, KeyField.InputShapes, KeyField.OutputShapes, KeyField.FwdTime, KeyField.BwdTime]
        )

    def record(self, op_type: str, input_shapes: list, output_shapes: list, fwd_time: float, bwd_time: float):
        _, _, exist = self.find(op_type, input_shapes)
        if not exist:
            input_shapes_str = OperateProfileCache.shapes_to_str(input_shapes)
            output_shape_str = OperateProfileCache.shapes_to_str(output_shapes)
            self.data_frame.loc[len(self.data_frame.index)] = [
                op_type, input_shapes_str, output_shape_str, fwd_time, bwd_time
            ]

    def find(self, op_type: str, input_shapes: list):
        input_shapes_str = OperateProfileCache.shapes_to_str(input_shapes)
        data = self.data_frame[
            (self.data_frame[KeyField.OpType] == op_type) &
            (self.data_frame[KeyField.InputShapes] == input_shapes_str)
        ]
        fwd_time = data[KeyField.FwdTime].mean()
        bwd_time = data[KeyField.BwdTime].mean()
        from_cache = False if np.isnan(fwd_time) and np.isnan(bwd_time) else True
        return fwd_time, bwd_time, from_cache

    @staticmethod
    def shapes_to_str(shapes):
        result = ''
        index = 0
        for shape in shapes:
            result += ','.join(map(lambda x: str(x), shape)) if isinstance(shape, list) else str(shape)
            if index < len(shapes) - 1:
                result += ';' if isinstance(shape, list) else ','
            index += 1
        result = '"' + result
        result = result + '"'
        return result


def get_cache_path():
    global SEARCH_CACHE_PATH
    if SEARCH_CACHE_PATH is None:
        SEARCH_CACHE_PATH = os.getcwd() + os.sep + 'autoparallel_temp_cache' + os.sep
        try:
            os.makedirs(SEARCH_CACHE_PATH, exist_ok=True)
            print(f"Create cache: {SEARCH_CACHE_PATH}")
        except Exception:
            print(f'Create cache directory failed')
            SEARCH_CACHE_PATH = os.getcwd()
    return SEARCH_CACHE_PATH


def analyse_module_profile(profile_file, key):
    if key not in ('step_time', 'transformer_act_mem'):
        raise AssertionError('key[{}] error'.format(key))
    
    if not os.path.exists(path=profile_file):
        return float('inf')
    
    with open(profile_file, 'r') as file:
        try:
            content = file.read()
            content = json.loads(content)
            return float(content.get(key))
        except Exception:
            return float('inf')


def set_kv_store(args):
    global KV_STORE
    if args.node_rank == 0:
        KV_STORE = torch.distributed.TCPStore(
            host_name=args.master_addr,
            port=int(args.master_port) + 2,
            world_size=args.nnodes,
            is_master=True,
            timeout=datetime.timedelta(seconds=30)
        )
    else:
        KV_STORE = torch.distributed.TCPStore(
            host_name=args.master_addr,
            port=int(args.master_port) + 2,
            world_size=args.nnodes,
            is_master=False
        )


def get_kv_store():
    global KV_STORE
    if KV_STORE is None:
        raise AssertionError('KV_STORE must be initialized')
    return KV_STORE


# init SingletonType class
model_manager = ModelManager()
sample_cache = SampleCache()
operator_cache = OperateProfileCache()
