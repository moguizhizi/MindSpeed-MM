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
from itertools import product

import numpy as np
import torch
from megatron.training.global_vars import get_args

from mindspeed.core.auto_parallel import SingletonType


class MemoryCostModel(metaclass=SingletonType):
    def __init__(self):
        args = get_args()
        self.num_layers = args.num_layers
        self.num_attn_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.seq_length = args.seq_length
        self.ffn_hidden_size = args.ffn_hidden_size
        if not self.ffn_hidden_size:
            self.ffn_hidden_size = 4 * self.hidden_size

        self.model = None
        self.profiled_configs = []
        self.profiled_configs_memory = []
        self.max_available_memory = None

    @staticmethod
    def cal_coeff(config):
        _, tp, _, cp, up, b = config
        coeff = [
            1,
            b * (1 / tp) * (1 / cp) * (1 / up),
            b * (1 / tp) * (1 / cp) * (1 / cp) * (1 / up),
            b * (1 / cp) * (1 / up)
        ]
        return np.array(coeff)
    
    @staticmethod
    def cal_coeff_matrix(configs):
        coeff_matrix = []
        for config in configs:
            _, tp, _, cp, up, b = config
            coeff_matrix.append([
                1,
                b * (1 / tp) * (1 / cp) * (1 / up),
                b * (1 / tp) * (1 / cp) * (1 / cp) * (1 / up),
                b * (1 / cp) * (1 / up)
            ])
        return np.array(coeff_matrix)

    def is_oom(self, cost_memory):
        if self.max_available_memory is None:
            properties = torch.npu.get_device_properties(0)
            self.max_available_memory = properties.total_memory / (1024 ** 3)
        # 以单卡最大可用内存的1.2倍作为OOM阈值
        return cost_memory > (self.max_available_memory * 1.2)

    def get_fitting_configurations(self, search_spaces):
        search_spaces_matrix = np.array(search_spaces)
        temp_search_spaces = [config for config in search_spaces if config[-1] < 8]

        tp_group = []
        max_tp = search_spaces_matrix[:, 1].max()
        for config in temp_search_spaces:
            _, tp, _, cp, up, _ = config
            if cp == 1 and up == 1 and tp == max_tp:
                tp_group.append(config)

        cp_group = []
        min_cp = search_spaces_matrix[:, 3].min()
        for config in temp_search_spaces:
            pp, tp, _, cp, up, _ = config
            if tp > 1 or up > 1:
                continue
            if pp > 1 and cp > min_cp:
                cp_group.append(config)

        up_group = []
        min_up = search_spaces_matrix[:, 4].min()
        for config in temp_search_spaces:
            pp, tp, _, cp, up, _ = config
            if tp > 1 or cp > 1:
                continue
            if pp > 1 and up > min_up:
                up_group.append(config)

        cp_up_group = []
        for config in temp_search_spaces:
            _, tp, _, cp, up, _ = config
            if tp == 1 and cp > 1 and up > 1:
                cp_up_group.append(config)

        tp_cp_up_group = []
        for config in temp_search_spaces:
            _, tp, _, cp, up, _ = config
            if tp > 1 and cp > 1 and up > 1:
                tp_cp_up_group.append(config)

        product_iter = product(*[tp_group, cp_group, up_group, cp_up_group, tp_cp_up_group])
        fitting_group, cur_condition_number = None, float('inf')

        for group in product_iter:
            # 条件数小于100的矩阵的逆矩阵数值更稳定，拟合效果更好
            if cur_condition_number < 100:
                break

            empty_set = set([row[-1] for row in group])
            if len(empty_set) < 2:
                continue

            coeff_matrix = MemoryCostModel.cal_coeff_matrix(group)
            coeff_matrix = coeff_matrix.transpose() @ coeff_matrix
            if np.linalg.matrix_rank(coeff_matrix) == coeff_matrix.shape[0]:
                con_num = np.linalg.cond(coeff_matrix)
                if con_num < cur_condition_number:
                    fitting_group = group
                    cur_condition_number = con_num

        print(f"fitting_group: {fitting_group} condition_number: {cur_condition_number}", flush=True)
        return fitting_group


    def fit_model(self):
        coeff_matrix = MemoryCostModel.cal_coeff_matrix(self.profiled_configs)
        profiled_configs_memory = np.array(self.profiled_configs_memory)
        self.model = np.linalg.inv(coeff_matrix.transpose() @ coeff_matrix) \
                     @ coeff_matrix.transpose() \
                     @ profiled_configs_memory

    def predict(self, config):
        config_matrix = MemoryCostModel.cal_coeff(config)
        pred_memory = config_matrix @ self.model
        return pred_memory

    def get_peak_memory(self, config):
        args = get_args()
        pp, tp, _ = config[0], config[1], config[-1]
        hidden_size = self.hidden_size
        ffn_hidden_size = self.ffn_hidden_size
        if args.swiglu:
            ffn_hidden_size *= 2
        transformer_params_count = (4 * hidden_size * hidden_size + 2 * hidden_size * ffn_hidden_size) / tp
        total_params_count = transformer_params_count * (self.num_layers // pp)

        mem_para = 2 * total_params_count
        mem_grad = 2 * total_params_count
        mem_optimizer = 12 * total_params_count if args.reuse_fp32_param else 16 * total_params_count
        mem_activation_layer = abs(self.predict(config)) * (1024 ** 3)
        mem_activation_batch = mem_activation_layer * (self.num_layers // pp)
        mem_activation = mem_activation_batch * pp
        m1 = mem_para + mem_optimizer + mem_activation
        m2 = mem_para + mem_optimizer + mem_activation + mem_grad - mem_activation_batch
        peak_memory = max(m1, m2)
        return peak_memory / (1024 ** 3) + 4