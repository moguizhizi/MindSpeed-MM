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
import stat
import sys
import time
import json
import copy
import re
import operator
import functools
import subprocess
import signal
import threading

import pandas as pd
import torch
import torch_npu
from torch_npu.profiler.profiler import analyse
from megatron.training.global_vars import set_args, get_args

from mindspeed.core.auto_parallel import (
    SingletonType,
    get_cache_path,
    get_kv_store,
    analyse_module_profile,
    MODULE_PATTERN,
    OPERATOR_PATTERN,
    BAND_WIDTH_UNIDIRECTIONAL
)


class BaseLaunch:
    def __init__(self):
        self.old_args = None

    def launch(self, config):
        def update_or_append_param(argv: list, key, value=None):
            if not value:
                argv.append(key)
                return

            if key in argv:
                argv[argv.index(key) + 1] = value
            else:
                argv.extend([key, value])

        def remove_param(argv: list, key, has_value=False):
            if key in argv:
                pos = argv.index(key)
                argv.pop(pos)
                if has_value:
                    argv.pop(pos)

        def monitor_exit(process):
            while True:
                exit_flag = get_kv_store().get("exit_flag")
                if int(exit_flag) == 1:
                    try:
                        process_group_id = os.getpgid(process.pid)
                        os.killpg(process_group_id, signal.SIGKILL)
                        break
                    except ProcessLookupError:
                        break
                time.sleep(60)

        args = get_args()
        argv: list = sys.argv[1:]
        update_or_append_param(argv, '--eval-iters', '0')
        update_or_append_param(argv, '--train-iters', '5')
        update_or_append_param(argv, '--global-batch-size', str(args.global_batch_size))
        update_or_append_param(argv, '--num-layers', str(args.num_layers))
        update_or_append_param(argv, '--pipeline-model-parallel-size', str(args.pipeline_model_parallel_size))
        update_or_append_param(argv, '--tensor-model-parallel-size', str(args.tensor_model_parallel_size))
        update_or_append_param(argv, '--micro-batch-size', str(args.micro_batch_size))
        update_or_append_param(argv, '--sequence-parallel')
        if args.profile_operator:
            update_or_append_param(argv, '--profile-operator')
        if args.profile_memory:
            update_or_append_param(argv, '--profile-memory')
        if args.module_profile_path:
            update_or_append_param(argv, '--prof-file', str(args.module_profile_path))
        if args.context_parallel_algo == 'hybrid_cp_algo':
            update_or_append_param(argv, '--context-parallel-algo', 'hybrid_cp_algo')
            update_or_append_param(argv, '--context-parallel-size', str(args.context_parallel_size))
            update_or_append_param(argv, '--ulysses-degree-in-cp', str(args.ulysses_degree_in_cp))
        if args.context_parallel_algo == 'megatron_cp_algo':
            update_or_append_param(argv, '--context-parallel-algo', 'megatron_cp_algo')
            update_or_append_param(argv, '--context-parallel-size', str(args.context_parallel_size))
        if args.context_parallel_algo == 'ulysses_cp_algo':
            update_or_append_param(argv, '--context-parallel-algo', 'ulysses_cp_algo')
            update_or_append_param(argv, '--context-parallel-size', str(args.context_parallel_size))
        remove_param(argv, '--auto-parallel')

        command = [
            'torchrun', 
            '--nproc_per_node', str(args.nproc_per_node),
            '--nnodes', str(args.nnodes),
            '--node-rank', str(args.node_rank),
            '--master_addr', str(args.master_addr),
            '--master_port', str(args.master_port),
            str(sys.argv[0])
        ] + argv

        get_kv_store().set("exit_flag", "0")
        process = subprocess.Popen(command, shell=False, preexec_fn=lambda: os.setpgrp())
        monitor_thread = threading.Thread(target=monitor_exit, args=(process,))
        monitor_thread.start()
        process.wait()
        get_kv_store().set("exit_flag", "1")
        torch.distributed.barrier()

    def update_args(self, config):
        args = get_args()
        self.old_args = copy.deepcopy(args)

        args.pipeline_model_parallel_size = config[0]
        args.tensor_model_parallel_size = config[1]
        args.data_parallel_size = config[2]
        args.context_parallel_size = config[3] * config[4]
        args.ulysses_degree_in_cp = config[4]
        args.micro_batch_size = config[5]
        if config[3] > 1 and config[4] > 1:
            args.context_parallel_algo = 'hybrid_cp_algo'
            args.use_cp_send_recv_overlap = True
        elif config[3] > 1 and config[4] == 1:
            args.context_parallel_algo = 'megatron_cp_algo'
            args.use_cp_send_recv_overlap = True
        elif config[3] == 1 and config[4] > 1:
            args.context_parallel_algo = 'ulysses_cp_algo'

    def recover_args(self):
        set_args(self.old_args)


class DistributedMemoryProfiler(BaseLaunch):
    def update_args(self, config):
        super().update_args(config)
        args = get_args()
        args.module_profile_path = (get_cache_path() + MODULE_PATTERN).format(*config)
        args.global_batch_size = args.pipeline_model_parallel_size * args.data_parallel_size * args.micro_batch_size
        args.num_layers = args.pipeline_model_parallel_size
        args.profile_memory = True

    def launch(self, config):
        args = get_args()
        if args.node_rank != 0:
            self.update_args(config)
            super().launch(config)
            super().recover_args()
            return None

        self.update_args(config)
        module_profile_path = get_args().module_profile_path
        if os.path.exists(module_profile_path):
            super().recover_args()
            return analyse_module_profile(module_profile_path, key='transformer_act_mem')

        buffer = config + [0]
        torch.distributed.broadcast(torch.tensor(buffer, dtype=torch.int), 0)

        super().launch(config)
        super().recover_args()
        return analyse_module_profile(module_profile_path, key='transformer_act_mem')


class DistributedOperateProfiler(BaseLaunch):
    def update_args(self, config):
        super().update_args(config)
        args = get_args()
        args.module_profile_path = None
        args.operator_profile_path = (get_cache_path() + OPERATOR_PATTERN).format(*config)
        args.global_batch_size = 4 * args.pipeline_model_parallel_size * args.data_parallel_size * args.micro_batch_size
        args.num_layers = 2 * args.pipeline_model_parallel_size
        args.profile_operator = True

    def launch(self, config):
        self.update_args(config)
        args = get_args()
        if args.node_rank != 0:
            super().launch(config)
            super().recover_args()
            return None

        operator_profile_path = args.operator_profile_path
        if os.path.exists(operator_profile_path):
            super().recover_args()
            return operator_profile_path, None

        buffer = config + [1]
        torch.distributed.broadcast(torch.tensor(buffer, dtype=torch.int), 0)

        os.environ['ASCEND_WORK_PATH'] = operator_profile_path
        os.makedirs(operator_profile_path)
        super().launch(config)
        super().recover_args()

        analyse_thread = threading.Thread(
            target=analyse, args=(operator_profile_path + os.sep + 'profiling_data', 32)
        )
        analyse_thread.daemon = True
        analyse_thread.start()
        return operator_profile_path, analyse_thread


class DistributedPerformanceProfiler(BaseLaunch):
    def update_args(self, config):
        super().update_args(config)
        args = get_args()
        args.module_profile_path = (get_cache_path() + MODULE_PATTERN).format(*config)

    def launch(self, config):
        self.update_args(config)
        args = get_args()
        if args.node_rank != 0:
            super().launch(config)
            super().recover_args()
            return None

        module_profile_path = get_args().module_profile_path
        if os.path.exists(module_profile_path):
            super().recover_args()
            return analyse_module_profile(module_profile_path, key='step_time')

        buffer = config + [2]
        torch.distributed.broadcast(torch.tensor(buffer, dtype=torch.int), 0)
        super().launch(config)
        super().recover_args()
        return analyse_module_profile(module_profile_path, key='step_time')


class OperateProfile(metaclass=SingletonType):
    def __init__(self, args):
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
            data_simplification=False
        )
        activities = [torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
        self.op_profiler = torch_npu.profiler.profile(
            activities=activities,
            record_shapes=True,
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=2),
            experimental_config=experimental_config,
        )
        self.op_profiler.start()

    def step(self):
        if torch.distributed.get_rank() in (0,):
            self.op_profiler.step()

    def stop(self):
        if torch.distributed.get_rank() in (0,):
            self.op_profiler.stop()


class Profiling(metaclass=SingletonType):
    MEMORY_UNIT = 1024 ** 3

    def __init__(self, args, warmup_step=3, stop_step=5):
        self.args = args
        self.warmup_step = warmup_step
        self.stop_step = stop_step
        self.curr_step = 0
        self.pattern = r'^module.module.language_model.encoder.layers.\d+$'
        self.context = {
            'step_time': 0,
            'transformer_act_mem': 0
        }

    def should_profiling(self):
        rank = torch.distributed.get_rank()
        if rank in self.args.profile_ranks and \
            self.warmup_step <= self.curr_step < self.stop_step:
            return True
        return False

    def forward_pre_hook(self):
        def hook(module, *args, **kwargs):
            if torch.distributed.get_rank() in self.args.profile_ranks:
                torch.npu.synchronize()
                self.start_memory = torch.npu.memory_allocated()
                torch.npu.reset_max_memory_allocated()
        return hook

    def forward_post_hook(self):
        def hook(module, *args, **kwargs):
            if torch.distributed.get_rank() in self.args.profile_ranks:
                torch.npu.synchronize()
                self.end_memory = torch.npu.max_memory_allocated()
                transformer_act_mem = (self.end_memory - self.start_memory) / Profiling.MEMORY_UNIT
                self.context['transformer_act_mem'] = transformer_act_mem
        return hook

    def register_recursive_hook(self, prefix_name, model):
        model = model[0] if isinstance(model, list) else model
        for name, module in model.named_children():
            next_name = prefix_name + "." + name if prefix_name != "" else name
            if re.fullmatch(self.pattern, next_name):
                module.register_forward_pre_hook(self.forward_pre_hook())
                module.register_forward_hook(self.forward_post_hook())
                break
            self.register_recursive_hook(next_name, module)

    def hook_train_step(self, train_step):
        def custom_train_step(*args, **kwargs):
            start_time = time.time()
            result = train_step(*args, **kwargs)
            torch.cuda.synchronize()
            step_time = time.time() - start_time
            if self.should_profiling():
                cur_step_time = self.context.get('step_time')
                cur_step_time += (step_time - cur_step_time) / (self.curr_step - self.warmup_step + 1)
                self.context['step_time'] = cur_step_time
            self.export_to_file()
            self.curr_step += 1
            return result
        return custom_train_step
    
    def export_to_file(self):
        if torch.distributed.get_rank() in self.args.profile_ranks:
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            modes = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(self.args.prof_file, flags, modes), 'w') as fout:
                fout.write(json.dumps(self.context))
            

class CommProfiling:
    @staticmethod
    def get_comm_time(shape, domains, op):
        if domains == 1:
            return 0

        if op == 'all_reduce':
            return CommProfiling.cal_all_reduce(shape, domains)
        if op == 'all_gather':
            return CommProfiling.cal_all_gather(shape, domains)
        if op == 'alltoall':
            return CommProfiling.cal_alltoall(shape, domains)
        if op == 'reduce_scatter':
            return CommProfiling.cal_reduce_scatter(shape, domains)
        raise AssertionError('communicate operator type error')

    @staticmethod
    def cal_all_reduce(shape, domains):
        data_size = CommProfiling.get_data_size(shape)
        data_size = data_size / domains * (domains - 1) * domains * 2
        band_width = domains * (domains - 1) / 2 * BAND_WIDTH_UNIDIRECTIONAL
        return CommProfiling.div(data_size, band_width)
    
    @staticmethod
    def cal_all_gather(shape, domains):
        data_size = CommProfiling.get_data_size(shape)
        data_size = data_size / domains * (domains - 1) * domains
        band_width = domains * (domains - 1) / 2 * BAND_WIDTH_UNIDIRECTIONAL
        return CommProfiling.div(data_size, band_width)
    
    @staticmethod
    def cal_alltoall(shape, domains):
        data_size = CommProfiling.get_data_size(shape)
        data_size = data_size / domains * (domains - 1) * domains
        band_width = domains * (domains - 1) / 2 * BAND_WIDTH_UNIDIRECTIONAL
        return CommProfiling.div(data_size, band_width)
    
    @staticmethod
    def cal_reduce_scatter(shape, domains):
        data_size = CommProfiling.get_data_size(shape)
        data_size = data_size / domains * (domains - 1) * domains
        band_width = domains * (domains - 1) / 2 * BAND_WIDTH_UNIDIRECTIONAL
        return CommProfiling.div(data_size, band_width)

    @staticmethod
    def get_send_recv_time(shape):
        data_size = CommProfiling.get_data_size(shape)
        return (data_size / BAND_WIDTH_UNIDIRECTIONAL) * 1e6

    @staticmethod
    def get_data_size(shape):
        return functools.reduce(operator.mul, shape) * 2 // 1024**3
    
    @staticmethod
    def div(data_size, band_width):
        try:
            return data_size / band_width * 1e6
        except ZeroDivisionError:
            print(f"band_width is zero")
            return 0