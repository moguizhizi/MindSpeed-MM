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
import json
import time
import math

import torch
from megatron.training.global_vars import get_args

from mindspeed.core.auto_parallel import set_kv_store
from mindspeed.core.auto_parallel.auto_parallel_optimizer import SearchByGreyBox
from mindspeed.core.auto_parallel.auto_parallel_memory import MemoryCostModel
from mindspeed.core.auto_parallel.auto_parallel_profiling import (
    DistributedMemoryProfiler,
    DistributedOperateProfiler,
    DistributedPerformanceProfiler
)


def filter_unvalid_configs(search_spaces):
    memory_model = MemoryCostModel()
    fitting_configs = memory_model.get_fitting_configurations(search_spaces)
    for config in fitting_configs:
        mem = DistributedMemoryProfiler().launch(config)
        if not math.isinf(mem):
            memory_model.profiled_configs.append(config)
            memory_model.profiled_configs_memory.append(mem)

    print(f"profiled_configs: {memory_model.profiled_configs}")
    print(f"profiled_configs_mem: {memory_model.profiled_configs_memory}")

    memory_model.fit_model()
    valid_configs, valid_configs_memory = [], []
    for config in search_spaces:
        cost_memory = memory_model.get_peak_memory(config)
        if not memory_model.is_oom(cost_memory):
            valid_configs.append(config)
            valid_configs_memory.append(cost_memory)
    return valid_configs


def build_initial_spaces(args):
    world_size = args.nproc_per_node * args.nnodes
    device_count = args.nproc_per_node

    solutions = []
    for pp in range(1, world_size + 1):
        if world_size % pp != 0 or args.num_layers % pp != 0:
            continue

        for i in range(device_count):
            tp = 2 ** i
            if tp > device_count or tp > (world_size // pp):
                break
            if (args.num_query_groups > 1 and args.num_query_groups % tp != 0) \
                or (args.num_attention_heads % tp != 0):
                break

            max_cp_size = world_size // (pp * tp)
            for cp_size in range(1, max_cp_size + 1):
                if world_size % (pp * tp * cp_size) != 0 or \
                        args.global_batch_size % (world_size // (pp * tp * cp_size)) != 0:
                    continue

                for up in range(1, cp_size + 1):
                    if cp_size % up != 0:
                        continue
                    cp = cp_size // up
                    head, remainder = divmod(args.num_attention_heads, up * tp)
                    if (head < 1 or remainder != 0) or (args.seq_length % (2 * cp) != 0):
                        continue

                    dp = world_size // (pp * tp * cp_size)
                    dp_group_batch_size = args.global_batch_size // dp
                    for num_mb in range(1, dp_group_batch_size + 1):
                        if dp_group_batch_size % num_mb != 0:
                            continue
                        mbs = dp_group_batch_size // num_mb
                        solutions.append([pp, tp, dp, cp, up, mbs])
    return solutions


def monitor_train_task():
    while True:
        message = torch.tensor([0 for _ in range(7)], dtype=torch.int)
        torch.distributed.broadcast(message, 0)
        task_type = message[-1].item()
        config = [m.item() for m in message[:-1]]
        if task_type == -1:
            break
        elif task_type == 0:
            DistributedMemoryProfiler().launch(config)
        elif task_type == 1:
            DistributedOperateProfiler().launch(config)
        elif task_type == 2:
            DistributedPerformanceProfiler().launch(config)


def export_results(config):
    results = {}
    results['optimal_parallel_strategy'] = {}
    results['optimal_parallel_strategy']['pipeline-model-parallel-size'] = config[0]
    results['optimal_parallel_strategy']['tensor-model-parallel-size'] = config[1]
    results['optimal_parallel_strategy']['data-parallel-size'] = config[2]
    results['optimal_parallel_strategy']['micro-batch-size'] = config[-1]
    if config[3] > 1 and config[4] > 1:
        results['optimal_parallel_strategy']['context-parallel-algo'] = 'hybrid_cp_algo'
        results['optimal_parallel_strategy']['context-parallel-size'] = config[3] * config[4]
        results['optimal_parallel_strategy']['ulysses-degree-in-cp'] = config[4]
    elif config[3] > 1 and config[4] == 1:
        results['optimal_parallel_strategy']['context-parallel-algo'] = 'megatron_cp_algo'
        results['optimal_parallel_strategy']['context-parallel-size'] = config[3]
    elif config[3] == 1 and config[4] > 1:
        results['optimal_parallel_strategy']['context-parallel-algo'] = 'ulysses_cp_algo'
        results['optimal_parallel_strategy']['context-parallel-size'] = config[4]
    return json.dumps(results)


def search_optimal_configuration(args):
    set_kv_store(args)

    init_method = 'tcp://{}:{}'.format(args.master_addr, int(args.master_port) + 1)
    torch.distributed.init_process_group(
        backend=torch.distributed.Backend.GLOO,
        init_method=init_method,
        rank=args.node_rank,
        world_size=args.nnodes
    )

    if args.node_rank == 0:
        start_time = time.time()
        search_space = build_initial_spaces(args)
        search_space = filter_unvalid_configs(search_space)
        print(f"filter search_space: {len(search_space)}")
        print("\n".join(str(item) for item in search_space), flush=True)
        
        config, _ = SearchByGreyBox().search(get_args(), search_space)
        torch.distributed.broadcast(torch.tensor([-1 for _ in range(7)], dtype=torch.int), 0)

        results = export_results(config)
        print(f"find optimal configuration: {results}, cost_time: {time.time() - start_time}")
    else:
        monitor_train_task()
