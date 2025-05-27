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
import time
import math
from functools import reduce

import numpy as np
import torch
import torch_npu
from megatron.training.global_vars import get_args

from mindspeed.core.auto_parallel import (
    ITERATION_LOOP_TIME,
    BAND_WIDTH_UNIDIRECTIONAL,
    operator_cache,
    GlobalMemoryBuffer
)
from mindspeed.core.auto_parallel.auto_parallel_rectify import Sampler
from mindspeed.core.auto_parallel.auto_parallel_profiling import CommProfiling
from mindspeed.model.transformer import (
    get_attention_mask,
    generate_attention_mask
)


class Linear(torch.nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, inputs):
        x, y = inputs
        return torch.matmul(x, y.t())


class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=hidden_size, eps=eps)

    def forward(self, x):
        return self.layer_norm(*x)


class FusedRmsNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=torch.float16)).npu()
        self.eps = eps

    def forward(self, x):
        return torch_npu.npu_rms_norm(x[0], self.weight, epsilon=self.eps)[0]


class BatchMatMul(torch.nn.Module):
    def __init__(self):
        super(BatchMatMul, self).__init__()

    def forward(self, inputs):
        x, y = inputs
        return torch.bmm(x, y)


class FlashAttention(torch.nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.pre_tockens = 65536
        self.next_tockens = 0

        generate_attention_mask()
        self.attention_mask = get_attention_mask()

    def forward(self, x):
        q, k, v = x
        seq_length, _, hd = q.shape[0], q.shape[1], q.shape[2]
        head_num = hd // self.head_dim
        output = torch_npu.npu_fusion_attention(
            q, k, v, head_num, 'SBH',
            pse=None,
            padding_mask=None,
            atten_mask=self.attention_mask,
            scale=self.scale,
            pre_tockens=self.pre_tockens,
            next_tockens=self.next_tockens,
            keep_prob=1.0,
            inner_precise=0,
            sparse_mode=get_args().sparse_mode
        )[0]
        return output


class TransformerBlock:
    def __init__(self):
        self.number_sample = 100
        self.noise_model = OperatorNoiseSampler(self.number_sample)

    def norm(self):
        args = get_args()
        tp = args.tensor_model_parallel_size
        cp = args.context_parallel_size // args.ulysses_degree_in_cp
        up = args.ulysses_degree_in_cp
        input_shape = [args.seq_length // cp // tp // up, args.micro_batch_size, args.hidden_size]
        if args.normalization == 'RMSNorm':
            ftime, btime = self.noise_model.fused_rms_norm(input_shape, input_shape, args.hidden_size)
        else:
            ftime, btime = self.noise_model.layernorm(input_shape, input_shape, args.hidden_size)
        return ftime, btime

    def self_attention_with_fa(self):
        args = get_args()
        tp = args.tensor_model_parallel_size
        cp = args.context_parallel_size // args.ulysses_degree_in_cp
        up = args.ulysses_degree_in_cp
        ftime, btime = self.noise_model.flash_attention(
            [args.seq_length // cp, args.micro_batch_size, args.hidden_size // tp // up],
            [args.seq_length // cp, args.micro_batch_size, args.hidden_size // tp // up],
            [args.seq_length // cp, args.micro_batch_size, args.hidden_size // tp // up],
            [args.seq_length // cp, args.micro_batch_size, args.hidden_size // tp // up],
            args.hidden_size // args.num_attention_heads,
        )
        return ftime, btime

    def get_block_time(self):
        args = get_args()
        s = args.seq_length
        a = args.num_attention_heads
        h = args.hidden_size
        ffn = args.ffn_hidden_size if args.ffn_hidden_size is not None else 4 * args.hidden_size
        d = args.hidden_size // args.num_attention_heads
        b = args.micro_batch_size
        tp = args.tensor_model_parallel_size
        cp = args.context_parallel_size // args.ulysses_degree_in_cp
        up = args.ulysses_degree_in_cp

        fwd_time = np.array([0 for _ in range(self.number_sample)]).astype(np.float64)
        bwd_time = np.array([0 for _ in range(self.number_sample)]).astype(np.float64)

        ftime, btime = self.norm()
        fwd_time += ftime
        bwd_time += btime

        all_gather_time = CommProfiling.get_comm_time([s // cp // up // tp, b, h], tp, 'all_gather')
        reduce_scatter_time = CommProfiling.get_comm_time([s // cp // up, b, h], tp, 'reduce_scatter')
        fwd_time += all_gather_time
        bwd_time += reduce_scatter_time

        ftime, btime = self.noise_model.matmul(
            [s // cp // up * b, h], 
            [3 * h // tp, h], 
            [s // cp // up * b, 3 * h // tp]
        )
        fwd_time += ftime
        bwd_time += btime

        if not args.use_flash_attn:
            raise AssertionError('the auto-parallel only support FA')
        else:
            alltoall_time = CommProfiling.get_comm_time([s // cp // up, b, a // tp, d], up, 'alltoall')
            fwd_time += (3 * alltoall_time)
            bwd_time += (3 * alltoall_time)

            send_recv_time = CommProfiling.get_send_recv_time([2, 2, s // cp // 2, b, a // tp // up * d])
            ftime, btime = self.self_attention_with_fa()
            for _ in range(cp - 1):
                fwd_time += max([ftime.max(), send_recv_time])
                bwd_time += max([btime.max(), send_recv_time])
            fwd_time += ftime
            bwd_time += btime

            alltoall_time = CommProfiling.get_comm_time([s // cp, b, a // tp // up, d], up, 'alltoall')
            fwd_time += alltoall_time
            bwd_time += alltoall_time

        ftime, btime = self.noise_model.matmul([s // cp // up * b, h // tp], [h, h // tp], [s // cp // up * b, h])
        fwd_time += ftime
        bwd_time += btime

        reduce_scatter_time = CommProfiling.get_comm_time([s // cp // up, b, h], tp, 'reduce_scatter')
        all_gather_time = CommProfiling.get_comm_time([s // cp // up // tp, b, h], tp, 'all_gather')
        fwd_time += reduce_scatter_time
        bwd_time += all_gather_time

        ftime, btime = self.norm()
        fwd_time += ftime
        bwd_time += btime

        all_gather_time = CommProfiling.get_comm_time([s // cp // up // tp, b, h], tp, 'all_gather')
        reduce_scatter_time = CommProfiling.get_comm_time([s // cp // up, b, h], tp, 'reduce_scatter')
        fwd_time += all_gather_time
        bwd_time += reduce_scatter_time

        ftime, btime = self.noise_model.matmul([s // cp // up * b, h], [ffn // tp, h], [s // cp // up * b, ffn // tp])
        fwd_time += ftime
        bwd_time += btime

        # 4h->h
        ftime, btime = self.noise_model.matmul([s // cp // up * b, ffn // tp], [h, ffn // tp], [s // cp // up * b, h])
        fwd_time += ftime
        bwd_time += btime

        reduce_scatter_time = CommProfiling.get_comm_time([s // cp // up, b, h], tp, 'reduce_scatter')
        all_gather_time = CommProfiling.get_comm_time([s // cp // up // tp, b, h], tp, 'all_gather')
        fwd_time += reduce_scatter_time
        bwd_time += all_gather_time

        return fwd_time, bwd_time


class OperatorNoiseSampler:
    def __init__(self, num_sample=100):
        self.sampling = Sampler(num_sample=num_sample)

    @staticmethod
    def measure_matmul_time(left_shape, left_transpose, right_shape, right_transpose):
        left_matrix = GlobalMemoryBuffer.get_tensor(left_shape, 0)
        left_matrix = left_matrix if not left_transpose else left_matrix.t()
        right_matrix = GlobalMemoryBuffer.get_tensor(right_shape, 1)
        right_matrix = right_matrix if not right_transpose else right_matrix.t()

        for _ in range(ITERATION_LOOP_TIME):
            torch.matmul(left_matrix, right_matrix)

        torch.npu.synchronize()
        start_time = time.time()
        for _ in range(ITERATION_LOOP_TIME):
            torch.matmul(left_matrix, right_matrix)
        torch.npu.synchronize()
        return (time.time() - start_time) * 1e6 / ITERATION_LOOP_TIME

    @staticmethod
    def measure_batchmatmul_time(left_shape, left_transpose, right_shape, right_transpose):
        left_matrix = GlobalMemoryBuffer.get_tensor(left_shape, 0)
        left_matrix = left_matrix if not left_transpose else left_matrix.permute(0, 2, 1)
        right_matrix = GlobalMemoryBuffer.get_tensor(right_shape, 0)
        right_matrix = right_matrix if not right_transpose else right_matrix.permute(0, 2, 1)

        for _ in range(ITERATION_LOOP_TIME):
            torch.bmm(left_matrix, right_matrix)

        torch.npu.synchronize()
        start_time = time.time()
        for _ in range(ITERATION_LOOP_TIME):
            torch.bmm(left_matrix, right_matrix)
        torch.npu.synchronize()
        return (time.time() - start_time) * 1e6 / ITERATION_LOOP_TIME

    def matmul(self, input_shape1, input_shape2, output_shape):
        ftime, _, from_cache = operator_cache.find('MatMul', [input_shape1, input_shape2])
        if not from_cache:
            ftime = self.measure_matmul_time(input_shape1, False, input_shape2, True)
        ftime_uncertainty = self.sampling.run('MatMul', ftime, output_shape, input_shape1, input_shape2)
        operator_cache.record('MatMul', [input_shape1, input_shape2], output_shape, ftime, 0)

        btime1, _, from_cache = operator_cache.find('MatMul', [output_shape, input_shape2])
        if not from_cache:
            btime1 = self.measure_matmul_time(output_shape, False, input_shape2, False)
        btime1_uncertainty = self.sampling.run('MatMul', btime1, input_shape1, output_shape, input_shape2)
        operator_cache.record('MatMul', [output_shape, input_shape2], input_shape1, btime1, 0)

        btime2, _, from_cache = operator_cache.find('MatMul', [output_shape, input_shape1])
        if not from_cache:
            btime2 = self.measure_matmul_time(output_shape, True, input_shape1, False)
        btime2_uncertainty = self.sampling.run('MatMul', btime2, input_shape2, output_shape, input_shape1)
        operator_cache.record('MatMul', [output_shape, input_shape1], input_shape2, btime2, 0)
        return ftime_uncertainty, btime1_uncertainty + btime2_uncertainty

    def batch_matmul(self, input_shape1, input_shape2, output_shape):
        ftime, _, from_cache = operator_cache.find('BatchMatMul', [input_shape1, input_shape2])
        if not from_cache:
            ftime = self.measure_batchmatmul_time(input_shape1, False, input_shape2, False)
        ftime_uncertainty = self.sampling.run('BatchMatMul', ftime, output_shape, input_shape1, input_shape2)
        operator_cache.record('BatchMatMul', [input_shape1, input_shape2], output_shape, ftime, 0)

        btime1, _, from_cache = operator_cache.find('BatchMatMul', [input_shape1, output_shape])
        if not from_cache:
            btime1 = self.measure_batchmatmul_time(input_shape1, True, output_shape, False)
        btime1_uncertainty = self.sampling.run('BatchMatMul', btime1, input_shape2, input_shape1, output_shape)
        operator_cache.record('BatchMatMul', [input_shape1, output_shape], input_shape2, btime1, 0)

        btime2, _, from_cache = operator_cache.find('BatchMatMul', [output_shape, input_shape2])
        if not from_cache:
            btime2 = self.measure_batchmatmul_time(output_shape, False, input_shape2, True)
        btime2_uncertainty = self.sampling.run('BatchMatMul', btime2, input_shape1, output_shape, input_shape2)
        operator_cache.record('BatchMatMul', [output_shape, input_shape2], input_shape1, btime2, 0)
        return ftime_uncertainty, btime1_uncertainty + btime2_uncertainty

    def layernorm(self, input_shape, output_shape, hidden_size, eps=1e-5):
        layernorm = LayerNorm(hidden_size, eps)
        ftime, btime, from_cache = operator_cache.find('LayerNorm', input_shape)
        if not from_cache:
            ftime, btime = TimeCostModel.profile(layernorm, [input_shape])
        ftime_uncertainty = self.sampling.run('LayerNorm', ftime, output_shape, input_shape)
        btime_uncertainty = self.sampling.run('LayerNormGrad', btime, input_shape, output_shape)
        operator_cache.record('LayerNorm', input_shape, output_shape, ftime, btime)
        return ftime_uncertainty, btime_uncertainty

    def fused_rms_norm(self, input_shape, output_shape, hidden_size, eps=1e-6):
        fused_rms_norm = FusedRmsNorm(hidden_size, eps)
        ftime, btime, from_cache = operator_cache.find('RmsNorm', input_shape)
        if not from_cache:
            ftime, btime = TimeCostModel.profile(fused_rms_norm, [input_shape])
        ftime_uncertainty = self.sampling.run('RmsNorm', ftime, output_shape, input_shape)
        btime_uncertainty = self.sampling.run('RmsNormGrad', btime, output_shape, input_shape)
        operator_cache.record('RmsNorm', input_shape, output_shape, ftime, btime)
        return ftime_uncertainty, btime_uncertainty

    def flash_attention(self, q, k, v, output_shape, head_dim):
        flash_attn = FlashAttention(head_dim)
        ftime, btime, from_cache = operator_cache.find('FlashAttentionScore', [q, k, v])
        if not from_cache:
            ftime, btime = TimeCostModel.profile(flash_attn, [q, k, v])
        ftime_uncertainty = self.sampling.run('FlashAttentionScore', ftime, output_shape, q, k, v)
        btime_uncertainty = self.sampling.run('FlashAttentionScoreGrad', btime, output_shape, q, k, v)
        operator_cache.record('FlashAttentionScore', [q, k, v], q, ftime, btime)
        return ftime_uncertainty, btime_uncertainty


class TimeCostModel(object):
    def __init__(self):
        args = get_args()
        self.seq_length = args.seq_length
        self.hidden_size = args.hidden_size
        self.pp_size = args.pipeline_model_parallel_size
        self.dp_size = args.data_parallel_size
        self.micro_batch_size = args.micro_batch_size
        self.num_layers_per_stage = args.num_layers // args.pipeline_model_parallel_size
        self.num_micro_batch = args.global_batch_size // args.micro_batch_size // args.data_parallel_size

    def get_iteration_time(self):
        transformer_block = TransformerBlock()
        fwd_time, bwd_time = transformer_block.get_block_time()
        fwd_time *= self.num_layers_per_stage
        bwd_time *= self.num_layers_per_stage
        iteration_times = np.array([0 for _ in range(fwd_time.shape[0])]).astype(np.float64)
        for i in range(fwd_time.shape[0]):
            iteration_times[i] = self.pipeline_costmodel(fwd_time[i], bwd_time[i])
        return iteration_times

    def pipeline_costmodel(self, fwd_time, bwd_time):
        if self.pp_size == 1:
            return (fwd_time + bwd_time) * self.num_micro_batch

        send_recv_time = CommProfiling.get_send_recv_time(
            [self.seq_length, self.micro_batch_size, self.hidden_size]
        )
        # p and m start with 1
        SF = np.zeros((self.pp_size + 1, self.num_micro_batch + 1), np.float64)
        SB = np.zeros((self.pp_size + 1, self.num_micro_batch + 1), np.float64)
        EF = np.zeros((self.pp_size + 1, self.num_micro_batch + 1), np.float64)
        EB = np.zeros((self.pp_size + 1, self.num_micro_batch + 1), np.float64)

        warmup = [self.pp_size - p - 1 for p in range(self.pp_size)]
        remaining = [self.num_micro_batch - warmup[p] for p in range(self.pp_size)]

        # warmup
        for p in range(1, self.pp_size + 1):
            for m in range(1, warmup[p - 1] + 1):
                if p == 1:
                    SF[p][m] = (m - 1) * fwd_time
                    EF[p][m] = m * fwd_time
                else:
                    SF[p][m] = max(EF[p][m - 1], EF[p - 1][m] + send_recv_time)
                    EF[p][m] = SF[p][m] + fwd_time

        # 1f1b
        for num_1f1b in range(1, self.num_micro_batch + 1):
            # forward of 1f1b
            for p in range(1, self.pp_size + 1):
                if num_1f1b > remaining[p - 1]:
                    # cool down phase
                    continue
                m = warmup[p - 1] + num_1f1b
                if p == 1:
                    SF[p][m] = EB[p][m + p - self.pp_size - 1]
                    EF[p][m] = SF[p][m] + fwd_time
                else:
                    SF[p][m] = max(EB[p][m + p - self.pp_size - 1], EF[p - 1][m] + send_recv_time)
                    EF[p][m] = SF[p][m] + fwd_time

            # backward of 1f1b
            for p in range(self.pp_size, 0, -1):
                m = num_1f1b
                if num_1f1b > remaining[p - 1]:
                    # cool down phase
                    continue
                if p == self.pp_size:
                    SB[p][m] = EF[p][m]
                else:
                    SB[p][m] = max(EF[p][m + self.pp_size - p], EB[p + 1][m] + send_recv_time)
                EB[p][m] = SB[p][m] + bwd_time

            # cool down phase
            for p in range(self.pp_size, 0, -1):
                m = num_1f1b
                if num_1f1b <= remaining[p - 1]:
                    continue
                SB[p][m] = max(EB[p][m - 1], EB[p + 1][m] + send_recv_time)
                EB[p][m] = SB[p][m] + bwd_time

        e2e_time = max([max(EB[p]) for p in range(self.pp_size)])
        # allreduce_gradients
        e2e_time += 0.0
        return e2e_time
    
    @staticmethod
    def profile(model, shapes):
        model.to(torch.cuda.current_device())

        input_tensors = []
        index = 0
        for shape in shapes:
            tensor = GlobalMemoryBuffer.get_tensor(shape, index).requires_grad_()
            input_tensors.append(tensor)
            index += 1

        sum_z = None
        for _ in range(3):
            sum_z = model(input_tensors)

        # forward_time
        torch.npu.synchronize()
        start_time = time.time()
        for _ in range(ITERATION_LOOP_TIME):
            model(input_tensors)
        torch.npu.synchronize()
        fwd_time = (time.time() - start_time) * 1e6 / ITERATION_LOOP_TIME

        for _ in range(3):
            z = model(input_tensors)
            loss = torch.sum(z)
            loss.backward()

        torch.npu.synchronize()
        start_time = time.time()
        for _ in range(ITERATION_LOOP_TIME):
            torch.sum(sum_z)
        torch.npu.synchronize()
        loss_time = (time.time() - start_time) * 1e6 / ITERATION_LOOP_TIME

        torch.npu.synchronize()
        start_time = time.time()
        for i in range(ITERATION_LOOP_TIME):
            z = model(input_tensors)
            loss = torch.sum(z)
            loss.backward()
        torch.npu.synchronize()
        bwd_time = (time.time() - start_time) * 1e6 / ITERATION_LOOP_TIME - fwd_time - loss_time
        return fwd_time, bwd_time
