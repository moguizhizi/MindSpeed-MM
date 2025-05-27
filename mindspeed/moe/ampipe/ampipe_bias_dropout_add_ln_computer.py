# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
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
import torch


class BiasDropoutAddNormComputer:
    def __init__(self, bdal_tensor_list, fwd_args):
        super().__init__()
        self.bdal_tensor_list = bdal_tensor_list
        self.fwd_args = fwd_args

    def forward(self, ctx, input_tensor):
        residual = self.fwd_args.residual
        bias = self.fwd_args.bias
        prob = self.fwd_args.prob

        input_tensor = input_tensor.detach()
        residual = residual.detach()
        input_tensor.requires_grad = True
        residual.requires_grad = True
        ctx.bias = False
        if isinstance(bias, torch.Tensor):
            bias = bias.detach()
            bias.requires_grad = True
            self.bdal_tensor_list.append(bias)
            ctx.bias = True

        with torch.enable_grad():
            ln_input = self.fwd_args.bias_dropout_add_func(input_tensor, bias, residual, prob)
            detach_ln_input = ln_input.detach()
            detach_ln_input.requires_grad = True
            output = self.fwd_args.post_attention_norm(detach_ln_input)
        self.bdal_tensor_list.extend([ln_input, detach_ln_input, input_tensor, residual])
        return output, ln_input

    def backward(self, ctx, saved_tensor_list, grad_ln_outs, grad_ln_ins):
        if ctx.bias:
            bias = saved_tensor_list.pop(0)
        ln_input, detach_ln_input, input_tensor, residual, output = saved_tensor_list
        output.backward(grad_ln_outs)
        grad_ln = detach_ln_input.grad
        ln_input.backward(grad_ln + grad_ln_ins)
        input_grad = input_tensor.grad
        residual_grad = residual.grad
        bias_grad = bias.grad if ctx.bias else None
        return input_grad, residual_grad, bias_grad
