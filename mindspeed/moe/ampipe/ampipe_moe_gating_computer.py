# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright (c) Microsoft Corporation.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

# copied from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py
# reworked/refactored some parts to make it run.
from collections import namedtuple

import torch

from megatron.training import get_args
from mindspeed.moe.utils import einsum


class MoEGatingComputer:
    def __init__(self, moe, gate_tensor_list):
        super().__init__()
        self.gate_tensor_list = gate_tensor_list
        self.moe = moe

    def forward(self, logits):
        detach_logits = logits.detach()
        detach_logits.requires_grad = True

        d_model = logits.shape[-1]
        with torch.enable_grad():
            reshaped_input = detach_logits.reshape(-1, d_model)
            global_args = get_args()
            if not global_args.enable_token_rearrange_opt:
                l_aux, combine_weights, dispatch_mask = self.moe.moe_layer.gate(reshaped_input)
                dispatch_mask = dispatch_mask.type_as(logits)
                dispatched_input = einsum("sec,sm->ecm", dispatch_mask, reshaped_input)
                self.gate_tensor_list.append(detach_logits)
                return dispatched_input, l_aux, combine_weights
            else:
                l_aux, (token_ec_idx, token_weights, expert_select_token_idx) = self.moe.moe_layer.gate(reshaped_input)
                org_dtype = reshaped_input.dtype
                if org_dtype == torch.bfloat16:  # 规避算子性能劣化问题, 解决后可删除
                    rearranged_input = torch.index_select(
                        reshaped_input.to(torch.float32), dim=0, index=expert_select_token_idx
                    ).to(org_dtype)
                else:
                    rearranged_input = torch.index_select(
                        reshaped_input, dim=0, index=expert_select_token_idx
                    )
                capacity = expert_select_token_idx.size(0) // self.moe.num_experts
                dispatched_input = rearranged_input.reshape(self.moe.num_experts, capacity, d_model).contiguous()
                self.gate_tensor_list.append(detach_logits)
                GatingComputerRet = namedtuple('GatingComputerRet',
                                               ['dispatched_input', 'l_aux',
                                                'token_ec_idx', 'token_weights'])
                gating_computer_ret = GatingComputerRet(dispatched_input=dispatched_input, l_aux=l_aux,
                                                        token_ec_idx=token_ec_idx, token_weights=token_weights)
                return gating_computer_ret

    def backward(self, saved_tensor_list, grad_output):
        logits, dispatched_input = saved_tensor_list
        dispatched_input.backward(grad_output)
        grad_logits = logits.grad
        logits.untyped_storage().resize_(0)
        return grad_logits
