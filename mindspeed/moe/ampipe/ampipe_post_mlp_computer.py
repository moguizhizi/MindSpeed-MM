# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright (c) Microsoft Corporation.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

# copied from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py
# reworked/refactored some parts to make it run.
import torch

from megatron.training import get_args
from mindspeed.moe.utils import einsum


class MoEPostMLPComputer:
    def __init__(self, save_tensor_list, fwd_args):
        super().__init__()
        self.fwd_args = fwd_args
        self.save_tensor_list = save_tensor_list

    def forward(self, ctx, mlp_outputs):
        global_args = get_args()
        weights_list = self.fwd_args.weights_list
        token_ec_idx_list = self.fwd_args.token_ec_idx_list
        moe_output_list = self.fwd_args.moe_output_list
        for c in range(global_args.ampipe_degree):
            if not global_args.ampipe_tp_sp_comm_overlap:
                self.fwd_args.a2a_events[c].wait()
            detach_exp_out = mlp_outputs[c].detach()
            detach_exp_out.requires_grad = True
            with torch.enable_grad():
                reshape_out = detach_exp_out.reshape(ctx.ep_size * ctx.num_local_experts, -1, ctx.hidden_size)
                if not global_args.enable_token_rearrange_opt:
                    combine_weights = weights_list[c].type_as(reshape_out)
                    combined_output = einsum("sec,ecm->sm", combine_weights.type_as(reshape_out), reshape_out)
                else:
                    token_rearranged_ec_idx, token_exp_weights = token_ec_idx_list[c], weights_list[c]
                    E, C, M = reshape_out.shape
                    org_dtype = reshape_out.dtype
                    if org_dtype == torch.bfloat16:
                        valid_expert_out = torch.index_select(
                            reshape_out.view(E * C, M).to(torch.float32), dim=0, index=token_rearranged_ec_idx
                        ).to(org_dtype)
                    else:
                        valid_expert_out = torch.index_select(
                            reshape_out.view(E * C, M), dim=0, index=token_rearranged_ec_idx
                        )
                    combined_output = valid_expert_out * token_exp_weights.unsqueeze(1).type_as(reshape_out)
                    if global_args.moe_router_topk == 2:
                        combined_output = torch.add(*torch.chunk(combined_output, global_args.moe_router_topk, dim=0))
                clone_out = combined_output.clone()
                clone_out.untyped_storage().resize_(0)
                self.save_tensor_list.extend([detach_exp_out, clone_out])
            moe_out = combined_output.reshape((self.fwd_args.seqlen, -1, ctx.hidden_size))
            moe_output_list.append(moe_out)
        return moe_output_list

    def backward(self, saved_tensor_list, grad_moe_out_chunk):
        exp_out, combined_output = saved_tensor_list
        combined_output.backward(grad_moe_out_chunk)
        exp_grad = exp_out.grad
        exp_out.untyped_storage().resize_(0)
        return exp_grad
