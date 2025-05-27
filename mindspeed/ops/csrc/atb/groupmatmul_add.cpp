// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <vector>
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <iostream>

#ifdef ENABLE_ATB
#include "inc/atb_adapter.h"
#include "atb/operation.h"
#include "atb/train_op_params.h"
#include "atb/infer_op_params.h"
#include "../flop_counter/flop_counter.h"
#endif

using namespace std;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
namespace {


void groupmatmul_add_fp32(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &group_list, at::Tensor & grad)
{
#ifndef ENABLE_ATB
        TORCH_CHECK(false, "ATB MatmulAdd not implemented");
#else
        atb::infer::GroupedMatmulInplaceAddParam  param;
        param.transposeA = true;                    // 是否转置A矩阵
        param.transposeB = false;                     // 是否转置B矩阵

        ParamSetter paramsetter;
        paramsetter.Input(x)
                   .Input(weight)
                   .Input(group_list)
                   .Input(grad)
                   .Output(grad);
        // 构造算子并执行
        atb::Operation* op = nullptr;
        atb::CreateOperation(param, &op);
        TORCH_CHECK(op != nullptr, "GroupMatmulAdd get op failed!");
        RunAtbCmd(op, paramsetter, "GroupedMatmulInplaceAddOperation");
        #ifdef FLOP_COUNT
        FLOP_COUNT(FlopCounter::gmm_add_flop, x, weight, group_list);
        #endif
        return ;
#endif
}
} // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_groupmatmul_add_fp32", &groupmatmul_add_fp32, "matmul_add on ascend device",
            pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("group_list"), pybind11::arg("grad"));
}
