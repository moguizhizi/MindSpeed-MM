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

#ifdef ENABLE_ATB
#include "inc/atb_adapter.h"
#include "atb/operation.h"
#include "atb/infer_op_params.h"
#endif

using namespace std;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
namespace {
const static int N = 32;
void InferSwigluForward(c10::SmallVector<int64_t, N> &out_tensor_shape, const at::Tensor &x, int32_t dim)
{
    int64_t split_dim = dim;
    if (split_dim < 0) {
        split_dim += x.dim();
    }
    TORCH_CHECK(split_dim >= 0 && split_dim < x.dim(), "Input dim range is invalid");
    const int32_t split_num = 2;
    out_tensor_shape[split_dim] = x.size(split_dim) / split_num;
}

void CheckSwigluForward(const at::Tensor &x)
{
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::BFloat16 ||
                x.scalar_type() == at::ScalarType::Float, "Input tensor dtype ", x.scalar_type(),
                " invalid, should be float32, float16 or bfloat16");
}

void CheckSwigluBackward(const at::Tensor &y_grad, const at::Tensor &x)
{
    TORCH_CHECK(y_grad.scalar_type() == at::ScalarType::Half || y_grad.scalar_type() == at::ScalarType::BFloat16 ||
                y_grad.scalar_type() == at::ScalarType::Float, "Input y_grad tensor dtype ", y_grad.scalar_type(),
                " invalid, should be float32, float16 or bfloat16");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::BFloat16 ||
                x.scalar_type() == at::ScalarType::Float, "Input x tensor dtype ", x.scalar_type(),
                " invalid, should be float32, float16 or bfloat16");
    TORCH_CHECK(x.scalar_type() == y_grad.scalar_type(), "Input x tensor dtype is not equal to y_grad");
}

class NPUSwigluFunction : public torch::autograd::Function<NPUSwigluFunction> {
public:
    static at::Tensor forward(AutogradContext *ctx, const at::Tensor &x, int32_t dim = -1)
    {
#ifndef ENABLE_ATB
        TORCH_CHECK(false, "swiglu_forward not implemented");
#else
        at::AutoNonVariableTypeMode g;
        CheckSwigluForward(x);
        c10::SmallVector<int64_t, N> out_tensor_shape{x.sizes()};
        InferSwigluForward(out_tensor_shape, x, dim);
        // apply tensor
        at::Tensor y = at::empty(out_tensor_shape, x.options());

        atb::infer::ActivationParam param;
        param.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
        param.dim = dim;

        // set input and output
        ParamSetter paramsetter;
        paramsetter.Input(x)
                   .Output(y);

        atb::Operation* op = nullptr;
        atb::CreateOperation(param, &op);
        TORCH_CHECK(op != nullptr, "swiglu_forward get op failed!");
        RunAtbCmd(op, paramsetter, "swiglu_forward");

        ctx->save_for_backward({x});
        ctx->saved_data["dim"] = dim;
        return y;
#endif
    }

    static std::vector<at::Tensor> backward(AutogradContext *ctx, std::vector<at::Tensor> grad_output)
    {
#ifndef ENABLE_ATB
        TORCH_CHECK(false, "swiglu_backward not implemented");
#else
        auto dim = ctx->saved_data["dim"].toInt();
        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        CheckSwigluBackward(grad_output[0], x);
        at::Tensor x_grad = at::empty(x.sizes(), x.options());

        atb::infer::ActivationParam param;
        param.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_BACKWARD;
        param.dim = dim;

        ParamSetter paramsetter;
        paramsetter.Input(grad_output[0])
                   .Input(x)
                   .Output(x_grad);

        atb::Operation* op = nullptr;
        atb::CreateOperation(param, &op);
        TORCH_CHECK(op != nullptr, "swiglu_backward get op failed!");
        RunAtbCmd(op, paramsetter, "swiglu_backward");

        return {x_grad, at::Tensor()};
#endif
    }
};
} // namespace

at::Tensor npu_swiglu(const at::Tensor &x, int32_t dim)
{
    return NPUSwigluFunction::apply(x, dim);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu", &npu_swiglu, "swiglu realization", pybind11::arg("x"), pybind11::arg("dim")=-1);
}