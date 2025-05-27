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
#include "atb/train_op_params.h"
#include "atb/infer_op_params.h"
#endif

using namespace std;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
namespace {
const static int RMSNORM_LAYERTYPE = 1;
const static int SAVE_X = 0;
const static int SAVE_RSTD = 1;
const static int SAVE_GAMMA = 2;
const static int N = 32;

void InferShapeRmsNorm(c10::SmallVector<int64_t, N> &size, const at::Tensor &self, const at::Tensor &gamma)
{
    int64_t rstd_dim = self.dim();
    rstd_dim -= gamma.dim();
    TORCH_CHECK(rstd_dim >= 0,
                "RmsNorm intensor gamma dim error,gamma's dim should not greater than x's dim");
    for (uint64_t i = 0; i < self.dim(); i++) {
        if (i < rstd_dim) {
            size.emplace_back(self.size(i));
        } else {
            size.emplace_back(1);
        }
    }
}

void CheckRmsNorm(const at::Tensor &x, const at::Tensor &gamma)
{
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::BFloat16 ||
                x.scalar_type() == at::ScalarType::Float,
                "Input x dtype ", x.scalar_type(), " invalid, should be float, float16 or bfloat16");
    TORCH_CHECK(x.scalar_type() == gamma.scalar_type(),
                "Input x dtype should be same with gamma, but got x ", x.scalar_type(), " gamma ", gamma.scalar_type());
}

class NPURmsNormFunction : public torch::autograd::Function<NPURmsNormFunction> {
public:
    static at::Tensor forward(
        AutogradContext *ctx, const at::Tensor &x, const at::Tensor &gamma, float epsilon)
    {
#ifndef ENABLE_ATB
        TORCH_CHECK(false, "ATB RmsNorm not implemented");
#else
        at::AutoNonVariableTypeMode g;
        c10::SmallVector<int64_t, N> tensor_rstd_shape;
        CheckRmsNorm(x, gamma);
        InferShapeRmsNorm(tensor_rstd_shape, x, gamma);
        // apply tensor
        at::Tensor tensor_rstd = at::empty(at::IntArrayRef(tensor_rstd_shape), x.options().dtype(at::ScalarType::Float));
        at::Tensor tensor_y = at::empty(x.sizes(), x.options());

        atb::infer::RmsNormParam param;
        param.layerType = (atb::infer::RmsNormParam::RmsNormType)RMSNORM_LAYERTYPE;
        param.normParam.epsilon = epsilon;
        param.normParam.rstd = true;

        // set input and output
        ParamSetter paramsetter;
        paramsetter.Input(x)
                   .Input(gamma)
                   .Output(tensor_y)
                   .Output(tensor_rstd);

        atb::Operation* op = nullptr;
        atb::CreateOperation(param, &op);
        TORCH_CHECK(op != nullptr, "RmsNorm get op failed!");
        RunAtbCmd(op, paramsetter, "RmsNorm_forward");

        ctx->save_for_backward({x, tensor_rstd, gamma});

        return tensor_y;
#endif
    }

    static std::vector<at::Tensor> backward(AutogradContext *ctx, std::vector<at::Tensor> grad_output)
    {
#ifndef ENABLE_ATB
        TORCH_CHECK(false, "RmsNormBackward not implemented");
#else
        auto saved = ctx->get_saved_variables();
        auto x = saved[SAVE_X];
        auto rstd = saved[SAVE_RSTD];
        auto gamma = saved[SAVE_GAMMA];
        atb::train::RmsNormBackwardParam param;

        at::Tensor tensor_x_grad = at::empty(x.sizes(), x.options());
        at::Tensor tensor_gamma_grad = at::empty(gamma.sizes(), gamma.options().dtype(at::ScalarType::Float));

        ParamSetter paramsetter;
        paramsetter.Input(grad_output[0])
                   .Input(x)
                   .Input(rstd)
                   .Input(gamma)
                   .Output(tensor_x_grad)
                   .Output(tensor_gamma_grad);

        atb::Operation* op = nullptr;
        atb::CreateOperation(param, &op);
        TORCH_CHECK(op != nullptr, "RmsNormBackward get op failed!");
        RunAtbCmd(op, paramsetter, "RmsNorm_backward");

        return {tensor_x_grad, tensor_gamma_grad, at::Tensor()};
#endif
    }
};
} // namespace

at::Tensor npu_rms_norm(const at::Tensor &x, const at::Tensor &gamma, float epsilon)
{
    return NPURmsNormFunction::apply(x, gamma, epsilon);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &npu_rms_norm, "rms_norm on ascend device",
            pybind11::arg("x"), pybind11::arg("gamma"), pybind11::arg("epsilon")=1e-6);
}
