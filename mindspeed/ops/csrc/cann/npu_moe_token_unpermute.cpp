// Copyright (c) 2024 Huawei Technologies Co., Ltd
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <torch_npu/csrc/include/ops.h>
#include "inc/aclnn_common.h"

using namespace at_npu::native;
using torch::autograd::AutogradContext;
using torch::autograd::Function;

namespace {
    const static int DIMS = 2;
    const static int MIN_DIMS = 1;
    const static int64_t DEFAULT_TOPK = 1;

    void CheckMoeTokenUnpermuteForward(
        const at::Tensor& permuted_tokens,
        const at::Tensor& sorted_indices,
        c10::optional<at::Tensor>& probs,
        bool padded_mode = false
    )
    {
        if (padded_mode) {
            throw std::runtime_error("current version only support padded_mode is false");
        }
        TORCH_CHECK(permuted_tokens.dim() == DIMS,
                    "The dims of input permuted_tokens should be 2 dimensional, but got ", permuted_tokens.dim(), "-dimensional.");
        TORCH_CHECK(sorted_indices.dim() == MIN_DIMS,
                    "The dims of input sorted_indices should be 1 dimensional, but got ", sorted_indices.dim(), "-dimensional.");
        if (probs.has_value()) {
            TORCH_CHECK(probs.value().dim() == DIMS,
                        "The dims of input probs should be 2 dimensional, but got ", probs.value().dim(), "-dimensional.");
        }
    }

    void CheckMoeTokenUnpermuteBackward(
        const at::Tensor &unpermuted_tokens_grad,
        const at::Tensor &sorted_indices,
        const at::Tensor &probs
    )
    {
        TORCH_CHECK(unpermuted_tokens_grad.dim() == DIMS,
                    "The dims of input unpermuted_tokens_grad should be 2 dimensional, but got ", unpermuted_tokens_grad.dim(), "-dimensional.");
        TORCH_CHECK(sorted_indices.dim() == MIN_DIMS,
                    "The dims of input sorted_indices should be 1 dimensional, but got ", sorted_indices.dim(), "-dimensional.");
        if (probs.defined()) {
            TORCH_CHECK(probs.dim() == DIMS,
                        "The dims of input probs should be 2 dimensional, but got ", probs.dim(), "-dimensional.");
        }
    }

    class NPUMoeTokenUnpermute : public torch::autograd::Function<NPUMoeTokenUnpermute> {
    public:
        static at::Tensor forward(
            AutogradContext *ctx,
            const at::Tensor& permuted_tokens,
            const at::Tensor& sorted_indices,
            c10::optional<at::Tensor>& probs,
            c10::optional<bool> padded_mode,
            c10::optional<at::IntArrayRef>& restore_shape
        )
        {
            at::AutoDispatchBelowADInplaceOrView guard;
            bool padded_mode_vale = padded_mode.value_or(false);
            auto restore_shape_vale = restore_shape.value_or(at::IntArrayRef{1});
            CheckMoeTokenUnpermuteForward(permuted_tokens, sorted_indices, probs, padded_mode_vale);
            int64_t topk = probs.has_value() ? probs.value().size(1) : DEFAULT_TOPK;
            // The sorted_indices actually implemented by the aclnn operator are different from the sorted_indices
            // output by the permute function of the megatron source code.
            // The actual sorted_indices implemented by the aclnn operator are not sliced.
            // so, num_unpermuted_tokens is obtained by dividing sorted_indices.size(0) by topk
            int64_t num_unpermuted_tokens = sorted_indices.size(0) / topk;
            at::Tensor unpermuted_tokens = at::empty({num_unpermuted_tokens, permuted_tokens.size(-1)}, permuted_tokens.options());
            at::Tensor probs_value = probs.has_value() ? probs.value() : at::Tensor();
            ACLNN_CMD(aclnnMoeTokenUnpermute, permuted_tokens, sorted_indices, probs_value, padded_mode_vale, restore_shape_vale, unpermuted_tokens);
            ctx->save_for_backward({permuted_tokens, sorted_indices, probs_value});
            ctx->saved_data["padded_mode"] = padded_mode_vale;
            ctx->saved_data["restore_shape"] = restore_shape;

            return unpermuted_tokens;
        }

        static std::vector<at::Tensor> backward(
            AutogradContext *ctx,
            const std::vector<at::Tensor>& grad_outputs
        )
        {
            auto saved_tensors = ctx->get_saved_variables();
            auto permuted_tokens = saved_tensors[0];
            auto sorted_indices = saved_tensors[1];
            auto probs = saved_tensors[2];
            bool padded_mode = ctx->saved_data["padded_mode"].toBool();
            auto restore_shape = ctx->saved_data["restore_shape"];
            at::IntArrayRef restore_shape_vale{1, 1};

            at::Tensor grad_unpermuted_tokens = grad_outputs[0];
            CheckMoeTokenUnpermuteBackward(grad_unpermuted_tokens, sorted_indices, probs);

            at::Tensor grad_permuted_tokens = at::empty(permuted_tokens.sizes(), permuted_tokens.options());
            at::Tensor grad_probs = probs.defined() ? at::empty(probs.sizes(), probs.options()) : at::empty({0}, permuted_tokens.options());
            ACLNN_CMD(aclnnMoeTokenUnpermuteGrad, permuted_tokens, grad_unpermuted_tokens, sorted_indices, probs, padded_mode, restore_shape_vale, grad_permuted_tokens, grad_probs);
            if (probs.defined()) {
                return {grad_permuted_tokens, at::Tensor(), grad_probs, at::Tensor(), at::Tensor()};
            } else {
                return {grad_permuted_tokens, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
            }
        }
    };
} // namespace

at::Tensor npu_moe_token_unpermute(
    const at::Tensor& permuted_tokens,
    const at::Tensor& sorted_indices,
    c10::optional<at::Tensor>& probs,
    c10::optional<bool> padded_mode,
    c10::optional<at::IntArrayRef>& restore_shape
)
{
    return NPUMoeTokenUnpermute::apply(permuted_tokens, sorted_indices, probs, padded_mode, restore_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_moe_token_unpermute", &npu_moe_token_unpermute,
          "npu moe token unpermute",
          pybind11::arg("permuted_tokens"),
          pybind11::arg("sorted_indices"),
          pybind11::arg("probs") = pybind11::none(),
          pybind11::arg("padded_mode") = false,
          pybind11::arg("restore_shape") = pybind11::none());
}
