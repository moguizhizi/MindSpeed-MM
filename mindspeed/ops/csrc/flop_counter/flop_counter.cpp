// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/core/npu/NPUException.h"

#include "flop_counter.h"


int64_t FlopCounter::mm_flop(const at::Tensor &tensor1, const at::Tensor &tensor2)
{
    // Count flops for matmul.
    // Inputs contains the shapes of two matrices.
    auto dim_tensor1 = tensor1.dim();
    auto dim_tensor2 = tensor2.dim();
    TORCH_CHECK(dim_tensor1 > 0 && dim_tensor2 > 0, "matmul got error dimentions: ", "(", dim_tensor1, ", ",
                dim_tensor2, ")");
    // A(x1, m, k1) and B(x2, k2, n)
    // Get x1 and x2's infer sizes
    auto x1_size = dim_tensor1 > 2 ? dim_tensor1 - 2 : 0;
    auto x2_size = dim_tensor2 > 2 ? dim_tensor2 - 2 : 0;
    at::IntArrayRef x1_sizes(tensor1.sizes().data(), x1_size);
    at::IntArrayRef x2_sizes(tensor2.sizes().data(), x2_size);
    std::vector<int64_t> output_size = at::infer_size(x1_sizes, x2_sizes);

    // Get m
    if (dim_tensor1 >= 2) {
        output_size.push_back(tensor1.size(-2));
    }
    // Get n
    if (dim_tensor2 >= 2) {
        output_size.push_back(tensor2.size(-1));
    }
    // Get k1 and k2
    int64_t k = tensor1.size(-1);
    // Compute
    int64_t flop = 2 * k;
    for (const auto& elem : output_size) {
        flop *= elem;
    }

    return flop;
}

int64_t FlopCounter::coc_flop(const at::Tensor &tensor1, const at::Tensor &tensor2, bool trans, int rankSize, bool is_ag_mm)
{
    // Count flops for coc.
    at::Tensor tensor2_transposed;
    if (trans) {
        tensor2_transposed = at::transpose(tensor2, 0, 1);
    } else {
        tensor2_transposed = tensor2;
    }
    int64_t total_flops = FlopCounter::mm_flop(tensor1, tensor2_transposed);
    return is_ag_mm ? total_flops * rankSize : total_flops;
}

int64_t FlopCounter::bmm_flop(const at::Tensor &self, const at::Tensor &mat2)
{
    // Count flops for the bmm operation.
    // Inputs should be a list of length 2.
    // Inputs contains the shapes of two tensor.
    int64_t b = self.size(0);
    int64_t m = self.size(1);
    int64_t k = self.size(2);
    int64_t b2 = mat2.size(0);
    int64_t k2 = mat2.size(1);
    int64_t n = mat2.size(2);
    TORCH_CHECK(b == b2 && k == k2, "The tensor dimension is incorrect");
    return b * m * n * 2 * k;
}

std::vector<std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>> _unpack_flash_attention_nested_shapes(std::vector<int64_t> query,
    std::vector<int64_t> key, std::vector<int64_t> value, int64_t head_num, std::vector<int64_t> grad_out,
    std::vector<int64_t> cum_seq_q, std::vector<int64_t> cum_seq_k, std::string input_layer_str)
{
    // Given inputs to a flash_attention_(forward|backward) kernel, this will handle behavior for
    // GQA and MQA and TND

    // for GQA and MQA, the dim 2 or 3 of kv should equal to q
    // for general, shape should view to [B, N, S, D]

    std::vector<std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>> result;
    int64_t q_0 = query[0];
    int64_t q_1 = query[1];
    int64_t q_2 = query[2];
    int64_t q_3 = query[3];
    int64_t k_0 = key[0];
    int64_t k_1 = key[1];
    int64_t k_2 = key[2];
    int64_t k_3 = key[3];
    int64_t v_0 = value[0];
    int64_t v_1 = value[1];
    int64_t v_2 = value[2];
    int64_t v_3 = value[3];

    // for GQA and MQA
    if (input_layer_str == "SBH" || input_layer_str == "BSH" || input_layer_str == "BSND") {
        if (q_2 != k_2 && q_2!= v_2) {
            k_2 = q_2;
            v_2 = q_2;
        }
    } else {
        if (q_1 != k_1 && q_1!= v_1) {
            k_1 = q_1;
            v_1 = q_1;
        }
    }

    std::vector<int64_t> new_query_shape;
    std::vector<int64_t> new_key_shape;
    std::vector<int64_t> new_value_shape;
    std::vector<int64_t> new_grad_out_shape;
    if (input_layer_str == "BSH") {
        new_query_shape = {q_0, head_num, q_1, q_2/head_num};
        new_key_shape = {k_0, head_num, k_1, k_2/head_num};
        new_value_shape = {v_0, head_num, v_1, v_2/head_num};
    } else if (input_layer_str == "SBH") {
        new_query_shape = {q_1, head_num, q_0, q_2/head_num};
        new_key_shape = {k_1, head_num, k_0, k_2/head_num};
        new_value_shape = {v_1, head_num, v_0, v_2/head_num};
    } else if (input_layer_str == "BSND") {
        new_query_shape = {q_0, q_2, q_1, q_3};
        new_key_shape = {k_0, k_2, k_1, k_3};
        new_value_shape = {v_0, v_2, v_1, v_3};
    } else if (input_layer_str == "TND") {
        TORCH_CHECK(!cum_seq_q.empty(), "The actual_seq_qlen is not empty when TND");
        TORCH_CHECK(!cum_seq_k.empty(), "The actual_seq_kvlen is not empty when TND");
        TORCH_CHECK(cum_seq_q.size() == cum_seq_k.size(), "The size of actual_seq_qlen is equal actual_seq_kvlen when TND");

        int64_t b = cum_seq_q.size();
        new_query_shape = {b, q_1, q_0/b, q_2};
        new_key_shape = {b, k_1, k_0/b, k_2};
        new_value_shape = {b, v_1, v_0/b, v_2};
    }

    if (!grad_out.empty()) {
        new_grad_out_shape = new_query_shape;
    }
    result.emplace_back(new_query_shape, new_key_shape, new_value_shape, new_grad_out_shape);
    return result;
}

int64_t sdpa_flop_count(const std::vector<int64_t> query_shape, const std::vector<int64_t> key_shape, const std::vector<int64_t> value_shape)
{
    int64_t b, h, s_q, d_q;
    int64_t _b2, _h2, s_k, _d2;
    int64_t _b3, _h3, _s3, d_v;

    b = query_shape[0];
    h = query_shape[1];
    s_q = query_shape[2];
    d_q = query_shape[3];

    _b2 = key_shape[0];
    _h2 = key_shape[1];
    s_k = key_shape[2];
    _d2 = key_shape[3];

    _b3 = value_shape[0];
    _h3 = value_shape[1];
    _s3 = value_shape[2];
    d_v = value_shape[3];

    TORCH_CHECK(b == _b2 && b == _b3, "the dim of 0 is not equal between q and kv");
    TORCH_CHECK(h == _h2 && h == _h3, "the dim of 1 is not equal between q and kv");
    TORCH_CHECK(s_k == _s3, "the dim of 2 is not equal between k and v");
    TORCH_CHECK(d_q == _d2, "the dim of 3 is not equal between q and k");

    int64_t total_flops = 0;

    // q: [b, h, s_q, d_q] @ k: [b, h, d_q, s_k] -> scores: [b, h, s_q, s_k]
    total_flops += b * h * s_q * d_q * s_k * 2;

    // scores: [b, h, s_q, s_k] @ v: [b, h, s_k, d_v] -> out: [b, h, s_q, d_v]
    total_flops += b * h * s_q * s_k * d_v * 2;

    return total_flops;
}

int64_t sdpa_backward_flop_count(const std::vector<int64_t> query_shape, const std::vector<int64_t> key_shape, const std::vector<int64_t> value_shape, const std::vector<int64_t> grad_out_shape)
{
    int64_t b, h, s_q, d_q;
    int64_t _b2, _h2, s_k, _d2;
    int64_t _b3, _h3, _s3, d_v;
    int64_t _b4, _h4, _s4, d_4;

    b = query_shape[0];
    h = query_shape[1];
    s_q = query_shape[2];
    d_q = query_shape[3];

    _b2 = key_shape[0];
    _h2 = key_shape[1];
    s_k = key_shape[2];
    _d2 = key_shape[3];

    _b3 = value_shape[0];
    _h3 = value_shape[1];
    _s3 = value_shape[2];
    d_v = value_shape[3];

    _b4 = grad_out_shape[0];
    _h4 = grad_out_shape[1];
    _s4 = grad_out_shape[2];
    d_4 = grad_out_shape[3];

    TORCH_CHECK(b == _b2 && b == _b3 && b == _b4, "the dim of 0 is not equal between qkv and grad");
    TORCH_CHECK(h == _h2 && h == _h3 && h == _h4, "the dim of 1 is not equal between qkv and grad");
    TORCH_CHECK(s_k == _s3, "the dim of 2 is not equal between k and v");
    TORCH_CHECK(s_q == _s4, "the dim of 2 is not equal between q and grad");
    TORCH_CHECK(d_q == _d2, "the dim of 3 is not equal between q and k");
    TORCH_CHECK(d_v == d_4, "the dim of 3 is not equal between v and grad");

    int64_t total_flops = 0;

    // gradOut: [b, h, s_q, d_v] @ v: [b, h, d_v, s_k] -> gradScores: [b, h, s_q, s_k]
    total_flops += b * h * s_q * d_v * s_k * 2;

    // scores: [b, h, s_k, s_q] @ gradOut: [b, h, s_q, d_v] -> gradV: [b, h, s_k, d_v]
    total_flops += b * h * s_k * s_q * d_v * 2;

    // gradScores: [b, h, s_q, s_k] @ k: [b, h, s_k, d_q] -> gradQ: [b, h, s_q, d_q]
    total_flops += b * h * s_q * s_k * d_q * 2;

    // q: [b, h, d_q, s_q] @ gradScores: [b, h, s_q, s_k] -> gradK: [b, h, d_q, s_k]
    total_flops += b * h * d_q * s_q * s_k * 2;

    return total_flops;
}

int64_t FlopCounter::flash_attention_forward_flop(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, int64_t head_num,
    const std::string &input_layout, const c10::optional<std::vector<int64_t>> &actual_seq_qlen,
    const c10::optional<std::vector<int64_t>> &actual_seq_kvlen)
{
    std::vector<int64_t> grad_out_shape;
    std::vector<int64_t> query_shape(query.sizes().begin(), query.sizes().end());
    std::vector<int64_t> key_shape(key.sizes().begin(), key.sizes().end());
    std::vector<int64_t> value_shape(value.sizes().begin(), value.sizes().end());
    auto ac_seq_qlen_tmp = actual_seq_qlen.value_or(std::vector<int64_t>{});
    auto ac_seq_kvlen_tmp = actual_seq_kvlen.value_or(std::vector<int64_t>{});

    auto sizes = _unpack_flash_attention_nested_shapes(query_shape, key_shape, value_shape, head_num, grad_out_shape, ac_seq_qlen_tmp, ac_seq_kvlen_tmp, input_layout);

    int64_t total_flops = 0;
    for (const auto& [query_shape_new, key_shape_new, value_shape_new, _] : sizes) {
        total_flops += sdpa_flop_count(query_shape_new, key_shape_new, value_shape_new);
    }
    return total_flops;
}

int64_t FlopCounter::flash_attention_backward_flop(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &dy, int64_t head_num,
    const std::string &input_layout, const c10::optional<std::vector<int64_t>> &actual_seq_qlen,
    const c10::optional<std::vector<int64_t>> &actual_seq_kvlen)
{
    std::vector<int64_t> dy_shape(query.sizes().begin(), query.sizes().end());
    std::vector<int64_t> query_shape(query.sizes().begin(), query.sizes().end());
    std::vector<int64_t> key_shape(key.sizes().begin(), key.sizes().end());
    std::vector<int64_t> value_shape(value.sizes().begin(), value.sizes().end());
    auto ac_seq_qlen_tmp = actual_seq_qlen.value_or(std::vector<int64_t>{});
    auto ac_seq_kvlen_tmp = actual_seq_kvlen.value_or(std::vector<int64_t>{});

    auto sizes = _unpack_flash_attention_nested_shapes(query_shape, key_shape, value_shape, head_num, dy_shape, ac_seq_qlen_tmp, ac_seq_kvlen_tmp, input_layout);

    int64_t total_flops = 0;
    for (const auto& [query_shape_new, key_shape_new, value_shape_new, grad_out_shape] : sizes) {
        total_flops += sdpa_backward_flop_count(query_shape_new, key_shape_new, value_shape_new, grad_out_shape);
    }
    return total_flops;
}

int64_t FlopCounter::gmm_flop_int(const at::TensorList &x, const at::TensorList &weight, c10::optional<std::vector<int64_t>> group_list, int64_t group_type_value)
{
    int64_t total_flops = 0;

    std::vector<int64_t> x_shape(x[0].sizes().begin(), x[0].sizes().end());
    std::vector<int64_t> weight_shape(weight[0].sizes().begin(), weight[0].sizes().end());
    auto group_list_real_ = group_list.value_or(std::vector<int64_t>{});
    at::IntArrayRef group_list_real(group_list_real_);

    int64_t before_i = 0;

    if (group_type_value == 0) {
        for (int64_t i = 0; i < group_list_real.size(); i++) {
            int64_t after_i = group_list_real[i];
            total_flops += (after_i - before_i) * x_shape.back() * weight_shape.back() * 2;
            before_i = after_i;
        }
    }

    if (group_type_value == 2) {
        for (int64_t i = 0; i < group_list_real.size(); i++) {
            int64_t after_i = group_list_real[i];
            total_flops += x_shape.front() * (after_i - before_i) * weight_shape.back() * 2;
            before_i = after_i;
        }
    }

    return total_flops;
}

int64_t FlopCounter::gmm_flop_tensor(const at::TensorList &x, const at::TensorList &weight, const c10::optional<at::Tensor> &group_list, int64_t group_type_value)
{
    int64_t total_flops = 0;

    std::vector<int64_t> x_shape(x[0].sizes().begin(), x[0].sizes().end());
    std::vector<int64_t> weight_shape(weight[0].sizes().begin(), weight[0].sizes().end());
    auto group_list_real = group_list.value_or(at::Tensor());
    auto num_elements = group_list_real.numel();

    int64_t before_i = 0;

    if (group_type_value == 0) {
        for (int64_t i = 0; i < num_elements; i++) {
            int64_t after_i = group_list_real[i].item<int64_t>();
            total_flops += (after_i - before_i) * x_shape.back() * weight_shape.back() * 2;
            before_i = after_i;
        }
    }

    if (group_type_value == 2) {
        for (int64_t i = 0; i < num_elements; i++) {
            int64_t after_i = group_list_real[i].item<int64_t>();
            total_flops += x_shape.front() * (after_i - before_i) * weight_shape.back() * 2;
            before_i = after_i;
        }
    }

    return total_flops;
}

int64_t FlopCounter::gmm_add_flop(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &group_list)
{
    int64_t total_flops = 0;

    std::vector<int64_t> x_shape(x.sizes().begin(), x.sizes().end());
    std::vector<int64_t> weight_shape(weight.sizes().begin(), weight.sizes().end());
    auto num_elements = group_list.numel();

    int64_t before_i = 0;

    for (int64_t i = 0; i < num_elements; i++) {
            int64_t after_i = group_list[i].item<int64_t>();
            total_flops += x_shape.back() * (after_i - before_i) * weight_shape.back() * 2;
            before_i = after_i;
        }

    return total_flops;
}