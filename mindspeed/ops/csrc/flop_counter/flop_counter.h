/******************************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd
 * All rights reserved.
 *
 * Licensed under the BSD 3-Clause License  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef FLOP_COUNTER_MS_H
#define FLOP_COUNTER_MS_H

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

class FlopCounter {
public:
    FlopCounter() = default;
    ~FlopCounter() = default;

    static int64_t mm_flop(const at::Tensor &tensor1, const at::Tensor &tensor2);
    static int64_t coc_flop(const at::Tensor &tensor1, const at::Tensor &tensor2, bool trans, int rankSize, bool is_ag_mm);
    static int64_t bmm_flop(const at::Tensor &self, const at::Tensor &mat2);
    static int64_t flash_attention_forward_flop(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
        int64_t head_num, const std::string &input_layout, const c10::optional<std::vector<int64_t>> &actual_seq_qlen,
        const c10::optional<std::vector<int64_t>> &actual_seq_kvlen);
    static int64_t flash_attention_backward_flop(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
        const at::Tensor &dy, int64_t head_num, const std::string &input_layout,
        const c10::optional<std::vector<int64_t>> &actual_seq_qlen,
        const c10::optional<std::vector<int64_t>> &actual_seq_kvlen);
    static int64_t gmm_flop_int(const at::TensorList &x, const at::TensorList &weight, c10::optional<std::vector<int64_t>> group_list, int64_t group_type_value);
    static int64_t gmm_flop_tensor(const at::TensorList &x, const at::TensorList &weight, const c10::optional<at::Tensor> &group_list, int64_t group_type_value);
    static int64_t gmm_add_flop(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &group_list);
};

#endif // FLOP_COUNTER_MS_H