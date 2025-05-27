/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MINDSPEED_OPS_CSRC_ATB_INC_ATB_ADAPTER_H
#define MINDSPEED_OPS_CSRC_ATB_INC_ATB_ADAPTER_H
#include <vector>
#include <string>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "atb/types.h"
#include "atb/operation.h"
#include "atb/utils.h"
#if __has_include("torch_npu/csrc/flopcount/FlopCount.h")
    #include "torch_npu/csrc/flopcount/FlopCount.h"
#endif

atb::Tensor AtTensor2Tensor(const at::Tensor atTensor);
atb::Context* GetContext();
at::Tensor GetWorkspaceTensor(uint64_t workspaceSize, atb::Operation *operation);
uint64_t OperationSetup(atb::VariantPack variantPack, atb::Operation *operation, atb::Context* contextPtr);
class ParamSetter {
public:
    ParamSetter& Input(const at::Tensor &tensor);
    ParamSetter& Input(const c10::optional<at::Tensor> &tensor);
    ParamSetter& Output(at::Tensor &tensor);
    atb::VariantPack variantPack;
};

void RunAtbCmd(atb::Operation *op, const ParamSetter &paramsetter, const std::string &name);

#endif
