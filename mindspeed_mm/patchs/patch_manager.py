# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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

from megatron.training import get_args
from mindspeed.optimizer.adamw import AdamW
from mindspeed.patch_utils import MindSpeedPatchesManager as pm

from mindspeed_mm.patchs import diffusers_patches
from mindspeed_mm.patchs import transformers_patches
from mindspeed_mm.patchs import models_patches


class PatchesManager:
    configs = {
        "t5_attention": [("transformers.models.t5.modeling_t5.T5Attention.forward", transformers_patches.t5_forward)],
        "t5_layer_norm": [("transformers.models.t5.modeling_t5.T5LayerNorm", transformers_patches.NpuRMSNorm)],
        "t5_gelu": [("transformers.activations.NewGELUActivation", transformers_patches.ApproximateGELU)],
        "diffusers_geglu": [("diffusers.models.activations.GEGLU.forward", diffusers_patches.geglu_forward)],
        "torch_adamw": [("torch.optim.AdamW", AdamW)],
        "ae_float32": [
            ("megatron.legacy.model.module.Float16Module.__init__", models_patches.float16Module_init),
            ("megatron.legacy.model.module.Float16Module.forward", models_patches.float16Module_forward)
        ],
        "moe_mlp": [
            ("megatron.core.transformer.moe.experts.SequentialMLP.forward", models_patches.SequentialMLP_forward)
        ]
    }

    @staticmethod
    def register_patch(orig_func_name, new_func=None):
        pm.register_patch(orig_func_name, new_func, force_patch=True)

    @staticmethod
    def apply_patches():
        pm.apply_patches()

    @staticmethod
    def apply_patches_from_config():
        cfg = get_args().mm.model
        if hasattr(cfg, "patch"):
            cfg = cfg.patch.to_dict()
            for key in cfg.keys() & PatchesManager.configs.keys():
                if not cfg.get(key):
                    continue
                for orig_func_name, new_func in PatchesManager.configs[key]:
                    PatchesManager.register_patch(orig_func_name, new_func)
        PatchesManager.apply_patches()
