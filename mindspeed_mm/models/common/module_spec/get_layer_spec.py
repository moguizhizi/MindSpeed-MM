from mindspeed_mm.models.common.module_spec.qwen2vl_layer_spec import get_qwen2vl_layer_spec, get_mlp_module_spec, get_qwen2vlllm_layer_local_spec
from mindspeed_mm.models.common.module_spec.internvl_layer_spec import get_language_layer_spec, get_vit_layer_spec
from mindspeed_mm.models.common.module_spec.llava_layer_spec import get_layer_spec
from mindspeed_mm.models.common.module_spec.minicpm_layer_spec import get_minicpm_vit_layer_spec
from mindspeed_mm.models.common.module_spec.deepseekvl_layer_spec import get_deepseekvl_model_spec

vit_layer_specs = {'qwen2vit': get_qwen2vl_layer_spec,
                   'InternViT': get_vit_layer_spec,
                   'clip': get_layer_spec,
                   'MiniCPMViT': get_minicpm_vit_layer_spec}
llm_layer_specs = {'qwen2lm': get_qwen2vlllm_layer_local_spec,
                   'qwen2_5_lm': get_qwen2vlllm_layer_local_spec,
                   'internllm': get_language_layer_spec,
                   'llava': get_layer_spec,
                   'deepseek': get_deepseekvl_model_spec}
projector_layer_specs = {'lnmlp': get_mlp_module_spec, 'mlp': get_mlp_module_spec}


def get_vit_layer_spec(config, *args, **kwargs):
    if getattr(config, 'model_id', None) is not None:
        if config.model_id in vit_layer_specs:
            return vit_layer_specs[config.model_id](config, is_vit=True)
    return None


def get_llm_layer_spec(config, *args, **kwargs):
    if getattr(config, 'model_id', None) is not None:
        if config.model_id in llm_layer_specs:
            return llm_layer_specs[config.model_id](config, is_vit=False)
    return None


def get_projector_layer_spec(config, *args, **kwargs):
    if getattr(config, 'model_id', None) is not None:
        if config.model_id in projector_layer_specs:
            return projector_layer_specs[config.model_id](config, use_te=False).submodules
    return None