import vllm

from mindspeed_rl.vllm_patch.qwen2_5_vl_visionmlp_patch import replace_with_npu_qwen2_5_visionmlp
from mindspeed_rl.vllm_patch.qwen2_5_vl_image_emb_patch import replace_with_npu_qwen2_5_image_emb, image_emb_reuse
from mindspeed_rl.vllm_patch.qwen2_5_vl_rope_sin_cos_patch import replace_with_npu_qwen2_5_rope_sin_cos
from mindspeed_rl.vllm_patch.set_device import set_device_offset_patch
from mindspeed_rl.vllm_patch.rotary_embedding_patch import MRotaryEmbedding_forward_patch

replace_with_npu_qwen2_5_visionmlp()
set_device_offset_patch()

# MRotaryEmbedding_forward_patch()

if vllm.__version__ == "0.7.3":
    replace_with_npu_qwen2_5_image_emb()
    replace_with_npu_qwen2_5_rope_sin_cos()
elif vllm.__version__ in ["0.8.5.post1", "0.8.5.post1+empty"]:
    image_emb_reuse()