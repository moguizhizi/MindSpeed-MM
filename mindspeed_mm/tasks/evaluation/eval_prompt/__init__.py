from .build_prompt_llava import LlavaPromptTemplate
from .build_prompt_internvl import InternvlPromptTemplate
from .build_prompt_qwen2vl import Qwen2vlPromptTemplate

eval_model_prompt_dict = {
    "llava_v1.5_7b": LlavaPromptTemplate,
    "internvl2_8b": InternvlPromptTemplate,
    "qwen2_vl_7b": Qwen2vlPromptTemplate
    }