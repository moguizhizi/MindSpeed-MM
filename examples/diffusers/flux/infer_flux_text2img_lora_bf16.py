import os

import torch
from diffusers import AutoPipelineForText2Image

output_path = "./flux_lora_NPU"
os.makedirs(output_path, exist_ok=True)
DEVICE = "npu"

MODEL_PATH = "/flux/model"  # FLUX模型路径
LORA_WEIGHTS = "./output/pytorch_lora_weights.safetensors"  # LoRA权重路径
pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True
)
pipe = pipe.to(DEVICE)

pipe.load_lora_weights(LORA_WEIGHTS)
prompts = [
    "masterpiece, best quality, Cute dragon creature, pokemon style, night, moonlight, dim lighting",
    "masterpiece, best quality, Pikachu walking in beijing city, pokemon style, night, moonlight, dim lighting",
    "masterpiece, best quality, red panda , pokemon style, evening light, sunset, rim lighting",
    "masterpiece, best quality, Photo of (Lion:1.2) on a couch, flower in vase, dof, film grain, crystal clear, pokemon style, dark studio",
    "masterpiece, best quality, siberian cat pokemon on river, pokemon style, evening light, sunset, rim lighting, depth of field",
    "masterpiece, best quality, pig, Exquisite City, (sky:1.3), (Miniature tree:1.3), Miniature object, many flowers, glowing mushrooms, (creek:1.3), lots of fruits, cute colorful animal protagonist, Firefly, meteor, Colorful cloud, pokemon style, Complicated background, rainbow,",
    "masterpiece, best quality, (pokemon), a cute pikachu, girl with glasses, (masterpiece, top quality, best quality, official art, beautiful and aesthetic:1.2),",
    "masterpiece, best quality, sugimori ken \(style\), (pokemon \(creature\)), pokemon electric type, grey and yellow skin, mechanical arms, cyberpunk city background, night, neon light",
]
# 设置随机数种子
seed_list = [8, 23, 42, 1334]
for prompt_key in prompts:
    for i in seed_list:
        generator = torch.Generator(device="cpu").manual_seed(i)
        image = pipe(
            prompt=prompt_key, generator=generator, num_inference_steps=25
        ).images
        image[0].save(f"{output_path}/{prompt_key[26:40]}-{i}.png")
