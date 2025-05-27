# Copyright 2024 Huawei Technologies Co., Ltd
# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
# Copyright 2024 Stability AI and The HuggingFace Team

import os

import torch
from diffusers import StableDiffusion3Pipeline

output_path = "./infer_result_lora"
os.makedirs(output_path, exist_ok=True)

MODEL_PATH = "stabilityai/stable-diffusion-3.5-large"  # 模型路径
DEVICE = "npu"
LORA_WEIGHTS = "./output/pytorch_lora_weights.safetensors"  # LoRA权重路径
DTYPE = torch.float16  # 混精模式

pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    local_files_only=True,
)
pipe = pipe.to(DEVICE)
pipe.load_lora_weights(LORA_WEIGHTS)

prompts = dict()
prompts["masterpiece, best quality, Cute dragon creature, pokemon style, night, moonlight, dim lighting"] = "deformed, disfigured, underexposed, overexposed, rugged, (low quality), (normal quality),"
prompts["masterpiece, best quality, Pikachu walking in beijing city, pokemon style, night, moonlight, dim lighting"] = "deformed, disfigured, underexposed, overexposed, (low quality), (normal quality),"
prompts["masterpiece, best quality, red panda , pokemon style, evening light, sunset, rim lighting"] = "deformed, disfigured, underexposed, overexposed, (low quality), (normal quality),"
prompts["masterpiece, best quality, Photo of (Lion:1.2) on a couch, flower in vase, dof, film grain, crystal clear, pokemon style, dark studio"] = "deformed, disfigured, underexposed, overexposed, "
prompts["masterpiece, best quality, siberian cat pokemon on river, pokemon style, evening light, sunset, rim lighting, depth of field"] = "deformed, disfigured, underexposed, overexposed, "
prompts["masterpiece, best quality, pig, Exquisite City, (sky:1.3), (Miniature tree:1.3), Miniature object, many flowers, glowing mushrooms, (creek:1.3), lots of fruits, cute colorful animal protagonist, Firefly, meteor, Colorful cloud, pokemon style, Complicated background, rainbow,"] = "Void background,black background,deformed, disfigured, underexposed, overexposed, "
prompts["masterpiece, best quality, (pokemon), a cute pikachu, girl with glasses, (masterpiece, top quality, best quality, official art, beautiful and aesthetic:1.2),"] = "(low quality), (normal quality), (monochrome), lowres, extra fingers, fewer fingers, (watermark), "
prompts["masterpiece, best quality, sugimori ken \(style\), (pokemon \(creature\)), pokemon electric type, grey and yellow skin, mechanical arms, cyberpunk city background, night, neon light"] = "(worst quality, low quality:1.4), watermark, signature, deformed, disfigured, underexposed, overexposed, "

# 设置随机数种子
seed_list = [8, 23, 42, 1334]

# 输出图片
for prompt_key, negative_prompt_key in prompts.items():
    for i in seed_list:
        generator = torch.Generator(device="cpu").manual_seed(i)
        image = pipe(
            prompt=prompt_key,
            negative_prompt=negative_prompt_key,
            generator=generator,
            num_inference_steps=28,
            height=1024,
            width=1024,
            guidance_scale=1.0,
        ).images
        image[0].save(f"{output_path}/{prompt_key[26:40]}-{i}.png")
