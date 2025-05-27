# Copyright 2024 Huawei Technologies Co., Ltd

import os

import torch
from diffusers import FluxPipeline

DEVICE = "npu"  # Device name
MODEL_PATH = "./output_FLUX_dreambooth"  # Dreambooth微调保存模型路径
OUTPUT_PATH = "./infer_result_dreambooth"  # 输出保存路径

# 创建目录如果不存在
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 设置prompt，可自定义设置
PROMPT = [
    "masterpiece, best quality, a sks dog in a bucket, night, moonlight, dim lighting",
    "masterpiece, best quality, a dog is walking in beijing city, night, moonlight, dim lighting",
    "masterpiece, best quality, a dog is holding a sign that sys hello world, evening light, sunset, rim lighting",
    "masterpiece, best quality, three big dogs on a couch, flower in vase, film grain, crystal clear, dark studio",
    "masterpiece, best quality, 8 cats and 8 dogs on river, evening light, sunset, rim lighting, depth of field",
    "masterpiece, best quality, 2 dogs, Exquisite City, (sky:1.3), (Miniature tree:1.3), Miniature object, many flowers, glowing mushrooms, (creek:1.3), lots of fruits, cute colorful animal protagonist, Firefly, meteor, Colorful cloud, Complicated background, rainbow,",
    "masterpiece, best quality, a cat is holding a sign that says hello world, girl with glasses, (masterpiece, top quality, best quality, official art, beautiful and aesthetic:1.2),",
    "masterpiece, best quality, two dog with grey and yellow skin, mechanical arms, cyberpunk city background, night, neon light",
]

# 设置随机数种子
seed_list = [8, 23, 42, 1334]

# pipeline 设置
pipe = FluxPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True)
pipe = pipe.to(DEVICE)

# 输出图片
for prompt_key in PROMPT:
    for i in seed_list:
        generator = torch.Generator(device="cpu").manual_seed(i)
        image = pipe(
            prompt=prompt_key,
            num_inference_steps=28,
            height=1024,
            width=1024,
            guidance_scale=1.0,
            generator=generator,
        ).images
        image[0].save(f"{OUTPUT_PATH}/{prompt_key[28:40]}-{i}.png")
