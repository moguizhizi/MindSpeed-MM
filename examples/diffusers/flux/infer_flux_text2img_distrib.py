import os

import torch
from accelerate import PartialState
from diffusers import FluxPipeline

output_path = "./flux_lora_NPU"
os.makedirs(output_path, exist_ok=True)

MODEL_PATH = "/black-forest-labs/FLUX.1-dev"  # FLUX模型路径
LORA_WEIGHTS = "./output/pytorch_lora_weights.safetensors"  # LoRA权重路径
pipe = FluxPipeline.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True
)

if os.path.exists(LORA_WEIGHTS):
    print(f"Loading LoRA weights from {LORA_WEIGHTS}")
    pipe.load_lora_weights(LORA_WEIGHTS)
else:
    print("LoRA weights not found. Using the base model")

distributed_state = PartialState()
pipe.to(distributed_state.device)

PROMPTS = [
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

for i in seed_list:
    generator = torch.Generator(device="npu").manual_seed(i)

    with distributed_state.split_between_processes(PROMPTS) as prompts:
        for prompt in prompts:
            image = pipe(
                prompt=prompt,
                generator=generator,
                num_inference_steps=28,
                height=1024,
                width=1024,
                guidance_scale=1.0,
            ).images

            # Create name for the image
            prompt_words = prompt.replace("masterpiece, best quality, ", "").split()[:3]
            prompt_abbr = "_".join(prompt_words)

            filename = (
                f"{prompt_abbr}_seed{i}_rank{distributed_state.process_index}.png"
            )
            filename = "".join(
                c for c in filename if c.isalnum() or c in "._-"
            )  # remove special chars

            image[0].save(f"{output_path}/{filename}")
