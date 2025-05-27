import os

import torch
from accelerate import PartialState
from diffusers import SanaPipeline

MODEL_PATH = "Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers"  # SANA模型权重
OUTPUT_PATH = "./infer_result"  # 输出保存路径
LORA_WEIGHTS = "./output/pytorch_lora_weights.safetensors"  # LoRA权重路径
DEVICE = "npu"

os.makedirs(OUTPUT_PATH, exist_ok=True)  # 创建推理保存目录

pipe = SanaPipeline.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True
)

pipe.transformer.to(torch.bfloat16)
for block in pipe.transformer.transformer_blocks:
    block.attn2.set_use_npu_flash_attention(True)
pipe.text_encoder.to(torch.bfloat16)
pipe.vae.to(torch.float32)

if pipe.transformer.config.sample_size == 128:
    pipe.vae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024)

if os.path.exists(LORA_WEIGHTS):  # 加载Lora权重
    print(f"Loading LoRA weights from {LORA_WEIGHTS}")
    pipe.load_lora_weights(LORA_WEIGHTS)
else:
    print("LoRA weights not found. Using the base model")

distributed_state = PartialState()
pipe.to(distributed_state.device)

PROMPTS = dict()
PROMPTS = {
    "masterpiece, best quality, Cute dragon creature, pokemon style, night, moonlight, dim lighting": "deformed, disfigured, underexposed, overexposed, rugged, (low quality), (normal quality),",
    "masterpiece, best quality, Pikachu walking in beijing city, pokemon style, night, moonlight, dim lighting": "deformed, disfigured, underexposed, overexposed, (low quality), (normal quality),",
    "masterpiece, best quality, red panda , pokemon style, evening light, sunset, rim lighting": "deformed, disfigured, underexposed, overexposed, (low quality), (normal quality),",
    "masterpiece, best quality, Photo of (Lion:1.2) on a couch, flower in vase, dof, film grain, crystal clear, pokemon style, dark studio": "deformed, disfigured, underexposed, overexposed, (low quality), (normal quality),",
    "masterpiece, best quality, siberian cat pokemon on river, pokemon style, evening light, sunset, rim lighting, depth of field": "deformed, disfigured, underexposed, overexposed,",
    "masterpiece, best quality, pig, Exquisite City, (sky:1.3), (Miniature tree:1.3), Miniature object, many flowers, glowing mushrooms, (creek:1.3), lots of fruits, cute colorful animal protagonist, Firefly, meteor, Colorful cloud, pokemon style, Complicated background, rainbow,": "Void background,black background,deformed, disfigured, underexposed, overexposed,",
    "masterpiece, best quality, (pokemon), a cute pikachu, girl with glasses, (masterpiece, top quality, best quality, official art, beautiful and aesthetic:1.2),": "(low quality), (normal quality), (monochrome), lowres, extra fingers, fewer fingers, (watermark),",
    "masterpiece, best quality, sugimori ken \(style\), (pokemon \(creature\)), pokemon electric type, grey and yellow skin, mechanical arms, cyberpunk city background, night, neon light": "(worst quality, low quality:1.4), watermark, signature, deformed, disfigured, underexposed, overexposed,",
}

# 设置随机数种子
seed_list = [8, 23, 42, 1334]

for i in seed_list:
    generator = torch.Generator(device="cpu").manual_seed(i)

    # convert dictionary to list
    prompt_list = list(PROMPTS.keys())
    negative_prompt_list = list(PROMPTS.values())

    with distributed_state.split_between_processes(
        list(zip(prompt_list, negative_prompt_list))
    ) as distributed_pairs:
        for prompt, negative_prompt in distributed_pairs:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=generator,
                num_inference_steps=28,
                height=4096,
                width=4096,
                guidance_scale=2.0,
            ).images

        # Create name for the image
        prompt_words = prompt.replace("masterpiece, best quality, ", "").split()[:3]
        prompt_abbr = "_".join(prompt_words)

        filename = f"{prompt_abbr}_seed{i}_rank{distributed_state.process_index}.png"
        filename = "".join(
            c for c in filename if c.isalnum() or c in "._-"
        )  # remove special chars

        image[0].save(f"{OUTPUT_PATH}/{filename}")
