# some constants used for dataset
PROMPT_MASK = "prompt_mask"
PROMPT_IDS = "prompt_ids"
PROMPT_MASK_2 = "prompt_mask_2"
PROMPT_IDS_2 = "prompt_ids_2"
TEXT = "text"
VIDEO = "video"
VIDEO_REJECTED = "video_rejected"
PROMPT = "prompt"
LATENTS = "latents"
VIDEO_MASK = "video_mask"
MASKED_VIDEO = "masked_video"
INPUT_MASK = "input_mask"
FILE_INFO = "file"
FILE_REJECTED_INFO = "file_rejected"
CAPTIONS = "captions"
SCORE = "score"
SCORE_REJECTED = "score_rejected"
IMG_FPS = 120
SORA_MODEL_PROTECTED_KEYS = [
    PROMPT_MASK,
    PROMPT_IDS,
    PROMPT,
    TEXT,
    VIDEO,
    VIDEO_MASK,
    LATENTS,
]
MODEL_CONSTANTS = {
    'llava': {
        "IMAGE_TOKEN": "<image>",
        "IGNORE_INDEX": -100,
        "IMAGE_TOKEN_INDEX": -200,
        "IMG_START_TOKEN": "<im_start>",
        "IMG_END_TOKEN": "<im_end>",
        "IMAGE_PATCH_TOKEN": "<im_patch>"
    },
    'internvl': {
        "IMG_CONTEXT_TOKEN": "<IMG_CONTEXT>",
        "IGNORE_INDEX": -100,
        "IMG_START_TOKEN": "<img>",
        "IMG_END_TOKEN": "</img>"
    },
    'internvl2_5': {
        'IMG_CONTEXT_TOKEN': '<IMG_CONTEXT>',
        'IMG_START_TOKEN': '<img>',
        'IMG_END_TOKEN': '</img>',
        'QUAD_START_TOKEN': '<quad>',
        'QUAD_END_TOKEN': '</quad>',
        'REF_START_TOKEN': '<ref>',
        'REF_END_TOKEN': '</ref>',
        'BOX_START_TOKEN': '<box>',
        'BOX_END_TOKEN': '</box>',
        "IGNORE_INDEX": -100
    },
    "deepseekvl2": {
        "IGNORE_INDEX": -100,
        "IMAGE_TOKEN": "<image>"
    }
}