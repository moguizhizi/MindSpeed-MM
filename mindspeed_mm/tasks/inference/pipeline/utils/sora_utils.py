import os
import math

import torch
import numpy as np
from diffusers.utils import load_image
from PIL import Image
from torchvision.io import write_video
from torchvision.io import write_png
from einops import rearrange
import imageio

try:
    import decord
except ImportError:
    print("Failed to import decord module.")

from torchvision.transforms import Compose, Lambda

from mindspeed_mm.data.data_utils.data_transform import CenterCropResizeVideo, SpatialStrideCropVideo, \
    ToTensorAfterResize, maxhwresize
from mindspeed_mm.utils.mask_utils import STR_TO_TYPE, TYPE_TO_STR, MaskType


def save_videos(videos, start_index, save_path, fps):
    os.makedirs(save_path, exist_ok=True)
    if isinstance(videos, (list, tuple)) or videos.ndim == 5:  # [b, t, h, w, c]
        for i, video in enumerate(videos):
            save_path_i = os.path.join(save_path, f"video_{start_index + i}.mp4")
            imageio.mimwrite(save_path_i, video, fps=fps, quality=6)
    elif videos.ndim == 4:
        save_path = os.path.join(save_path, f"video_{start_index}.mp4")
        imageio.mimwrite(save_path, video, fps=fps, quality=6)
    else:
        raise ValueError("The video must be in either [b, t, h, w, c] or [t, h, w, c] format.")


def save_image_or_videos(videos, save_path, start_idx, fps, value_range=(-1, 1), normalize=True):
    os.makedirs(save_path, exist_ok=True)
    if isinstance(videos, (list, tuple)) or videos.ndim == 5:  # b,c,t,h,w
        for i, video in enumerate(videos):
            if video.shape[1] == 1:
                save_path_i = os.path.join(save_path, str(i + start_idx) + ".png")
                imageio.imwrite(save_path_i, video[:, 0])
            else:
                save_path_i = os.path.join(save_path, str(i + start_idx) + ".mp4")
                imageio.mimwrite(save_path_i, video, fps=fps, quality=6)
    elif videos.ndim == 4:
        _save_video(videos, os.path.join(save_path, "0" + ".mp4"), fps, value_range, normalize)
    else:
        raise ValueError("The video must be in either [b,c,t,h,w] or [c,t,h,w] format.")


def save_video_grid(videos, save_path, fps, nrow=None):
    b, t, h, w, c = videos.shape
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = torch.zeros(
        (
            t,
            (padding + h) * nrow + padding,
            (padding + w) * ncol + padding,
            c
        ),
        dtype=torch.uint8
    )

    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r: start_r + h, start_c: start_c + w] = videos[i]

    imageio.mimwrite(os.path.join(save_path, "video_grid.mp4"), video_grid, fps=fps, quality=6)


def load_prompts(prompt):
    if os.path.exists(prompt):
        with open(prompt, "r") as f:
            lines = f.readlines()
            if len(lines) > 100:
                print("The file has more than 100 lines of prompts, we can only proceed the first 100")
                lines = lines[:100]
            prompts = [line.strip() for line in lines]
        return prompts
    else:
        return [prompt]


def safe_load_image(path):
    # safe load the image to check the image size (<=100M)
    file_size = os.path.getsize(path)
    if file_size > 100 * 1024 * 1024:
        raise ValueError("The image has to be less than 100M")
    else:
        return load_image(path)


def load_images(image=None):
    if image is None:
        print("The input image is None, execute text to video task")
        return None

    if os.path.exists(image):
        if os.path.splitext(image)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]:
            return [safe_load_image(image)]
        else:
            with open(image, "r") as f:
                lines = f.readlines()
                if len(lines) > 100:
                    print("The file has more than 100 lines of images, we can only process the first 100")
                    lines = lines[:100]
                images = [safe_load_image(line.strip()) for line in lines]
            return images
    else:
        raise FileNotFoundError(f"The image path {image} does not exist")


def load_videos(video=None, start_frame=0, num_frames=None):
    if video is None:
        print("The input video is None, execute text to video task")
        return None
    if os.path.exists(video):
        if is_video_file(video):
            return [open_video(video, start_frame, num_frames)]
        else:
            with open(video, "r") as f:
                lines = f.readlines()
                if len(lines) > 100:
                    print("The file has more than 100 lines of videos, we can only process the first 100")
                    lines = lines[:100]
                videos = []
                for line in lines:
                    video_path = line.strip()
                    if os.path.exists(video_path) and is_video_file(video_path):
                        videos.append(open_video(video_path, start_frame, num_frames))
                    else:
                        print(f"The path {video_path} does not exist or is not a valid video file")
            return videos
    else:
        raise FileNotFoundError(f"The video path {video} does not exist")


def load_conditional_pixel_values(conditional_pixel_values_path):
    if os.path.exists(conditional_pixel_values_path):
        with open(conditional_pixel_values_path, "r") as f:
            lines = f.readlines()
            if len(lines) > 100:
                print("The file has more than 100 lines of images, we can only proceed the first 100")
                lines = lines[:100]
            conditional_pixel_values = [line.strip().split(",") for line in lines]
        return conditional_pixel_values
    else:
        return [conditional_pixel_values_path]


def is_video_file(file_path):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg', '.3gp'}
    file_extension = os.path.splitext(file_path)[1].lower()
    return file_extension in video_extensions


def is_image_file(file_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    file_extension = os.path.splitext(file_path)[1].lower()
    return file_extension in image_extensions


def open_video(file_path, start_frame_idx, num_frames, frame_interval=1):
    decord_vr = decord.VideoReader(file_path, ctx=decord.cpu(0), num_threads=1)

    total_frames = len(decord_vr)
    frame_indices = list(
        range(start_frame_idx, min(start_frame_idx + num_frames * frame_interval, total_frames), frame_interval))

    if len(frame_indices) == 0:
        raise ValueError("No frames selected. Check your start_frame_idx and num_frames.")

    if len(frame_indices) < num_frames:
        raise ValueError(
            f"Requested {num_frames} frames but only {len(frame_indices)} frames are available, please adjust the start_frame_idx and num_frames or decrease the frame_interval.")

    if len(frame_indices) > 1000:
        raise ValueError("Frames has to be less than or equal to 1000")

    video_data = decord_vr.get_batch(frame_indices).asnumpy()
    video_data = torch.from_numpy(video_data)
    video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
    return video_data


def get_resize_transform(
        ori_height,
        ori_width,
        height=None,
        width=None,
        crop_for_hw=False,
        hw_stride=32,
        max_hxw=236544,  # 480 x 480
):
    if crop_for_hw:
        transform = CenterCropResizeVideo(transform_size=dict(max_height=height, max_width=width))
    else:
        new_height, new_width = maxhwresize(ori_height, ori_width, max_hxw)
        transform = Compose(
            [
                CenterCropResizeVideo(transform_size=dict(max_height=new_height, max_width=new_width)),
                # We use CenterCropResizeVideo to share the same height and width, ensuring that the shape of the crop remains consistent when multiple images are captured
                SpatialStrideCropVideo(stride=hw_stride),
            ]
        )
    return transform


def get_video_transform():
    norm_fun = Lambda(lambda x: 2. * x - 1.)
    transform = Compose([
        ToTensorAfterResize(),
        norm_fun
    ])
    return transform


def get_pixel_values(file_path, num_frames):
    if is_image_file(file_path[0]):
        pixel_values = [safe_load_image(path) for path in file_path]
        pixel_values = [torch.from_numpy(np.array(image)) for image in pixel_values]
        pixel_values = [rearrange(image, 'h w c -> c h w').unsqueeze(0) for image in pixel_values]
    elif is_video_file(file_path[0]):
        pixel_values = [open_video(video_path, 0, num_frames) for video_path in file_path]
    return pixel_values


def get_mask_type_cond_indices(mask_type, conditional_pixel_values_path, conditional_pixel_values_indices,
                               num_frames):
    if mask_type is not None and mask_type in STR_TO_TYPE.keys():
        mask_type = STR_TO_TYPE[mask_type]
    if is_image_file(conditional_pixel_values_path[0]):
        if len(conditional_pixel_values_path) == 1:
            mask_type = MaskType.i2v if mask_type is None else mask_type
            if num_frames > 1:
                conditional_pixel_values_indices = [
                    0] if conditional_pixel_values_indices is None else conditional_pixel_values_indices
        elif len(conditional_pixel_values_path) == 2:
            mask_type = MaskType.transition if mask_type is None else mask_type
            if num_frames > 1:
                conditional_pixel_values_indices = [0,
                                                    -1] if conditional_pixel_values_indices is None else conditional_pixel_values_indices
        else:
            if num_frames > 1:
                mask_type = MaskType.random_temporal if mask_type is None else mask_type
    elif is_video_file(conditional_pixel_values_path[0]):
        # When the input is a video, video continuation is executed by default, with a continuation rate of double
        mask_type = MaskType.continuation if mask_type is None else mask_type
    return mask_type, conditional_pixel_values_indices
