import os
import math
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, TypedDict, Union
import av

import numpy as np
from typing_extensions import override

from PIL import Image
from PIL.Image import Image as ImageObject

VIDEO_PLACEHOLDER = os.getenv("VIDEO_PLACEHOLDER", "<video>")
IMAGE_PLACEHOLDER = os.getenv("IMAGE_PLACEHOLDER", "<image>")
IGNORE_INDEX = -100


@dataclass
class LengthParams:
    imglens: Sequence[int]
    vidlens: Sequence[int]
    seqlens: Sequence[int]


if TYPE_CHECKING:
    import torch
    from av.stream import Stream
    from transformers import ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor


    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]


    ImageInput = Union[str, EncodedImage, ImageObject]
    VideoInput = str


def _get_paligemma_token_type_ids(
        imglens: Sequence[int], seqlens: Sequence[int], processor: "ProcessorMixin"
) -> List[List[int]]:
    r"""
    Gets paligemma token type ids for computing loss.

    Returns:
        batch_token_type_ids: shape (batch_size, sequence_length)
    """
    batch_token_type_ids = []
    for imglen, seqlen in zip(imglens, seqlens):
        image_seqlen = imglen * getattr(processor, "image_seqlen")
        batch_token_type_ids.append([0] * image_seqlen + [1] * (seqlen - image_seqlen))

    return batch_token_type_ids


class BasePlugin:
    def __init__(self, image_token: Optional[str], video_token: Optional[str]) -> None:
        self.image_token = image_token
        self.video_token = video_token
        self.expand_mm_tokens = True

    def _validate_input(
            self,
            images: Sequence["ImageInput"],
            videos: Sequence["VideoInput"],
    ) -> None:
        r"""
        Validates if this model accepts the input modalities.
        """
        if len(images) != 0 and self.image_token is None:
            raise ValueError(
                "This model does not support image input. Please check whether the correct `template` is used."
            )

        if len(videos) != 0 and self.video_token is None:
            raise ValueError(
                "This model does not support video input. Please check whether the correct `template` is used."
            )

    @staticmethod
    def _preprocess_image(image: "ImageObject", **kwargs) -> "ImageObject":
        r"""
        Pre-processes a single image.
        """
        image_resolution: int = kwargs.get("image_resolution")
        if (image.width * image.height) > image_resolution:
            resize_factor = math.sqrt(image_resolution / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.Resampling.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    @staticmethod
    def _get_video_sample_frames(video_stream: "Stream", **kwargs) -> int:
        r"""
        Computes video sample frames according to fps.
        """
        video_fps: float = kwargs.get("video_fps")
        video_maxlen: int = kwargs.get("video_maxlen")
        total_frames = video_stream.frames
        sample_frames = float(video_stream.duration * video_stream.time_base) * video_fps
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return math.floor(sample_frames)

    def _regularize_images(self, images: Sequence["ImageInput"], **kwargs) -> List["ImageObject"]:
        r"""
        Regularizes images to avoid error. Including reading and pre-processing.
        """
        results = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])

            if not isinstance(image, ImageObject):
                raise ValueError(f"Expect input is a list of Images, but got {type(image)}.")

            results.append(self._preprocess_image(image, **kwargs))

        return results

    def _regularize_videos(self, videos: Sequence["VideoInput"], **kwargs) -> List[List["ImageObject"]]:
        r"""
        Regularizes videos to avoid error. Including reading, resizing and converting.
        """
        results = []
        for video in videos:
            with av.open(video, "r") as container:
                video_stream = next(stream for stream in container.streams if stream.type == "video")
                total_frames = video_stream.frames
                sample_frames = self._get_video_sample_frames(video_stream, **kwargs)
                sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
                frames: List["ImageObject"] = []
                container.seek(0)
                for frame_idx, frame in enumerate(container.decode(video_stream)):
                    if frame_idx in sample_indices:
                        frames.append(frame.to_image())
                frames = self._regularize_images(frames, **kwargs)
                results.append(frames)

        return results

    def _get_mm_inputs(
            self,
            images: Sequence["ImageInput"],
            videos: Sequence["VideoInput"],
            processor: "ProcessorMixin",
    ) -> Dict[str, "torch.Tensor"]:
        r"""
        Processes visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height

        It holds num_patches == torch.prod(image_grid_thw)
        """
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        video_processor: "BaseImageProcessor" = getattr(processor, "video_processor", image_processor)
        input_dict = {"images": None}  # default key
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_resolution=getattr(processor, "image_resolution", 768 * 768),
            )
            input_dict["images"] = images

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_resolution=getattr(processor, "video_resolution", 256 * 256),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            input_dict["videos"] = videos

        mm_inputs = {}
        if image_processor != video_processor:
            if input_dict.get("images") is not None:
                mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))
            if input_dict.get("videos") is not None:
                mm_inputs.update(video_processor(input_dict["videos"], return_tensors="pt"))
        elif input_dict.get("images") is not None or input_dict.get("videos") is not None:  # same processor (qwen2-vl)
            mm_inputs.update(image_processor(**input_dict, return_tensors="pt"))

        return mm_inputs

    def process_messages(
            self,
            messages: Sequence[Dict[str, str]],
            images: Sequence["ImageInput"],
            videos: Sequence["VideoInput"],
            processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        r"""
        Pre-processes input messages before tokenization for VLMs.
        """
        self._validate_input(images, videos)
        return messages

    def process_token_ids(
            self,
            input_ids: List[int],
            labels: Optional[List[int]],
            images: Sequence["ImageInput"],
            videos: Sequence["VideoInput"]
    ) -> Tuple[List[int], Optional[List[int]]]:
        r"""
        Pre-processes token ids after tokenization for VLMs.
        """
        self._validate_input(images, videos)
        return input_ids, labels

    def get_mm_inputs(
            self,
            images: Sequence["ImageInput"],
            videos: Sequence["VideoInput"],
            imglens: Sequence[int],
            vidlens: Sequence[int],
            batch_ids: Sequence[List[int]],
            processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        r"""
        Builds batched multimodal inputs for VLMs.

        Arguments:
            images: a list of image inputs, shape (num_images,)
            videos: a list of video inputs, shape (num_videos,)
            imglens: number of images in each sample, shape (batch_size,)
            vidlens: number of videos in each sample, shape (batch_size,)
            batch_ids: token ids of input samples, shape (batch_size, seq_len)
            processor: a processor for pre-processing images and videos
        """
        self._validate_input(images, videos)
        return {}


class Qwen2vlPlugin(BasePlugin):
    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.Resampling.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.Resampling.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.Resampling.NEAREST)

        return image

    @override
    def _regularize_videos(self, videos: Sequence["VideoInput"], **kwargs) -> List[List["ImageObject"]]:
        results = []
        for video in videos:
            container = av.open(video, "r")
            video_stream = next(stream for stream in container.streams if stream.type == "video")
            total_frames = video_stream.frames
            sample_frames = self._get_video_sample_frames(video_stream, **kwargs)
            sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
            frames: List["ImageObject"] = []
            container.seek(0)
            for frame_idx, frame in enumerate(container.decode(video_stream)):
                if frame_idx in sample_indices:
                    frames.append(frame.to_image())

            if len(frames) % 2 != 0:  # qwen2-vl requires even number of frames
                frames.append(frames[-1])

            frames = self._regularize_images(frames, **kwargs)
            results.append(frames)

        return results

    @override
    def process_messages(
            self,
            messages: Sequence[Dict[str, str]],
            images: Sequence["ImageInput"],
            videos: Sequence["VideoInput"],
            processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_length: int = getattr(image_processor, "merge_size") ** 2
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])

        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if num_image_tokens >= len(image_grid_thw):
                    raise ValueError(f"`len(images)` is less than the number of {IMAGE_PLACEHOLDER} tokens.")

                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER, f"<|vision_start|>{self.image_token * image_seqlen}<|vision_end|>", 1
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                if num_video_tokens >= len(video_grid_thw):
                    raise ValueError(f"`len(videos)` is less than the number of {VIDEO_PLACEHOLDER} tokens.")

                video_seqlen = video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    VIDEO_PLACEHOLDER, f"<|vision_start|>{self.video_token * video_seqlen}<|vision_end|>", 1
                )
                num_video_tokens += 1

            message["content"] = content

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        if len(videos) != num_video_tokens:
            raise ValueError(f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens.")

        return messages

    @override
    def get_mm_inputs(
            self,
            images: Sequence["ImageInput"],
            videos: Sequence["VideoInput"],
            params: LengthParams,
            processor: Optional["ProcessorMixin"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        self._validate_input(images, videos)
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        if "second_per_grid_ts" in getattr(image_processor, "model_input_names", []) and "video_grid_thw" in mm_inputs:
            video_fps = getattr(processor, "video_fps", 2.0)
            mm_inputs["second_per_grid_ts"] = [image_processor.temporal_patch_size / video_fps] * len(
                mm_inputs["video_grid_thw"]
            )

        return mm_inputs


PLUGINS = {
    "base": BasePlugin,
    "qwen2_vl": Qwen2vlPlugin,
}


def get_mm_plugin(
        name: str,
        image_token: Optional[str] = None,
        video_token: Optional[str] = None,
) -> "BasePlugin":
    plugin_class = PLUGINS.get(name, None)
    if plugin_class is None:
        raise ValueError("Multimodal plugin `{}` not found.".format(name))

    return plugin_class(image_token, video_token)
