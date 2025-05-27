import os
import random
from typing import Union
import copy

import torch

from mindspeed_mm.data.data_utils.constants import (
    CAPTIONS,
    FILE_INFO,
    IMG_FPS,
    PROMPT_IDS,
    PROMPT_MASK,
    TEXT,
    VIDEO,
    MASKED_VIDEO
)
from mindspeed_mm.data.data_utils.data_transform import (
    MaskGenerator,
    add_aesthetic_notice_image,
    add_aesthetic_notice_video,
)
from mindspeed_mm.data.data_utils.transform_pipeline import get_transforms
from mindspeed_mm.data.data_utils.utils import (
    VID_EXTENSIONS,
    DataSetProg,
    ImageProcesser,
    TextProcesser,
    VideoProcesser,
    VideoReader,
)
from mindspeed_mm.data.datasets.t2v_dataset import T2VDataset
from mindspeed_mm.utils.mask_utils import STR_TO_TYPE, MaskProcessor


def type_ratio_normalize(mask_type_ratio_dict):
    total = sum(mask_type_ratio_dict.values())
    length = len(mask_type_ratio_dict)
    if total == 0:
        return {k: 1.0 / length for k in mask_type_ratio_dict.keys()}
    return {k: v / total for k, v in mask_type_ratio_dict.items()}


I2VOutputData = {
    VIDEO: [],
    MASKED_VIDEO: [],
    TEXT: [],
    PROMPT_IDS: [],
    PROMPT_MASK: [],
}


class I2VDataset(T2VDataset):

    def __init__(
        self,
        basic_param: dict,
        vid_img_process: dict,
        use_text_processer: bool = False,
        use_clean_caption: bool = True,
        support_chinese: bool = False,
        tokenizer_config: Union[dict, None] = None,
        use_feature_data: bool = False,
        vid_img_fusion_by_splicing: bool = False,
        use_img_num: int = 0,
        use_img_from_vid: bool = True,
        mask_type_ratio_dict_video: Union[dict, None] = None,
        mask_type_ratio_dict_image: Union[dict, None] = None,
        default_text_ratio: float = 0.5,
        min_clear_ratio: float = 0.0,
        max_clear_ratio: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            basic_param=basic_param,
            vid_img_process=vid_img_process,
            use_text_processer=use_text_processer,
            use_clean_caption=use_clean_caption,
            support_chinese=support_chinese,
            tokenizer_config=tokenizer_config,
            use_feature_data=use_feature_data,
            vid_img_fusion_by_splicing=vid_img_fusion_by_splicing,
            use_img_num=use_img_num,
            use_img_from_vid=use_img_from_vid,
            **kwargs,
        )

        if self.num_frames != 1:
            self.mask_type_ratio_dict_video = mask_type_ratio_dict_video if mask_type_ratio_dict_video is not None else {
                'i2v': 1.0}
            self.mask_type_ratio_dict_video = {STR_TO_TYPE[k]: v for k, v in self.mask_type_ratio_dict_video.items()}
            self.mask_type_ratio_dict_video = type_ratio_normalize(self.mask_type_ratio_dict_video)

        self.mask_type_ratio_dict_image = mask_type_ratio_dict_image if mask_type_ratio_dict_image is not None else {
            'clear': 1.0}
        self.mask_type_ratio_dict_image = {STR_TO_TYPE[k]: v for k, v in self.mask_type_ratio_dict_image.items()}
        self.mask_type_ratio_dict_image = type_ratio_normalize(self.mask_type_ratio_dict_image)

        self.mask_processor = MaskProcessor(
            max_height=self.max_height,
            max_width=self.max_width,
            min_clear_ratio=min_clear_ratio,
            max_clear_ratio=max_clear_ratio,
        )

        self.train_pipeline_after_resize = vid_img_process.get("train_pipeline_after_resize", None)
        self.video_transforms_after_resize = get_transforms(is_video=True, train_pipeline=self.train_pipeline_after_resize)
        self.image_transforms_after_resize = get_transforms(
            is_video=False, train_pipeline=self.train_pipeline_after_resize
        )

        self.default_text_ratio = default_text_ratio

    def getitem(self, index):
        # init output data
        examples = copy.deepcopy(I2VOutputData)

        if self.data_storage_mode == "combine":
            examples = self.get_merge_data(examples, index)
        elif self.data_storage_mode == "standard":
            sample = self.data_samples[index]
            if self.use_feature_data:
                video_path, masked_video_path, text_path = sample[FILE_INFO], sample[MASKED_VIDEO], sample[CAPTIONS]
                if self.data_folder:
                    video_path = os.path.join(self.data_folder, video_path)
                    masked_video_path = os.path.join(self.data_folder, masked_video_path)
                    text_path = os.path.join(self.data_folder, text_path)
                    video_value = self.get_data_from_feature_data(video_path)
                    masked_video_value = self.get_data_from_feature_data(masked_video_path)
                    texts = self.get_data_from_feature_data(text_path)
                    examples[VIDEO] = video_value
                    examples[MASKED_VIDEO] = masked_video_value
                    examples[TEXT] = texts
                    examples[PROMPT_IDS] = texts
                    examples[PROMPT_MASK] = texts
            else:
                raise NotImplementedError(
                f"Not support now: data_storage_mode={self.data_storage_mode} and use_feature_data=false"
            )
        else:
            raise NotImplementedError(
                f"Not support now: data_storage_mode={self.data_storage_mode}."
            )
        return examples

    def drop(self, text, is_video=True):
        rand_num = random.random()
        rand_num_text = random.random()

        if rand_num < self.cfg:
            if rand_num_text < self.default_text_ratio:
                if not is_video:
                    text = "The image showcases a scene with coherent and clear visuals."
                else:
                    text = "The video showcases a scene with coherent and clear visuals."
            else:
                text = ''

        return dict(text=text)

    def get_merge_data(self, examples, index):
        sample = self.dataset_prog.cap_list[index]
        file_path = sample["path"]
        if not os.path.exists(file_path):
            raise AssertionError(f"file {file_path} do not exist!")
        file_type = self.get_type(file_path)
        if file_type == "video":
            frame_indice = sample["sample_frame_index"]
            vframes, _, is_decord_read = self.video_reader(file_path)
            start_frame_idx = sample.get("start_frame_idx", 0)
            clip_total_frames = sample.get("num_frames", -1)
            resolution_crop = tuple(sample.get("crop", (None, None, None, None)))
            video = self.video_processer(
                vframes,
                is_decord_read=is_decord_read,
                predefine_num_frames=len(frame_indice),
                start_frame_idx=start_frame_idx,
                clip_total_frames=clip_total_frames,
                resolution_crop=resolution_crop
            )
            video = video.permute(1, 0, 2, 3)
            inpaint_cond_data = self.mask_processor(video, mask_type_ratio_dict=self.mask_type_ratio_dict_video)
            mask, masked_video = inpaint_cond_data['mask'], inpaint_cond_data['masked_pixel_values']

            video = self.video_transforms_after_resize(video)  # T C H W -> T C H W
            masked_video = self.video_transforms_after_resize(masked_video)  # T C H W -> T C H W

            video = torch.cat([video, masked_video, mask], dim=1)  # T 2C+1 H W

            video = video.transpose(0, 1)  # T C H W -> C T H W

            examples[VIDEO] = video
        elif file_type == "image":
            image = self.image_processer(file_path)
            image = image.transpose(0, 1)
            inpaint_cond_data = self.mask_processor(image, mask_type_ratio_dict=self.mask_type_ratio_dict_image)
            mask, masked_image = inpaint_cond_data['mask'], inpaint_cond_data['masked_pixel_values']

            image = self.image_transforms_after_resize(image)
            masked_image = self.image_transforms_after_resize(masked_image)

            image = torch.cat([image, masked_image, mask], dim=1)  # [1 2C+1 H W]

            image = image.transpose(0, 1)  # [1 C H W] -> [C 1 H W]
            examples[VIDEO] = image

        text = sample["cap"]
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]
        if self.use_aesthetic:
            if sample.get('aesthetic', None) is not None or sample.get('aes', None) is not None:
                aes = sample.get('aesthetic', None) or sample.get('aes', None)
                if file_type == "video":
                    text = [add_aesthetic_notice_video(text[0], aes)]
                elif file_type == "image":
                    text = [add_aesthetic_notice_image(text[0], aes)]
        prompt_ids, prompt_mask = self.get_text_processer(text)
        examples[PROMPT_IDS], examples[PROMPT_MASK] = prompt_ids, prompt_mask
        return examples