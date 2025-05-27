import os
import torch
from transformers import PreTrainedTokenizer
from mindspeed_rl.utils.verl.verl_utils import compute_position_id_with_mask
from datasets import load_dataset
import mindspeed_rl.utils.verl.verl_utils as verl_F
import torchvision.transforms as transforms

from mindspeed_rl.datasets.utils import process_image
from mindspeed_rl.datasets.utils import get_rope_index
from mindspeed_rl.datasets.base_dataset import BaseDataset


class MultiModalDataset(BaseDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """
    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 processor=None,
                 prompt_key='prompt',
                 image_key='images',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 filter_overlong_prompts=False):
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer.tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        print(f"data_path:{data_path}")
        if os.path.isdir(data_path):
            # dataset = load_dataset("parquet", data_dir=data_path, split="train")
            dataset = load_dataset("arrow", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            dataset = load_dataset("parquet", data_files=data_path, split="train")
        else:
            dataset = load_dataset(data_path, split=data_split)

        super().__init__(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataset[index]
        chat = row_dict.pop(self.prompt_key)
        to_tensor = transforms.ToTensor()

        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        is_multi_modal = self.image_key in row_dict
        if is_multi_modal:  # expand image token
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(image, 2048*2048, 512*512) for image in row_dict.pop(self.image_key)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')

            flattened_images = []
            image_shapes = []
            for m in row_dict['multi_modal_data']['image']:
                image = to_tensor(m)
                image_shapes.append(torch.tensor(image.shape))
                image = image.reshape(1, -1)
                flattened_images.append(image)

            row_dict["image"] = torch.cat(flattened_images, dim=1)
            row_dict["image_shape"] = torch.stack(image_shapes, dim=0)

            # row_dict["labels"] = row_dict["extra_info"]["answer"]
            row_dict["labels"] = row_dict["answer"]

            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['pixel_values'] = image_inputs['pixel_values']
            row_dict['image_grid_thw'] = image_grid_thw
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1
                row_dict['image_num'] = index
                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        if is_multi_modal:
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict["input_ids_length"] = torch.tensor([attention_mask.sum().item()])
        row_dict['position_ids'] = position_ids
        row_dict['prompts'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
