# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import copy
import time
import threading
from abc import ABC
from typing import List, Dict, Union, Optional, Tuple
from operator import itemgetter

import ray
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from mindspeed_rl.utils.loggers import Loggers

logger = Loggers("transfer_dock")


class TimeoutException(Exception):
    """Custom Timeout Exception"""

    def __init__(self, message="TIMEOUT: Time Sleep Too Long"):
        super().__init__(message)


class TransferDock(ABC):
    """
    TransferDock is a data structure class that serves as the base class for GRPOTransferDock,
    providing data storage and retrieval functions.
    """

    def __init__(
        self,
        prompts_num: int,
        n_samples_per_prompt: int,
        experience_columns: Union[List[str], None],
        timeout: Union[int, None],
        timeout_interval: Union[int, None],
    ) -> None:
        """TransferDock initialize.

        Args:
            prompts_num: The number of prompts loaded from the dataset.
            n_samples_per_prompt: The number of responses for a prompt.
            experience_columns: Data columns in TransferDock.
            timeout: The waiting time for over time printing
            timeout_interval: Time interval for timeout printing
        """
        super().__init__()

        self.prompts_num = prompts_num
        self.n_samples_per_prompt = n_samples_per_prompt
        self.max_len = prompts_num * n_samples_per_prompt

        self.experience_columns = (
            experience_columns if experience_columns is not None else []
        )
        self.experience_data = {
            key: [None for _ in range(self.max_len)]
            for key in self.experience_columns
        }
        self.experience_data_status = {
            key: torch.zeros(self.max_len, dtype=torch.int32)
            for key in self.experience_columns
        }

        self.timeout = timeout if timeout is not None else 300
        self.timeout_interval = timeout_interval if timeout_interval is not None else 5

    def _put(
        self,
        experience_columns: List[str],
        experience: List[List[List[torch.Tensor]]],
        indexes: List[int] = None,
    ):
        """Put data into specified columns and rows.

        Args:
            experience_columns: Columns to put data in.
                ['prompts', 'attention_mask']
            experience: Data for the corresponding columns.
                [
                    [
                        tensor([1, 1, 1, 1]),
                        tensor([2, 2, 2, 2]),
                        tensor([3, 3, 3, 3]),
                        tensor([4, 4, 4, 4])
                    ],
                    [
                        tensor([1]),
                        tensor([2, 2]),
                        tensor([3, 3, 3]),
                        tensor([4, 4, 4, 4])
                    ]
                ]
            indexes: Rows to put data in.
                [0, 1, 2, 4]

        Returns: None

        """
        for experience_column in experience_columns:
            if experience_column not in self.experience_columns:
                raise ValueError(
                    f"put experience ERROR: {experience_column} not in TD experience_column {self.experience_columns}"
                )

        if not indexes:
            raise ValueError(
                "put experience into TD without indexes, indexes must be provided"
            )

        if max(indexes) >= self.max_len:
            raise ValueError(
                f"Put experience index {max(indexes)} exceeds the Transfer Dock range {self.max_len}."
            )

        for column_idx, single_column in enumerate(experience_columns):
            for i, index in enumerate(indexes):
                self.experience_data[single_column][index] = experience[column_idx][i]
                self.experience_data_status[single_column][index] = 1

    def _get(self, experience_columns: List[str], indexes: List[int]):
        """Get data based on row and column numbers.

        Args:
            experience_columns: Columns from which to get data.
                ['prompts', 'attention_mask']
            indexes: Rows to get data from.
                [0, 2]

        Returns: Data list.
            [
                [
                    tensor([1, 1, 1, 1]),
                    tensor([2, 2, 2, 2]),
                    tensor([3, 3, 3, 3]),
                    tensor([4, 4, 4, 4])
                ],
                [
                    tensor([1]),
                    tensor([2, 2]),
                    tensor([3, 3, 3]),
                    tensor([4, 4, 4, 4])
                ]
            ]

        """
        if len(indexes) == 0:
            return [[] for _ in range(len(experience_columns))]

        if max(indexes) >= self.max_len:
            raise ValueError(
                f"Get experience index {max(indexes)} exceeds the Transfer Dock range {self.max_len}."
            )

        experience = []
        for single_column in experience_columns:
            self._wait_for_data(single_column, indexes)
            if len(indexes) == 1:
                experience.append([self.experience_data[single_column][indexes[0]]])
            else:
                experience.append(list(itemgetter(*indexes)(self.experience_data[single_column])))

        return experience

    def _wait_for_data(self, single_column: str, indexes: List[int]):
        """Wait for data in which column and row to be ready.

        Args:
            single_column: Column that need to wait for data to be ready.
            indexes: Rows that need to wait for data to be ready.

        Returns: None

        """
        if len(indexes) == 1:
            data_ready = self.experience_data_status[single_column][indexes] == 1
        else:
            data_ready = sum(itemgetter(*indexes)(self.experience_data_status[single_column])) == len(indexes)

        start_time = time.time()
        while not data_ready:
            elapsed_time = time.time() - start_time
            if (
                elapsed_time > self.timeout
                and elapsed_time % self.timeout_interval < 0.1
            ):
                logger.warning(f"TIMEOUT: data_ready has slept {elapsed_time} second")
            time.sleep(0.1)
            if len(indexes) == 1:
                data_ready = self.experience_data_status[single_column][indexes] == 1
            else:
                data_ready = sum(
                    itemgetter(*indexes)(self.experience_data_status[single_column])
                ) == len(indexes)

    def _clear_experience_data_and_status(self, indexes=None):
        """Clear data and data status in TransferDock.

        Returns: None

        """
        if indexes is None:
            self.experience_data = {
                key: [None for _ in range(self.max_len)]
                for key in self.experience_columns
            }
            self.experience_data_status = {
                key: torch.zeros(self.max_len, dtype=torch.int32)
                for key in self.experience_columns
            }
        else:
            for key in self.experience_columns:
                self.experience_data_status[key][indexes] = 0
            for key in self.experience_columns:
                for idx in indexes:
                    self.experience_data[key][idx] = None

    def get_experience_data(self):
        """Get all data in TransferDock.

        Returns: Data dict.

        """
        return self.experience_data

    def get_experience_status(self):
        """Get all data status in TransferDock.

        Returns: Data status dict.

        """
        return self.experience_data_status

    def get_experience_len(self):
        """Get the maximum length of data in TransferDock.

        Returns: The maximum length of data.

        """
        return self.max_len


@ray.remote(max_concurrency=100, num_cpus=10)
class GRPOTransferDock(TransferDock):
    """
    GRPOTransferDock is based on TransferDock and supports managing data transfer between
    GRPO asynchronous tasks in the Ray cluster.
    """

    def __init__(
        self,
        prompts_num: int,
        n_samples_per_prompt: int,
        metrics=None,
        addition_columns: Union[List[str], None] = None,
        addition_consumers: Union[List[str], None] = None,
        timeout: Union[int, None] = None,
        timeout_interval: Union[int, None] = None,
    ) -> None:
        """GRPOTransferDock initialize.

        Args:
            prompts_num: The number of prompts loaded from the dataset.
            n_samples_per_prompt: The number of responses for a prompt.
            metrics: The metrics stored in TransferDock.
            addition_columns: Additional experience columns in TransferDock.
            addition_consumers: Additional consumers in TransferDock.
            timeout: The waiting time for over time printing.
            timeout_interval: Time interval for timeout printing.
        """
        self.experience_columns = [
            "prompts",
            "prompt_length",
            "responses",
            "response_length",
            "attention_mask",
            "labels",
            "input_ids",
            "input_ids_length",
            "actor_rollout",
            "rm_scores",
            "token_level_rewards",
            "old_log_prob",
            "ref_log_prob",
            "advantages",
            "returns",
        ]
        self.experience_consumers = [
            "trainer",
            "actor_rollout",
            "actor_log_prob",
            "ref_log_prob",
            "actor_train",
            "compute_advantage",
            "rule_reward",
            "reward_scores",
            "grpo_metrics",
            "actor_image_embeds",
        ]
        if addition_columns:
            for column in addition_columns:
                if column not in self.experience_columns:
                    self.experience_columns.append(column)

        if addition_consumers:
            for consumer in addition_consumers:
                if consumer not in self.experience_consumers:
                    self.experience_consumers.append(consumer)

        super().__init__(
            prompts_num,
            n_samples_per_prompt,
            self.experience_columns,
            timeout,
            timeout_interval,
        )
        self.experience_consumer_status = {
            key: torch.zeros(self.max_len, dtype=torch.int32)
            for key in self.experience_consumers
        }
        self.consumer_sampling_lock = {
            key: threading.Lock()
            for key in self.experience_consumers
        }
        self.metrics = metrics

    def get_metrics(self):
        return self.metrics

    def update_metrics(self, key="", value=None, cumulate=False):
        self.metrics.update(key, value, cumulate=cumulate)

    def get_experience(
        self,
        consumer: str,
        experience_columns: List[str],
        experience_count: int = None,
        indexes: List[int] = None,
        get_n_samples: bool = True,
    ):
        """Get padded experience data from GRPOTransferDock.

        Args:
            consumer: GRPO task stage to get in.
            experience_columns: Columns from which to get data.
            experience_count: Number of data to get.
            indexes: Rows from which to get data.
            pad_id: Pad token.
            multiple: The multiple of TP to pad.
            get_n_samples: Whether to get n samples at the same time.
            target_seq_len: Target sequence length.

        Returns: Data dict and row numbers.

        """
        if consumer not in self.experience_consumers:
            raise ValueError(
                f"get experience ERROR: {consumer} not in TD experience_consumers {self.experience_consumers}"
            )

        for experience_column in experience_columns:
            if experience_column not in self.experience_columns:
                raise ValueError(
                    f"get experience ERROR: {experience_column} not in TD experience_column {self.experience_columns}"
                )

        if indexes is None:
            if experience_count > self.max_len:
                raise ValueError(
                    f"TD max_len: {self.max_len} need >= experience_count: {experience_count}"
                )

            if self.max_len % experience_count != 0:
                raise ValueError(
                    f"TD max_len:{self.max_len} need be divisible by experience_count: {experience_count}"
                )

            if get_n_samples:
                if experience_count % self.n_samples_per_prompt != 0:
                    raise ValueError(
                        f"get_n_samples need experience_count:{experience_count} must be divisible by "
                        f"n_samples_per_prompt: {self.n_samples_per_prompt}"
                    )
                indexes = self._sample_ready_index_n_samples(
                    consumer, experience_count, experience_columns
                )
            else:
                indexes = self._sample_ready_index(
                    consumer, experience_count, experience_columns
                )

            if not indexes:
                return None, None
            experience = self._get(experience_columns, indexes)
        else:
            self.experience_consumer_status[consumer][indexes] = 1
            experience = self._get(experience_columns, indexes)

        experience_batch = {}
        for i, experience_column in enumerate(experience_columns):
            experience_batch[experience_column] = experience[i]
        return experience_batch, indexes

    def put_experience(
        self,
        data_dict: Dict[str, Union[Tensor, List[Tensor]]],
        indexes: List[int] = None,
    ):
        """Put data into specified columns and rows.

        Args:
            data_dict: Data dict to put in GRPOTransferDock.
            indexes: Rows to put data in.

        Returns: None

        """

        if not indexes:
            raise ValueError(
                "put experience into TD without indexes, indexes must be provided"
            )
        experience_columns, experience = trans_input_to_experience(data_dict)
        self._put(experience_columns, experience, indexes)

    def put_prompts_experience(
        self, batch: Dict[str, Tensor], dataset_additional_keys: List[str] = None
    ):
        """Put data into specified columns and rows.

        Args:
            batch: Batch datas from original dataloader.
            dataset_additional_keys: The additional experience types from the dataset.

        Returns: None

        """

        prompts = batch["prompts"]
        prompt_length = []
        for prompt in prompts:
            for _ in range(self.n_samples_per_prompt):
                prompt_length.append(torch.tensor([len(prompt)]))

        prompts_data = prompts
        prompts = []
        for prompt in prompts_data:
            for _ in range(self.n_samples_per_prompt):
                prompts.append(copy.deepcopy(prompt))

        add_vals = {}
        for add_keys in dataset_additional_keys:
            if add_keys in batch.keys():
                values = []
                for value in batch[add_keys]:
                    for _ in range(self.n_samples_per_prompt):
                        values.append(value)
                add_vals[add_keys] = values

        indexes = [i for i in range(len(prompt_length))]
        data_dict = dict(
            {"prompt_length": prompt_length, "prompts": prompts}, **add_vals
        )
        experience_columns, experience = trans_input_to_experience(data_dict)

        self._put(experience_columns, experience, indexes)

    def _sample_ready_index(
        self,
        consumer: str,
        experience_count: int,
        experience_columns: List[str],
        target_seq_len: int = None,
    ) -> Optional[List[int]]:
        """Randomly select a specified number of prepared experiences from TransferDock.

        Args:
            consumer: GRPO task stage to sample in.
            experience_count: Number for rows to sample.
            experience_columns: Columns from which to sample.

        Returns: Sampled row numbers.

        """

        with self.consumer_sampling_lock[consumer]:
            not_consumed_indexes = self.experience_consumer_status[consumer] == 0
            data_ready_indexes = torch.all(
                torch.stack(
                    [self.experience_data_status[single_column] == 1 for single_column in experience_columns]
                ), dim=0,
            )
            usable_indexes = (not_consumed_indexes & data_ready_indexes).nonzero(as_tuple=True)[0]

            if len(usable_indexes) < experience_count:
                return None

            if experience_count > 0:
                sampled_indexes = self.batch_balencing_sampler(
                    experience_columns, usable_indexes, experience_count, target_seq_len
                )
                self.experience_consumer_status[consumer][sampled_indexes] = 1
            else:
                sampled_indexes = None

        return sampled_indexes

    def _sample_ready_index_n_samples(
        self,
        consumer: str,
        experience_count: int,
        experience_columns: List[str],
        target_seq_len: int = None,
    ) -> Optional[List[int]]:
        """Randomly select a specified number of prepared experiences from TransferDock at multiples of n_sample.

        Args:
            consumer: GRPO task stage to sample in.
            experience_count: Number for rows to sample.
            experience_columns: Columns from which to sample.
            target_seq_len: Sample according with seq_len and target_seq_len.

        Returns: Sampled row numbers.

        """
        experience_count_n_samples = experience_count // self.n_samples_per_prompt
        with self.consumer_sampling_lock[consumer]:
            experience_consumer_status_n_samples = (
                1 - torch.all(
                    torch.tensor(
                        torch.reshape(
                            self.experience_consumer_status[consumer],
                            (self.prompts_num, self.n_samples_per_prompt),
                        ) == 0
                    ), dim=1,
                ).int()
            )
            not_consumed_indexes = experience_consumer_status_n_samples == 0

            experience_data_status_n_samples = {}
            for key, value in self.experience_data_status.items():
                experience_data_status_n_samples[key] = torch.all(
                    torch.tensor(
                        torch.reshape(value, (self.prompts_num, self.n_samples_per_prompt)) == 1
                    ), dim=1,
                ).int()

            data_ready_indexes = torch.all(
                torch.stack(
                    [experience_data_status_n_samples.get(single_column) == 1 for single_column in experience_columns]),
                dim=0,
            )

            usable_indexes = (not_consumed_indexes & data_ready_indexes).nonzero(as_tuple=True)[0]

            if len(usable_indexes) < experience_count_n_samples:
                return None

            sampled_indexes_n_sample = self.batch_balencing_sampler(
                experience_columns,
                usable_indexes,
                experience_count_n_samples,
                target_seq_len,
            )

            sampled_indexes = []
            for n_sample_index in sampled_indexes_n_sample:
                index_list = []
                for index in range(
                        n_sample_index * self.n_samples_per_prompt,
                        (n_sample_index + 1) * self.n_samples_per_prompt
                ):
                    index_list.append(index)

                sampled_indexes += index_list

                self.experience_consumer_status[consumer][sampled_indexes] = 1

        return sampled_indexes

    def all_consumed(self, consumer: str):
        """If consumer has consumed all data in GRPOTransferDock.

        Args:
            consumer: GRPO task stage to consume in.

        Returns: True or False.

        """
        return self.experience_consumer_status[consumer].sum() == self.max_len

    def clear(self):
        """Reset consumer status.Clear data and data status in GRPOTransferDock.

        Returns: None

        """
        self.experience_consumer_status = {
            key: torch.zeros(self.max_len, dtype=torch.int32)
            for key in self.experience_consumers
        }
        self.metrics.reset()
        self._clear_experience_data_and_status()

    def get_consumer_status(self):
        """Get consumer status.

        Returns: Consumer status dict.

        """
        return self.experience_consumer_status

    def batch_balencing_sampler(
        self, experience_columns, usable_indexes, experience_count, target_seq_len=None
    ):
        if target_seq_len is None:
            weights = torch.ones(len(usable_indexes))
        else:
            seq_len = torch.tensor(
                [
                    sum([self.experience_data[key][idx].numel() for key in experience_columns])
                    for idx in usable_indexes
                ]
            )
            weights = torch.sigmoid(1 / (torch.abs(seq_len - target_seq_len) + 0.001), dim=0)

        sampled_indexes_idx = torch.multinomial(weights, experience_count, replacement=False).tolist()
        sampled_indexes = [int(usable_indexes[i]) for i in sampled_indexes_idx]

        return sampled_indexes


@ray.remote(max_concurrency=100, num_cpus=10)
class MMGRPOTransferDock(TransferDock):
    def __init__(self, prompts_num: int,
                 n_samples_per_prompt: int,
                 reuse_image_embeds: bool = False,
                 timeout: Union[int, None] = None,
                 timeout_interval: Union[int, None] = None) -> None:
        self.experience_columns = [
            'image',
            'pixel_values',  # 'pixel_values_videos'
            'image_grid_thw',  # 'video_grid_thw
            'image_shape',
            'labels',
            'vit_embeds',
            'position_ids',
            'image_num',
            'video',
            'video_shape',
            'video_fps',
            'video_num'
        ]
        super().__init__(
            prompts_num,
            1,
            self.experience_columns, 
            timeout, 
            timeout_interval
            )
        
        self.n_samples_per_prompt = n_samples_per_prompt
        self.consumer_columns = {
            "actor_rollout": ["image", "image_shape", "image_num", "video", "video_shape", "video_fps", "video_num"],
            "actor_log_prob": ["pixel_values", "image_grid_thw", "image_num", "video_num"],
            "ref_log_prob": ["pixel_values", "image_grid_thw", "image_num", "video_num"],
            "actor_train": ["pixel_values", "image_grid_thw", "image_num", "video_num"],
            "rule_reward": ["labels"],
        }

        if reuse_image_embeds:
            self.consumer_columns["actor_image_embeds"] = ["pixel_values", "image_grid_thw", "image_num", "video_num"]
            self.consumer_columns["actor_rollout"] = ["vit_embeds", "image_grid_thw", "image_num", "video_num"]
            self.consumer_columns["actor_log_prob"] = ["vit_embeds", "image_grid_thw", "image_num", "video_num"]
            self.consumer_columns["ref_log_prob"] = ["vit_embeds", "image_grid_thw", "image_num", "video_num"]

    def get_columns(self, consumer: str):
        if consumer not in self.consumer_columns:
            return []

        for column in self.consumer_columns[consumer]:
            if all(data is None for data in self.experience_data[column]):
                return []
            
        return self.consumer_columns[consumer]

    def get_experience(
            self,
            experience_columns: List[str],
            indexes: List[int] = None,
            get_n_samples: bool = True,
    ):
        """Get multimodal experience data from GRPOTransferDock.

        Args:
            experience_columns: Columns from which to get data.
            indexes: Rows from which to get data.

        Returns: Data dict.

        """
        if indexes is None:
            return {}
        
        if get_n_samples:
            indexes = indexes[::self.n_samples_per_prompt]
        
        indexes = [i // self.n_samples_per_prompt for i in indexes]
        experience = []
        for single_column in experience_columns:
            if len(indexes) == 1:
                experience.append([self.experience_data[single_column][indexes[0]]])
            else:
                experience.append(list(itemgetter(*indexes)(self.experience_data[single_column])))

        return trans_mm_experience_to_output(experience, experience_columns)

    def put_experience(
            self,
            batch: Dict[str, Tensor],
            indexes: List[int] = None,
    ):
        """Put data into specified columns and rows.

        Args:
            data_dict: Data dict to put in GRPOTransferDock.
            indexes: Rows to put data in.

        Returns: None

        """
        for experience_column, experience_list in batch.items():
            if experience_column in self.experience_columns:
                for i in range(len(experience_list)):
                    self.experience_data[experience_column][indexes[i]] = experience_list[i]
    
    def clear(self):
        """Reset consumer status.Clear data and data status in GRPOTransferDock.

        Returns: None

        """
        self._clear_experience_data_and_status()


def pad_experience(
        experience_batch: Dict[str, List[Tensor]],
        pad_id: int,
        multiple: int = 1,
):
    """ Pad dict data.

    Args:
        experience_batch: Dict
            {
                'prompts': [ tensor([1, 1, 1, 1]), 
                             tensor([2, 2, 2, 2]),
                             tensor([3, 3, 3, 3]), 
                             tensor([4, 4, 4, 4])], 
                'attention_mask': [ tensor([1]), 
                                    tensor([2, 2]), 
                                    tensor([3, 3, 3]), 
                                    tensor([4, 4, 4, 4])], 
            }
        pad_id: Pad token.
            0.0
        multiple: The multiple of TP to pad.
            1

    Returns: Merged and padded data dict.
        {
            "prompts": tensor(
                [[1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3],
                [4, 4, 4, 4]]),
            "attention_mask": tensor(
                [[1, 0, 0, 0],
                [2, 2, 0, 0],
                [3, 3, 3, 0],
                [4, 4, 4, 4]]),
        }

    """
    def pad_multiples(data_list: List[Tensor], pad_id: Union[float, int], multiple: int = 1) -> Tensor:
        """Pad method for data list.

        Args:
            data_list: Data list.
            pad_id: Pad token.
            multiple: The multiple of TP to pad.

        Returns: Padded tensor.

        """
        padded = pad_sequence(data_list, batch_first=True, padding_value=pad_id)
        max_len = padded.size(1)
        target_len = ((max_len + multiple - 1) // multiple) * multiple
        padded = F.pad(padded, (0, target_len - max_len), value=pad_id)
        return padded

    batch = {}
    if not experience_batch:
        raise ValueError("ERROR: when pad, get an empty experience_batch")
    else:
        for experience_column, experience in experience_batch.items():
            if experience_column in ["prompt_length", "response_length"]:
                padded = torch.cat(experience).reshape(-1, 1)
            elif experience_column in ["position_ids"]:
                padded = pad_sequence(experience, batch_first=True, padding_value=pad_id)
            elif experience[0].is_floating_point():
                padded = pad_multiples(experience, pad_id=0.0, multiple=multiple)
            else:
                padded = pad_multiples(experience, pad_id=pad_id, multiple=multiple)

            batch[experience_column] = padded

    return batch


def trans_mm_experience_to_output(
        experience: List[Union[Tensor, str]],
        experience_columns: List[str],
):
    """Merge and pad multimodal data into dict.

    Args:
        experience: Data list.
        experience_columns: Columns for the corresponding data.

    Returns: Merged data dict.

    """
    batch = {}
    for i, experience_column in enumerate(experience_columns):
        if experience_column == 'labels':
            data = experience[i]
        elif experience_column in ['image_num', 'video_num', 'video_fps']:
            data = torch.tensor(experience[i]).reshape(-1, 1)
        elif experience_column == 'image':
            data = torch.concat(experience[i], dim=1)
        else:
            data = torch.concat(experience[i])

        batch[experience_column] = data

    return batch


def trans_input_to_experience(experience_dict: Dict[str, Union[Tensor, List[Tensor]]]):
    """Split data dict into columns and data list.

    Args:
        experience_dict: Data dict.
            {
                "prompts": tensor(
                    [[1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [3, 3, 3, 3],
                    [4, 4, 4, 4]]),
                "attention_mask": [
                    tensor([1]),
                    tensor([2, 2]),
                    tensor([3, 3, 3]),
                    tensor([4, 4, 4, 4])]
            }
        num_responses: The number of data to put in each row.
            2

    Returns: Columns and data list.
        ['prompts', 'attention_mask']
        [
            [
                tensor([1, 1, 1, 1]),
                tensor([2, 2, 2, 2]),
                tensor([3, 3, 3, 3]),
                tensor([4, 4, 4, 4])
            ],
            [
                tensor([1)],
                tensor([2, 2]),
                tensor([3, 3, 3]),
                tensor([4, 4, 4, 4])
            ]
        ]

    """
    experience_columns = []
    experience_list = []
    for key, value in experience_dict.items():
        if value is not None:
            experience_columns.append(key)
            if isinstance(value, Tensor):
                if value.dtype == torch.int64:
                    value = value.to(torch.int32)
                value = list(torch.unbind(value, dim=0))
            elif isinstance(value, List):
                value = [val.to(torch.int32) if val.dtype == torch.int64 else val for val in value]
            else:
                raise ValueError(f"value type {type(value)} not supported")
            experience_list.append(value)

    return experience_columns, experience_list


def pack_experience_columns(experience_dict, experience_count):
    """ 
    Compress experiences by packing tensors into ONE.
    from experience_dict
        {
            'prompts': [ tensor([1, 1, 1]), 
                            tensor([2, 2, 2, 2]),
                            tensor([3, 3, 3]), 
                            tensor([4, 4, 4, 4])], 
            'attention_mask': [ tensor([1]), 
                                tensor([2, 2]), 
                                tensor([3, 3, 3]), 
                                tensor([4, 4, 4, 4])], 
            'position_ids': [ tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]]), 
                             tensor([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])],
        }
    To batch_data
        {
            'prompts': tensor([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4]),
            'attention_mask': tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4]),
            'position_ids': tensor([[0, 0, 0, 0, 0, 0, 0], 
                                    [1, 1, 1, 1, 1, 1, 1], 
                                    [2, 2, 2, 2, 2, 2, 2]])
        }
        batch_data_length
        {
            'prompts': tensor([3, 4, 3, 4]), 
            'attention_mask': tensor([1, 2, 3, 4]),
            'position_ids': tensor([3, 4])
        }
    """

    if not experience_dict:
        raise ValueError(f"ERROR: when pack, get an empty experience_dict")

    batch_data = {}
    batch_data_length = {}

    for key, value in experience_dict.items():
        if len(value) != experience_count:
            raise ValueError(f"ERROR: when pack, experience '{key}' number does not match experience_count")
        
        # 判断是一维张量还是二维张量
        is_2d = len(value[0].shape) > 1
        
        if is_2d:
            # 处理二维张量，如position_ids
            first_dim = value[0].shape[0]
            # 确保所有张量的第一维相同
            for i in range(experience_count):
                if value[i].shape[0] != first_dim:
                    raise ValueError(f"ERROR: when pack 2D tensor, first dimension must be the same for all experiences")
            
            # 准备存储连接后的二维张量
            packed_data = []
            for dim_idx in range(first_dim):
                dim_data = []
                for i in range(experience_count):
                    dim_data.extend(value[i][dim_idx].tolist())
                packed_data.append(dim_data)
            
            batch_data[key] = torch.tensor(packed_data, dtype=value[0].dtype)
            
            # 仅记录第二维的长度
            data_length = [value[i].shape[1] for i in range(experience_count)]
            batch_data_length[key] = torch.tensor(data_length, dtype=torch.int32)
        else:
            # 原有的一维张量处理逻辑
            packed_experience = []
            data_length = []
            for i in range(experience_count):
                packed_experience.extend(value[i].tolist())
                data_length.append(len(value[i]))

            batch_data[key] = torch.tensor(packed_experience, dtype=value[0].dtype)
            batch_data_length[key] = torch.tensor(data_length, dtype=torch.int32)

    return batch_data, batch_data_length


def unpack_pad_experience(batch_data, batch_data_length, pad_id, multiple):
    """
    1. restore the received experience dict
    2. pad the tensor (consider the requirement of multiple)
    from batch_data
        {
            'prompts': tensor([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4]),
            'attention_mask': tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4]),
            'position_ids': tensor([[0, 0, 0, 0, 0, 0, 0], 
                                    [1, 1, 1, 1, 1, 1, 1], 
                                    [2, 2, 2, 2, 2, 2, 2]])
        }
        batch_data_length
        {
            'prompts': tensor([3, 4, 3, 4]),
            'attention_mask': tensor([1, 2, 3, 4]),
            'position_ids': tensor([3, 4])
        }
    To padded_batch_data (multiple=2)
        {
            "prompts": tensor(
                [[1, 1, 1, -1, -1, -1, -1, -1],
                [2, 2, 2, 2, -1, -1, -1, -1],
                [3, 3, 3, -1, -1, -1, -1, -1],
                [4, 4, 4, 4, -1, -1, -1, -1]]),
            "attention_mask": tensor(
                [[1, -1, -1, -1, -1, -1, -1, -1],
                [2, 2, -1, -1, -1, -1, -1, -1],
                [3, 3, 3, -1, -1, -1, -1, -1],
                [4, 4, 4, 4, -1, -1, -1, -1]]),
            "position_ids": tensor(
                [[[0, 0, 0, -1, -1, -1, -1, -1],
                  [1, 1, 1, -1, -1, -1, -1, -1],
                  [2, 2, 2, -1, -1, -1, -1, -1]],
                 [[0, 0, 0, 0, -1, -1, -1, -1],
                  [1, 1, 1, 1, -1, -1, -1, -1],
                  [2, 2, 2, 2, -1, -1, -1, -1]]]),
        }
    """
    if not batch_data:
        raise ValueError(f"ERROR: empty batch_data")

    if set(batch_data.keys()) != set(batch_data_length.keys()):
        raise ValueError(f"ERROR: when unpack, keys from batch_data and batch_data_length dictionaries do not match")

    data_device = batch_data[list(batch_data.keys())[0]].device

    padded_batch_data = {}
    for key, length_list in batch_data_length.items():
        if key in ['prompt_length', 'response_length']:
            padded_batch_data[key] = batch_data[key].view(-1, 1)
            continue
        
        data = batch_data[key]
        data_dtype = batch_data[key].dtype
        
        # 判断是一维还是二维张量
        is_2d = len(data.shape) > 1
        
        lengths = length_list.to(data_device)
        
        if is_2d:
            # 处理二维张量，如position_ids
            first_dim = data.shape[0]
            
            # 计算最大长度
            max_row_len = torch.max(lengths).item()
            if multiple > 1:
                max_row_len = ((max_row_len + multiple - 1) // multiple) * multiple
                
            # 创建结果张量，每个样本是一个单独的2D张量
            sample_count = len(lengths)
            result = []
            
            # 预分配张量
            if data[0].is_floating_point():
                padded_tensor = torch.full((sample_count, first_dim, max_row_len), 0.0,
                                          dtype=data_dtype, device=data_device)
            else:
                padded_tensor = torch.full((sample_count, first_dim, max_row_len), pad_id,
                                          dtype=data_dtype, device=data_device)
            
            # 计算累积长度
            cum_length = torch.cat([torch.tensor([0], device=data_device), 
                                   torch.cumsum(lengths, 0)])
            
            # 填充每个样本
            for i in range(sample_count):
                seq_len = lengths[i]
                for dim_idx in range(first_dim):
                    start_idx = cum_length[i]
                    end_idx = cum_length[i] + seq_len
                    padded_tensor[i, dim_idx, :seq_len] = data[dim_idx, start_idx:end_idx]
            
            padded_batch_data[key] = padded_tensor
        else:
            # 原有的一维张量处理逻辑
            # 计算最大长度
            max_row_len = torch.max(lengths).item()
            if multiple > 1:
                max_row_len = ((max_row_len + multiple - 1) // multiple) * multiple

            # 预分配张量
            if data.is_floating_point():
                padded_tensor = torch.full((len(lengths), max_row_len), 0.0,
                                       dtype=data_dtype, device=data_device)
            else:
                padded_tensor = torch.full((len(lengths), max_row_len), pad_id,
                                       dtype=data_dtype, device=data_device)

            # 向量化填充
            cum_length = torch.cat([torch.tensor([0], device=data_device
                                             ), torch.cumsum(lengths, 0)])

            for i, _ in enumerate(lengths):
                seq_len = lengths[i]
                padded_tensor[i, :seq_len] = data[cum_length[i]:cum_length[i + 1]]
            padded_batch_data[key] = padded_tensor

    return padded_batch_data


def calculate_split_indices(tensor_shapes: Tensor, merge_shape: bool=False) -> Tuple[List[int], List[int]]:
    """
    Calculate tensor sizes and split indices based on tensor shapes.
    
    Args:
        tensor_shapes: A tensor shape
    
    Returns:
        A tuple containing:
            - tensor_sizes: A list of total elements in each tensor
            - split_indices: A list of indices to split the flattened tensor
    """
    if merge_shape:
        from megatron.training import get_args
        merge_size = get_args().mm.model.image_encoder.vision_encoder.spatial_merge_size

    if isinstance(tensor_shapes, List):
        tensor_shapes = torch.cat(tensor_shapes)

    tensor_sizes = []
    for shape in tensor_shapes:
        size = shape.prod()
        if merge_shape:
            size //= (merge_size * merge_size)
        tensor_sizes.append(size.item())
    
    split_indices = [0]
    for size in tensor_sizes:
        split_indices.append(split_indices[-1] + size)
    
    return tensor_sizes, split_indices


def restore_images_from_tensors(flattened_tensors: Tensor, tensor_shapes: Tensor, image_num: Tensor) -> List:
    """
    Restore PIL images from tensor shapes and flattened tensors.
    
    Args:
        flattened_tensors: A list of flattened tensors, each representing a flattened image.
        tensor_shapes: A tensor of shape [num_images, 3] where each row contains [channels, height, width]
                      for each image.
        image_num: image nums in prompt
       
    Returns:
        A list of PIL Image objects reconstructed from the flattened_tensors.
    """
    tensor_sizes, split_indices = calculate_split_indices(tensor_shapes)
    
    reconstructed_images = []
    to_pil = transforms.ToPILImage()
    
    flattened_tensors = flattened_tensors.squeeze(0)
    for i in range(len(tensor_sizes)):
        start_idx = split_indices[i]
        end_idx = split_indices[i+1]
        flat_tensor = flattened_tensors[start_idx:end_idx]
        
        reconstructed_tensor = flat_tensor.reshape(tensor_shapes[i].tolist())
        reconstructed_image = to_pil(reconstructed_tensor)
        reconstructed_images.append(reconstructed_image)

    res_images = []
    start_idx = 0
    image_num = image_num.squeeze(0)
    for i in image_num:
        res_images.append(reconstructed_images[start_idx: start_idx + i.item()])
        start_idx += i.item()
    return res_images


def restore_videos_from_tensors(flattened_tensors: Tensor, tensor_shapes: Tensor, video_num: Tensor) -> List:
    
    print(f"restore_videos_from_tensors flattened_tensors:{flattened_tensors.size()}")
    print(f"restore_videos_from_tensors tensor_shapes:{tensor_shapes.size()}")
    print(f"restore_videos_from_tensors video_num:{video_num.size()}")
    
    tensor_sizes, split_indices = calculate_split_indices(tensor_shapes)

    reconstructed_videos = []
    flattened_tensors = flattened_tensors.squeeze(0)
    for i in range(len(tensor_sizes)):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]
        flat_tensor = flattened_tensors[start_idx:end_idx]

        reconstructed_videos.append(flat_tensor.reshape(tensor_shapes[i].tolist()))

    res_video = []
    start_idx = 0
    video_num = video_num.squeeze(0)
    for i in video_num:
        res_video.append(reconstructed_videos[start_idx: start_idx + i.item()])
        start_idx += i.item()
    return res_video


def restore_pixel_valaues_from_flattend(flattened_tensors: Tensor, tensor_shapes: Tensor, image_num: Tensor=None, merge_shape: bool=False) -> List[Tensor]:
    """
    Restore Ppixel_valaues from tensor shapes and flattened tensors.
    
    Args:
        flattened_tensors: A list of flattened tensors, each representing a flattened pixel_values.
        tensor_shapes: A tensor_shapes of original pixel_values
       
    Returns:
        reconstructed_pixel_values: A list of pixel_values reconstructed from the flattened_tensors.
        reconstructed_tensor_shapes: tensor_shapes repeat n at dim 0
    """
    tensor_sizes, split_indices = calculate_split_indices(tensor_shapes, merge_shape=merge_shape)
    
    reconstructed_pixel_values = []
    for i in range(len(tensor_sizes)):        
        start_idx = split_indices[i]
        end_idx = split_indices[i+1]

        reconstructed_tensor = flattened_tensors[start_idx:end_idx, :]
        reconstructed_pixel_values.append(reconstructed_tensor)

    if image_num is None:
        return reconstructed_pixel_values

    # multi image pixel value process
    res_pixel_values = []
    start_idx = 0
    image_num = image_num.squeeze(0)
    for i in image_num:
        res_pixel_values.append(torch.cat(reconstructed_pixel_values[start_idx: start_idx + i.item()]))
        start_idx += i.item()

    return res_pixel_values


def restore_split_data(data_tensor: Tensor, split_num: Tensor):
    """
    reconstruct data like image_grid_thw, video_fps:
        [[1,30,40],[1,20,20]]    data1
        [[1,30,40],[1,20,20]]    data2
    will concat like [[1,30,40],[1,20,20],[1,30,40],[1,20,20]] -> [[[1,30,40],[1,20,20]], [[1,30,40],[1,20,20]] ]
    this func used to reconstruct by split_num recorded in data
    """
    res = []
    start_idx = 0
    split_num = split_num.squeeze(0)
    for i in split_num:
        res.append(data_tensor[start_idx: start_idx + i.item()])
        start_idx += i.item()
    return res


def restore_image_grid_thw(image_grid_thw_tensor: Tensor, image_num: Tensor):
    res_image_grid_thw = []
    start_idx = 0
    image_num = image_num.squeeze(0)
    for i in image_num:
        res_image_grid_thw.append(image_grid_thw_tensor[start_idx: start_idx + i.item()])
        start_idx += i.item()
    return res_image_grid_thw


def unpack_mm_experience(batch_data: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Handles multimodal data by restoring images and pixel values from flattened tensors.

    Args:
        batch_data (Dict[str, Tensor]): A dictionary containing the batch data.
        n_samples_per_prompt (int): The number of samples per prompt.

    Returns:
        Dict[str, Tensor]: The processed batch data with restored images and pixel values."
    """
    image_keys = {"image", "image_shape", "video", "video_shape"}
    pixel_values_keys = {"pixel_values", "image_grid_thw"}
    vit_embeds_keys = {"vit_embeds", "image_grid_thw"}
    
    # breakpoint()

    if image_keys.issubset(batch_data.keys()):
        # not support hybrid image&video dataset
        if torch.sum(batch_data["image_num"]).item() > 0:
            batch_data["image"] = restore_images_from_tensors(batch_data["image"], batch_data["image_shape"], batch_data["image_num"])
        else:
            batch_data["video"] = restore_videos_from_tensors(batch_data["video"], batch_data["video_shape"], batch_data["video_num"])
            batch_data["video_fps"] = restore_split_data(batch_data["video_fps"], batch_data["video_num"])

    if pixel_values_keys.issubset(batch_data.keys()):
        mm_data_num = batch_data["image_num"] if torch.sum(batch_data["image_num"]).item() else batch_data["video_num"]
        batch_data["pixel_values"] = restore_pixel_valaues_from_flattend(batch_data["pixel_values"],
                                                                         batch_data["image_grid_thw"], mm_data_num)
        batch_data["image_grid_thw"] = restore_split_data(batch_data["image_grid_thw"], mm_data_num)

    if vit_embeds_keys.issubset(batch_data.keys()):
        mm_data_num = batch_data["image_num"] if torch.sum(batch_data["image_num"]).item() else batch_data["video_num"]
        batch_data["vit_embeds"] = restore_pixel_valaues_from_flattend(batch_data["vit_embeds"],
                                                                       batch_data["image_grid_thw"], mm_data_num,
                                                                       merge_shape=True)
        batch_data["image_grid_thw"] = restore_split_data(batch_data["image_grid_thw"], mm_data_num)

    return batch_data
