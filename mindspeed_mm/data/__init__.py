__all__ = [
    "build_mm_dataset", "build_mm_dataloader"
]

import copy

from torch.utils.data import ConcatDataset
from torch.distributed.distributed_c10d import _get_default_group

from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from mindspeed_mm.data.dataloader.dataloader import (
    prepare_base_dataloader,
    prepare_sampler_dataloader,
    prepare_variable_dataloader,
)
from mindspeed_mm.data.datasets.multimodal_dataset import DeepSeekVLDataset, MultiModalChatDataset
from mindspeed_mm.data.datasets.t2i_dataset import T2IDataset
from mindspeed_mm.data.datasets.t2v_dataset import T2VDataset, DynamicVideoTextDataset
from mindspeed_mm.data.datasets.i2v_dataset import I2VDataset
from mindspeed_mm.data.datasets.video_dataset import VideoDataset
from mindspeed_mm.data.datasets.audio_dataset import AudioDataset
from mindspeed_mm.data.datasets.qwen2vl_dataset import get_qwen2vl_dataset
from mindspeed_mm.data.datasets.ae_dataset import TrainVideoDataset
from mindspeed_mm.models.ae.training.global_vars import get_ae_args



def build_mm_dataset(dataset_param):
    """
    Build a multimodal dataset based on different tasks

    Args:
        dataset_param
    Return:
        dataset
    """
    if not isinstance(dataset_param, dict):
        dataset_param = dataset_param.to_dict()
    for check_key in ["dataset_type", "basic_parameters", "preprocess_parameters"]:
        if check_key not in dataset_param:
            raise AssertionError(f"Key parameter missing: {check_key}")
    dataset_type = dataset_param["dataset_type"]
    basic_param = dataset_param["basic_parameters"]
    preprocess_param = dataset_param["preprocess_parameters"]
    if dataset_type == "t2v":
        return T2VDataset(basic_param, preprocess_param, **dataset_param)
    elif dataset_type == "i2v":
        return I2VDataset(basic_param, preprocess_param, **dataset_param)
    elif dataset_type == "t2i":
        return T2IDataset(basic_param, preprocess_param, **dataset_param)
    elif dataset_type == "dt2v":  # 构建动态分辨率数据集
        return DynamicVideoTextDataset(basic_param, preprocess_param, **dataset_param)
    elif dataset_type == "video":
        return VideoDataset(basic_param, preprocess_param, **dataset_param)
    elif dataset_type == "multimodal":
        if not isinstance(basic_param, list):
            basic_param = [basic_param]
        datasets = []
        for single_param in basic_param:
            dataset_param["repeat_time"] = single_param.get("repeat_time", 1)
            dataset_param_copy = copy.deepcopy(dataset_param)
            dataset = MultiModalChatDataset(single_param, preprocess_param, **dataset_param_copy)
            datasets.append(dataset)
        return ConcatDataset(datasets)
    elif dataset_type == "audio":
        return AudioDataset(basic_param, preprocess_param, **dataset_param)
    elif dataset_type == "huggingface":
        return get_qwen2vl_dataset(basic_param, preprocess_param, dataset_param)
    elif dataset_type == "deepseekvl2":
        if not isinstance(basic_param, list):
            basic_param = [basic_param]
        datasets = []
        for single_param in basic_param:
            dataset_param["repeat_time"] = single_param.get("repeat_time", 1)
            dataset_param_copy = copy.deepcopy(dataset_param)
            dataset = DeepSeekVLDataset(single_param, **dataset_param_copy)
            datasets.append(dataset)
        return ConcatDataset(datasets)
    else:
        raise NotImplementedError(dataset_type)


def build_mm_dataloader(dataset, dataloader_param, process_group=None, consumed_samples=0, dataset_param=None):
    """
    Build a multimodal dataloader based on different tasks

    dataloader_type interpretation:
    base: raw dataloader based on torch.utils.data.DataLoader
    sampler: prepare a dataloader for distributed training by building a specific sampler
    variable: used for variable dataset

    Args:
        dataloader_param_dict
    Return:
        dataloader
    """
    if not isinstance(dataloader_param, dict):
        dataloader_param = dataloader_param.to_dict()
    if "dataloader_mode" not in dataloader_param:
        raise AssertionError("Key parameter missing: dataloader_mode")
    dataloader_mode = dataloader_param.pop("dataloader_mode")
    if process_group is None:
        process_group = mpu.get_data_parallel_group()
    args = get_args()
    dataloader_param.update(
        {
            "batch_size": args.micro_batch_size,
            "num_workers": args.num_workers,
            "seed": args.seed,
        }
    )
    print_rank_0(f'[INFO] initialize `batch_size`/`num_workers`/`seed` from argument parser rather than `data.json`')
    if dataloader_mode == "base":
        data_loader = prepare_base_dataloader(dataset, **dataloader_param)
        return data_loader
    elif dataloader_mode == "sampler":
        data_loader = prepare_sampler_dataloader(
            dataset, **dataloader_param, process_group=process_group, consumed_samples=consumed_samples,
            dataset_param=dataset_param
        )
        return data_loader
    elif dataloader_mode == "variable":
        data_loader = prepare_variable_dataloader(
            dataset, **dataloader_param, process_group=process_group, consumed_samples=consumed_samples)
        return data_loader
    else:
        raise NotImplementedError(dataloader_param["dataloader_mode"])


def build_ae_dataset(dataset_param):
    """
    Build an AE dataset based on different tasks

    Args:
        dataset_param
    Return:
        dataset
    """
    if not isinstance(dataset_param, dict):
        dataset_param = dataset_param.to_dict()
    return TrainVideoDataset(**dataset_param)


def build_ae_dataloader(dataset, dataloader_param, process_group=None):
    """
    Build an AE dataloader based on different tasks

    Args:
        dataloader_param_dict
    Return:
        dataloader
    """
    if not isinstance(dataloader_param, dict):
        dataloader_param = dataloader_param.to_dict()
    dataloader_mode = dataloader_param.pop("dataloader_mode")
    process_group = process_group if process_group is not None else _get_default_group()

    if dataloader_mode == "sampler":
        args = get_ae_args()
        batch_size = args.micro_batch_size
        num_workers = args.num_workers
        data_loader = prepare_sampler_dataloader(
            dataset, batch_size=batch_size, num_workers=num_workers, **dataloader_param, process_group=process_group
        )
        return data_loader
    else:
        raise NotImplementedError(dataloader_mode)