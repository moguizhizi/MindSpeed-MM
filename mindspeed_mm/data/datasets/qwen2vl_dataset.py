import os
import warnings
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset
from transformers.training_args import TrainingArguments

from megatron.training import get_args
from mindspeed_mm.data.data_utils.func_utils.convert import (
    DataArguments,
    DatasetAttr,
    load_tokenizer,
    convert_sharegpt,
    preprocess_supervised_dataset,
    preprocess_pairwise_dataset
)
from mindspeed_mm.data.data_utils.func_utils.log import get_logger
from mindspeed_mm.data.data_utils.func_utils.model_args import ProcessorArguments
from mindspeed_mm.data.data_utils.func_utils.template import get_template_and_fix_tokenizer

logger = get_logger(__name__)


class DistributedIterableDataset(IterableDataset):
    def __init__(self, dataset, rank=None):
        args = get_args()
        self.data_parallel_size = args.data_parallel_size
        self.dataset = dataset
        self.rank = torch.distributed.get_rank() if rank is None else rank

    def __iter__(self):
        for idx, item in enumerate(self.dataset):
            if idx % self.data_parallel_size == self.rank % self.data_parallel_size:
                yield item


def get_qwen2vl_dataset(basic_param, preprocess_param, dataset_param):
    if "cutoff_len" in basic_param.keys():
        raise ValueError("`cutoff_len` is deprecated, please use `seq_length` instead.")
    data_args = DataArguments(**basic_param)
    data_args.cutoff_len = get_args().seq_length
    process_args = ProcessorArguments(**preprocess_param)
    dataset_attr = DatasetAttr(**dataset_param["attr"])

    tokenizer_module = load_tokenizer(process_args)
    tokenizer, processor = tokenizer_module['tokenizer'], tokenizer_module['processor']
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template)
    # 确保主进程进行数据处理，其他进程复用缓存避免重复计算，该策略和llamafactory对数据处理策略一致
    with TrainingArguments(output_dir='./').main_process_first(desc="pre-process dataset"):
        # -----------------load dataset from file-------------------------------------------------------------------------
        train_dataset = load_dataset(path="json", data_files=data_args.dataset, split="train", cache_dir=data_args.cache_dir,
                               streaming=data_args.streaming)
        if data_args.max_samples and not data_args.streaming:
            train_dataset = train_dataset.select(range(data_args.max_samples))

        val_dataset = None
        if data_args.val_dataset:
            val_dataset = load_dataset(
                path="json",
                data_files=data_args.val_dataset,
                split="train",
                cache_dir=data_args.cache_dir,
                streaming=data_args.streaming
            )
            if data_args.val_max_samples:
                val_dataset = val_dataset.select(range(data_args.val_max_samples))
            if data_args.val_rate is not None and data_args.val_rate > 0.0:
                warnings.warn("Warning: Both val_dataset and val_rate have been provided. The val_dataset will take priority, and the val_rate will be ignored.", UserWarning)

        local_process_index = int(os.getenv("LOCAL_RANK", -1))
        if data_args.streaming:
            kwargs = {}
        else:
            kwargs = {
                "num_proc": data_args.preprocessing_num_workers,
                # 配置了overwrite_cache为false（默认为false)时，非rank0节点读取cache不再进行map处理
                # 配置了overwrite_cache为true（默认为false)时，所有节点都读取cache不再进行map处理
                "load_from_cache_file": (not data_args.overwrite_cache) or (local_process_index != 0)
            }
        logger.debug(f'Rank: %s, kwargs: %s', local_process_index, kwargs)
        # -----------------convert to sharegpt ---------------------------------------------------------------------------
        convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr, dataset_dir=data_args.dataset_dir)
        if data_args.streaming:
            train_dataset = train_dataset.map(
                convert_func,
                batched=False,
                remove_columns=(list(next(iter(train_dataset)).keys())),
                **kwargs,
            )
            if val_dataset:
                val_dataset = val_dataset.map(
                    convert_func,
                    batched=False,
                    remove_columns=(list(next(iter(val_dataset)).keys())),
                    **kwargs,
                )
        else:
            train_dataset = train_dataset.map(
                convert_func,
                batched=False,
                remove_columns=(list(next(iter(train_dataset)).keys())),
                desc=f"Rank {local_process_index}, converting format of train_dataset",
                **kwargs,
            )
            if val_dataset:
                val_dataset = val_dataset.map(
                    convert_func,
                    batched=False,
                    remove_columns=(list(next(iter(val_dataset)).keys())),
                    desc=f"Rank {local_process_index}, converting format of val_dataset",
                    **kwargs,
                )

        # -----------------convert text to token id ----------------------------------------------------------------------
        if dataset_attr.ranking:
            preprocess_func = partial(
                preprocess_pairwise_dataset,
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                data_args=data_args,
            )
        else:
            preprocess_func = partial(
                preprocess_supervised_dataset,
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                data_args=data_args,
            )

        if data_args.streaming:
            train_dataset = train_dataset.map(
                preprocess_func,
                batched=True,
                batch_size=data_args.preprocessing_batch_size,
                remove_columns=(list(next(iter(train_dataset)).keys())),
                **kwargs,
            )
            train_dataset = DistributedIterableDataset(train_dataset)
            if val_dataset:
                val_dataset = val_dataset.map(
                    preprocess_func,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=(list(next(iter(val_dataset)).keys())),
                    **kwargs,
                )
                val_dataset = DistributedIterableDataset(val_dataset)
                return train_dataset, val_dataset
        else:
            train_dataset = train_dataset.map(
                preprocess_func,
                batched=True,
                batch_size=data_args.preprocessing_batch_size,
                remove_columns=(list(next(iter(train_dataset)).keys())),
                desc=f"Rank {local_process_index}, running tokenizer on train_dataset",
                **kwargs,
            )
            if val_dataset:
                val_dataset = val_dataset.map(
                    preprocess_func,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=(list(next(iter(val_dataset)).keys())),
                    desc=f"Rank {local_process_index}, running tokenizer on val_dataset",
                    **kwargs,
                )
                return train_dataset, val_dataset   
        return train_dataset