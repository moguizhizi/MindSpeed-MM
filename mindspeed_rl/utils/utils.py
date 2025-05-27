# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
import sys

import time
import math
import random
from functools import wraps
from typing import Dict, List

import omegaconf
import numpy as np
import torch
import torch_npu
from torch import Tensor


def get_current_dp_range_indexes(experience_count, assign_batch_size, current_dp_rank=0):
    all_indexes = list(range(assign_batch_size * current_dp_rank, assign_batch_size * (current_dp_rank + 1)))
    return [all_indexes[i:i + experience_count] for i in range(0, len(all_indexes), experience_count)]


def synchronize_time():
    """Synchronize training start time across all distributed processes."""
    cur_time = time.time()
    start_time_tensor = torch.tensor([cur_time], dtype=torch.float, device='cuda')
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    min_time = time.asctime(time.localtime(start_time_tensor.item()))
    return min_time


def generate_position_ids(input_ids: Tensor):
    position_ids_list = []
    for i in range(input_ids.size(0)):
        position_ids = list(range(input_ids.size(1)))
        position_ids_list.append(position_ids)
    return position_ids_list


def generate_mask(data_pad: Tensor, seq_lengths: Tensor):
    seq_masks = []
    max_length = data_pad.size(1)
    seq_lengths = seq_lengths.flatten().tolist()
    for seq_length in seq_lengths:
        seq_pad_length = max_length - seq_length
        seq_mask = torch.ones(max_length)
        if seq_pad_length > 0:
            seq_mask[-seq_pad_length:] = 0
        seq_masks.append(seq_mask.numpy().tolist())
    return torch.tensor(seq_masks, dtype=torch.int64)


def get_tune_attention_mask(attention_mask_1d, reset_attention_mask=True, tokenizer_padding_side="right"):
    """
    Generate the final attention mask based on the input parameters and attention mask.

    Args:
        reset_attention_mask: Indicating whether to reset the attention mask.
        tokenizer_padding_side: Indicating the padding direction of the tokenizer, which can be "left" or "right".
        attention_mask_1d: A 1D attention mask tensor.

    Returns:
        The final attention mask tensor.
    """
    micro_batch_size, seq_length = attention_mask_1d.size()
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1

    if tokenizer_padding_side == "left":
        attention_mask = torch.tril(
            torch.ones(seq_length, seq_length, device=attention_mask_1d.device, dtype=torch.bool)).view(1, 1,
                                                                                                        seq_length,
                                                                                                        seq_length)
        attention_mask_tran = attention_mask_1d.view(seq_length, 1, -1)
        attention_mask = attention_mask.masked_fill((attention_mask_tran < 0.5).view(-1, 1, 1, seq_length), value=0)
    else:
        attention_mask = torch.tril(torch.ones(
            (att_mask_batch, seq_length, seq_length), device=attention_mask_1d.device)).view(
            att_mask_batch, 1, seq_length, seq_length)
    attention_mask = attention_mask.masked_fill((attention_mask_1d < 0.5).view(-1, 1, 1, seq_length), value=0)
    attention_mask = (attention_mask < 0.5)
    return attention_mask


def append_to_dict(data: Dict, new_data: Dict):
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


def num_floating_point_operations(args, batch_size):
    """
    Calculate the number of floating-point operations for a given model configuration and batch size.
    
    Args:
        args (object): An object containing various model configuration parameters, including:
            - kv_channels: The number of key-value channels in attention layers.
            - num_attention_heads: The number of attention heads in the model.
            - hidden_size: The dimensionality of the hidden layers.
            - num_layers: The number of hidden layers in the model.
            - seq_length: The length of input sequences.
            - group_query_attention: A boolean indicating whether to use group query attention.
            - num_experts: The number of experts in the Mixture of Experts (MoE) layer.
            - moe_router_topk: The number of top experts to route inputs to in MoE.
            - swiglu: A boolean indicating whether to use SwiGLU activation.
            - padded_vocab_size: The size of the padded vocabulary.
        batch_size (int): The number of samples processed in one batch.

    Returns:
        int: The total number of floating-point operations required for the model with the given configuration and batch size.
    """
    # Attention projection size.
    query_projection_size = args.kv_channels * args.num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size
    # Group Query Attention.
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
    # MoE.
    num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    return (
            12
            * batch_size
            * args.seq_length
            * args.num_layers
            * args.hidden_size
            * args.hidden_size
            * (
                # Attention.
                    (
                            (
                                    1
                                    + (args.num_query_groups / args.num_attention_heads)
                                    + (args.seq_length / args.hidden_size)
                            ) * query_projection_to_hidden_size_ratio
                    )
                    # MLP.
                    + (
                            (args.ffn_hidden_size / args.hidden_size)
                            * num_experts_routed_to
                            * gated_linear_multiplier
                    )
                    # Logit.
                    + (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))
            )
    )


def get_batch_metrices_mean(metrics_list: List[Dict]) -> Dict[str, Tensor]:
    """
    Calculate the mean of each metric across a list of metric dictionaries.

    Args:
        metrics_list: A list of dictionaries, where each dictionary contains metrics as key-value pairs.

    Returns:
        metrics_mean: A dictionary where each key is a metric name and 
                      each value is the mean of that metric across all batches.
    """
    batch_metrics = {}
    for metrics in metrics_list:
        if metrics:
            append_to_dict(batch_metrics, metrics)
    metrics_mean = {key: torch.tensor(value).mean() for key, value in batch_metrics.items()}
    return metrics_mean


def metrics_post_processing(metrics) -> Dict[str, Tensor]:
    """
    Calculate the mean of each metric across a list of metric dictionaries.

    Args:
        metrics_list: A list of dictionaries, where each dictionary contains metrics as key-value pairs.

    Returns:
        metrics_mean: A dictionary where each key is a metric name and 
                      each value is the mean of that metric across all batches.
    """
    new_metrics = {}
    for key, value in metrics.metric.items():
        if "timing" in key:
            if isinstance(value, list):
                new_metrics[key] = metrics.compute_max(key, value) - metrics.compute_min(key, value)
            else:
                new_metrics[key] = value
        elif "start_time" in key:
            if isinstance(value, list):
                new_metrics[key] = metrics.compute_min(key, value)
            else:
                new_metrics[key] = value
        elif "end_time" in key:
            if isinstance(value, list):
                new_metrics[key] = metrics.compute_max(key, value)
            else:
                new_metrics[key] = value
        elif isinstance(value, list):
            new_metrics[key] = metrics.compute_mean(key, value)
        else:
            new_metrics[key] = value
    return new_metrics


def metrics_sort(metrics, time_all) -> Dict[str, Tensor]:
    custom_order = ['timing/all', 'timing/update', 'timing/resharding_to_train', 'timing/rollout', 'timing/resharding_to_infer',
                    'timing/old_log_p', 'timing/reference_model', 'timing/non_overlap_reference_model']
    special_keys = ['timing/non_overlap_rule_reward', 'timing/non_overlap_reward_model', 'timing/rule_reward',
                    'timing/reward_model', 'timing/adv', 'timing/non_overlap_adv', 'timing/onload', 'timing/offload']

    old_log_p_end_time = metrics.pop('end_time/old_log_p', None)
    end_adv_time = metrics.pop('end_time/end_adv_time', None)

    reference_start_time = metrics.pop('start_time/reference_model', None)
    reference_end_time = metrics.pop('end_time/reference', None)

    if old_log_p_end_time is None:
        old_log_p_end_time = reference_end_time
        custom_order.remove('timing/old_log_p')
    non_overlap_reference_model_time = max(reference_end_time - max(old_log_p_end_time, reference_start_time), 0)
    non_overlap_adv_time = max(max(old_log_p_end_time, end_adv_time) - old_log_p_end_time, 0)

    if "timing/rule_reward" in metrics.keys():
        reward_start_time = metrics.pop('start_time/rule_reward', None)
        reward_end_time = metrics.pop('end_time/rule_reward', None)
        non_overlap_rule_reward_time = max(reward_end_time - max(old_log_p_end_time, reward_start_time), 0)   
        metrics["timing/non_overlap_rule_reward"] = non_overlap_rule_reward_time
    if "timing/reward_model" in metrics.keys():
        reward_start_time = metrics.pop('start_time/reward_model', None)
        reward_end_time = metrics.pop('end_time/reward_model', None)
        non_overlap_reward_model_time = max(reward_end_time - max(old_log_p_end_time, reward_start_time), 0)  
        metrics["timing/non_overlap_reward_model"] = non_overlap_reward_model_time
 

    metrics["timing/non_overlap_reference_model"] = non_overlap_reference_model_time
    metrics["timing/non_overlap_adv"] = non_overlap_adv_time
    metrics["timing/all"] = time_all

    sort_metrics = dict(sorted(metrics.items()))

    keys_to_move = [key for key in sort_metrics.keys() if key in special_keys]
    remaining_keys = []
    for key in sort_metrics:
        if key not in custom_order and key not in special_keys:
            remaining_keys.append(key)
    new_order = custom_order + keys_to_move + remaining_keys
    sorted_metric = {key: sort_metrics[key] for key in new_order}
    return sorted_metric


def compute_tps(compute_kwargs, metrics_result, gbs, n_samples, time_all):
    
    actor_resource = compute_kwargs.get('actor_resource', {})
    reference_resource = compute_kwargs.get('reference_resource', {})
    reward_resource = compute_kwargs.get('reward_resource', None)
    actor_resource_only = compute_kwargs.get('use_integrated_worker', False)

    actor_npus = actor_resource.get('num_npus', 0)
    reference_npus = reference_resource.get('num_npus', 0)
    reward_npus = reward_resource.get('num_npus', 0) if reward_resource is not None else 0

    world_size = actor_npus + reference_npus + reward_npus if not actor_resource_only else actor_npus
    tps = (metrics_result['response_length/mean'] + metrics_result['prompt_length/mean']) * gbs * n_samples / world_size / time_all
    return tps


def compute_vllm_throughput(compute_kwargs, metrics_result, gbs, n_samples, time_rollout):
    actor_resource = compute_kwargs.get('actor_resource', {})
    reference_resource = compute_kwargs.get('reference_resource', {})
    reward_resource = compute_kwargs.get('reward_resource', None)
    actor_resource_only = compute_kwargs.get('use_integrated_worker', False)

    actor_npus = actor_resource.get('num_npus', 0)
    reference_npus = reference_resource.get('num_npus', 0)
    reward_npus = reward_resource.get('num_npus', 0) if reward_resource is not None else 0

    world_size = actor_npus + reference_npus + reward_npus if not actor_resource_only else actor_npus
    vllm_throughput = metrics_result['response_length/mean'] * gbs * n_samples / world_size / time_rollout
    return vllm_throughput


def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['HCCL_DETERMINISTIC'] = str(True)
    os.environ['LCCL_DETERMINISTIC'] = str(1)
    os.environ['CLOSE_MATMUL_K_SHIFT'] = str(1)
    os.environ['ATB_MATMUL_SHUFFLE_K_ENABLE'] = "0"
    os.environ['ATB_LLM_LCOC_ENABLE'] = "0"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)


def parse_args_from_config(config):
    # model configs
    # Parsing utils parameters.
    for key, value in config.items():  # config is transformed into a dict
        if isinstance(value, omegaconf.listconfig.ListConfig):
            sys.argv.append(f"--{key.replace('_', '-')}")
            for i in value:
                sys.argv.append(f"{i}")
        elif isinstance(value, bool):
            if value:
                sys.argv.append(f"--{key.replace('_', '-')}")
        elif value is None:
            continue
        else:
            sys.argv.append(f"--{key.replace('_', '-')}={value}")


def get_least_common_multiple(num_1: int, num_2: int):
    return abs(num_1 * num_2) // math.gcd(num_1, num_2)


def get_attr_wrapped_model(model, attr, allow_none=True, return_model_obj=False):
    if isinstance(model, list):
        raise RuntimeError("_get_attr_wrapped_model given a list of models")

    if allow_none:

        def condition(model, attr):
            return not hasattr(model, attr)

    else:

        def condition(model, attr):
            return getattr(model, attr, None) is None

    while condition(model, attr):
        if not hasattr(model, "module"):
            raise RuntimeError(f"_get_attr_wrapped_model couldn't find attribute {attr}")

        model = model.module

    if return_model_obj:
        return model
    return getattr(model, attr)


def get_grpo_profiler(profiler_config, role: str = None):
    args = profiler_config
    if not args or not args.profile:
        return None

    profiler_this_rank = False
    if args.profile_ranks == "all":
        profiler_this_rank = True
    else:
        try:
            ranks = list(args.profile_ranks)
        except (TypeError, AttributeError):
            ranks = [0]
        if (torch.distributed.get_rank() in ranks):
            profiler_this_rank = True
    if not profiler_this_rank:
        return None

    if args.profile_level == 'level_none':
        profiler_level = torch_npu.profiler.ProfilerLevel.Level_none
    elif args.profile_level == 'level0':
        profiler_level = torch_npu.profiler.ProfilerLevel.Level0
    elif args.profile_level == 'level1':
        profiler_level = torch_npu.profiler.ProfilerLevel.Level1
    elif args.profile_level == 'level2':
        profiler_level = torch_npu.profiler.ProfilerLevel.Level2
    else:
        raise ValueError(f"profiler_level only supports level0,"
                         f" 1, 2, and level_none, but gets {args.profile_level}")

    if args.profile_export_type == 'text':
        profile_export_type = torch_npu.profiler.ExportType.Text
    elif args.profile_export_type == 'db':
        profile_export_type = torch_npu.profiler.ExportType.Db
    else:
        raise ValueError(f"profile_export_type only supports text or db,"
                         f"but gets {args.export_type}")

    base_path = args.profile_save_path
    if role:
        profile_save_path = os.path.join(base_path, role)
    else:
        profile_save_path = base_path

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=profiler_level,
        export_type=profile_export_type,
        data_simplification=True,
        msprof_tx=args.mstx
    )
    if args.stage == "all":
        skip_first = args.profile_step_start - 1
        active = args.profile_step_end - args.profile_step_start
    else:
        skip_first = 0
        active = 1

    activites = []
    if args.profile_with_npu:
        activites.append(torch_npu.profiler.ProfilerActivity.NPU)
    if args.profile_with_cpu:
        activites.append(torch_npu.profiler.ProfilerActivity.CPU)

    prof = torch_npu.profiler.profile(
        with_modules=args.profile_with_module,
        record_shapes=args.profile_record_shapes,
        profile_memory=args.profile_with_memory,
        activities=activites,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=active, repeat=1, skip_first=skip_first),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_save_path, analyse_flag=args.profile_analysis),
        experimental_config=experimental_config)

    return prof


def mstx_timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        range_id = torch_npu.npu.mstx.range_start(func.__qualname__)
        result = func(*args, **kwargs)
        torch_npu.npu.mstx.range_end(range_id)
        return result
    return wrapper


def profiler_start(profiler_config, role="profiler_data", profiler_iteration=None):
    if not profiler_config:
        return None
    if profiler_iteration is not None and (
            profiler_iteration < profiler_config.profile_step_start or 
            profiler_iteration >= profiler_config.profile_step_end):
        return None
    if profiler_config.stage == "all" and role != profiler_config.role:
        return None
    if profiler_config.stage != "all" and role != profiler_config.stage:
        return None
    profiler = get_grpo_profiler(profiler_config, role)
    if not profiler:
        return None
    profiler.start()
    return profiler


def profiler_step(profiler):
    if profiler:
        profiler.step()