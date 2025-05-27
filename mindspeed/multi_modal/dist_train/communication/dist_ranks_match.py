# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from itertools import accumulate
import torch.distributed as dist
from ..config import dist_train_config as config


"""
key:int:    cur rank
value:list: dst ranks
"""
_MODEL_COMM_RANKS = {}


def generate_model_comm_ranks(pp_ranks_prev: [[]], tp_ranks_prev: [[]], pp_ranks_last: [[]], tp_ranks_last: [[]]):
    global _MODEL_COMM_RANKS
    if _MODEL_COMM_RANKS and config.get_all_config_size() != 2:
        # If the size is 2, this method is expected to be invoked only once.
        raise RuntimeError(f'Get config size ({config.get_all_config_size()}) is not equal to 2, '
                           f'and _MODEL_COMM_RANKS is initialized.')
    tp_ranks_prev_ = []
    tp_ranks_last_ = []

    # Take the ranks of the last stage of 'prev' and the first stage of 'last'.
    for pp_ranks in pp_ranks_prev:
        for tp_ranks in tp_ranks_prev:
            if pp_ranks[-1] in tp_ranks and tp_ranks not in tp_ranks_prev_:
                tp_ranks_prev_.append(tp_ranks)

    for pp_ranks in pp_ranks_last:
        for tp_ranks in tp_ranks_last:
            if pp_ranks[0] in tp_ranks and tp_ranks not in tp_ranks_last_:
                tp_ranks_last_.append(tp_ranks)

    if not (len(tp_ranks_prev_) and len(tp_ranks_last_)):
        raise ValueError("tp ranks must not empty")

    # Place the TP units with fewer counts at the front and those with more at the back,
    # so that when generating the forward correspondence, it traverses through fewer iterations.
    if len(tp_ranks_prev_) > len(tp_ranks_last_):
        tp_ranks_prev_, tp_ranks_last_ = tp_ranks_last_, tp_ranks_prev_

    # Generate correspondence.
    lens_last = get_size_list(len(tp_ranks_last_), len(tp_ranks_prev_), 1)
    index_for_last = [0] + list(accumulate(lens_last))
    ranks_dict_prev = {}
    for i, prev_ranks in enumerate(tp_ranks_prev_):
        last_ranks = [rank for lst in tp_ranks_last_[index_for_last[i]: index_for_last[i + 1]] for rank in lst]
        num_take_last = lens_last[i]  # The actual number of data sets taken from tp_ranks_last_ in this round.
        num_unit_last = len(tp_ranks_last_[0])

        # Place the elements with fewer counts at the front and those with more at the back,
        # to facilitate the execution of the general logic.
        if len(last_ranks) < len(prev_ranks):
            prev_ranks, last_ranks = last_ranks, prev_ranks
            num_take_last = 1  # Only one sublist will be extracted from tp_ranks_Prev_ in each round.
            num_unit_last = len(tp_ranks_prev_[0])

        # Establish the corresponding relationships.
        per_ranks = get_size_list(len(last_ranks), len(prev_ranks), num_unit_last)
        index_for_prev = [0] + list(accumulate(per_ranks))
        for j, rank_ in enumerate(prev_ranks):
            ranks_dict_prev[rank_] = last_ranks[index_for_prev[j]: index_for_prev[j + 1]]

        print(f"rank={dist.get_rank()}, num_take_last: {num_take_last}, num_unit_last: {num_unit_last}, "
              f"prev: {prev_ranks}, last: {last_ranks}")

    # Conversely, establish the corresponding relationships again;
    # currently, this is only compatible with scenarios where the model is divided into two parts.
    ranks_dict_last = {last: [prev] for prev in ranks_dict_prev for last in ranks_dict_prev.get(prev, None)}
    if None in ranks_dict_last.keys():
        raise KeyError('Found unexpected keys in `ranks_dict_last`')

    # Update data
    keys = ranks_dict_prev.keys() | ranks_dict_last.keys()
    for k in keys:
        _MODEL_COMM_RANKS[k] = _MODEL_COMM_RANKS.get(k, []) + ranks_dict_prev.get(k, []) + ranks_dict_last.get(k, [])


def get_dst_ranks(rank=None):
    global _MODEL_COMM_RANKS
    if rank is None:
        rank = dist.get_rank()

    return _MODEL_COMM_RANKS.get(rank, None)


def clear_model_comm_ranks():
    global _MODEL_COMM_RANKS
    _MODEL_COMM_RANKS = {}


def get_size_list(sum_, len_, base_):
    """
    sum, len, base:
        12, 2, 7 => 12, 2, 6 => [6, 6]             base is too large, let the base cycle subtract 1 first
        15, 2, 5             => [5, 5] => [10, 5]  base is appropriate, try to allocate with multiple of base num
        12, 2, 5             => [5, 5] => [6, 6]   base is too small, try to allocate as much as possible
    """
    if not all(isinstance(num, int) for num in (sum_, len_, base_)):
        raise ValueError("sum_, base_ and len_ must be integers.")
    if base_ <= 0 or len_ <= 0:
        raise ValueError("base_ and len_ cannot be zero.")
    while sum_ // base_ < len_:
        base_ -= 1
    list_base_ = sum_ // len_ // base_ * base_
    list_ = [list_base_ for _ in range(len_)]
    rem_ = sum_ - len_ * list_base_
    base_ = base_ if rem_ % base_ == 0 else 1
    index_ = 0

    while rem_ > 0:
        list_[index_ % len_] += base_
        rem_ -= base_
        index_ += 1

    return list_
