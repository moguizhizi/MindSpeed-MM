#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : qwen2vl_resplit.py
@Time    : 2025/01/14
@Desc    : mindspeed-mm训练出的模型重新切分成新的pp配置
"""
from tqdm import tqdm

from checkpoint.qwen2vl_mm_to_hf import merge_by_tp
from checkpoint.utils import (ConvertResplitConfig, merge_pp_index, split_model_by_pipeline, save_by_pp, load_from_mm,
                              split_by_tp)


def main(cfg: ConvertResplitConfig):
    source = cfg.source_parallel_config
    target = cfg.target_parallel_config
    tp_state_dicts = load_from_mm(cfg.source_dir, source.vit_pp_layers, source.llm_pp_layers, source.tp_size)
    state_dict = merge_by_tp(tp_state_dicts, source.tp_size)
    tp_state_dicts = split_by_tp(state_dict, target.tp_size)
    pp_split = merge_pp_index(target.vit_pp_layers, target.llm_pp_layers)

    for tp_rank, tp_state_dict in enumerate(tqdm(tp_state_dicts, desc="tp step")):
        pp_state_dicts = split_model_by_pipeline(tp_state_dict, pp_split)
        save_by_pp(pp_state_dicts, cfg.target_dir, tp_rank=tp_rank)
