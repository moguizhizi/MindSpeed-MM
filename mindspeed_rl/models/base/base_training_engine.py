# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import random
from abc import ABC
from typing import Callable, Dict, List
from functools import partial

import torch
from torch.utils.data import DataLoader
import itertools

from mindspeed_rl.models.loss.base_loss_func import BaseLossFunc
from mindspeed_rl.models.loss.loss_func_factory import LossFuncFactory
from mindspeed_rl.utils.utils import (
    append_to_dict, generate_mask, generate_position_ids, get_tune_attention_mask
)
from mindspeed_rl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
from mindspeed_rl.utils.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from mindspeed_rl.utils.compute import get_parallel_state

class BaseTrainingEngine(ABC):
    """
    Initialize the base trainning engine.

    Args:
        model: The network model to be trained.
        optimizer: The optimizer for updating model parameters (e.g., Adam).
        opt_param_scheduler: The scheduler for optimizer parameters (e.g., learning rate scheduler).
        beta: float = 0 The weight coefficient for KL divergence (used in algorithms like PPO).
        mini_batch_size_per_dp: int = 1  The size of the mini-batch for each data parallel stage.
        epochs: int = 1 The number of training epochs.
        shuffle_mini_batch: bool = False Whether to shuffle the mini-batch data at each epoch.
        stage(str): str = None The training stage identifier (e.g., ray_grpo).
        kl_ctrl: float = 0.1 Adaptive kl ctrl.
        clip_ratio: float = 0.1 The clipping ratio threshold for PPO (limits the policy update range).
        role: str The role of actor in the RLHF frameworker.
        micro_batch_size: int = 1 Micro batch size for actor rollout.
        forward_backward_func: Callable = None The forward-backward function for distributed training.
        **kwargs: Additional keyword arguments.
    """
    def __init__(
            self,
            model,
            optimizer=None,
            opt_param_scheduler=None,
            beta: float = 0,
            mini_batch_size_per_dp: int = 1,
            epochs: int = 1,
            shuffle_mini_batch: bool = False,
            stage: str = None,
            kl_ctrl: float = 0.0,
            clip_ratio: float = 0.1,
            temperature: float = 1.0,
            role: str = None,
            micro_batch_size: int = 1,
            forward_micro_batch_size: int = 1,
            vit_micro_batch_size: int = 1,
            use_dynamic_bsz: bool = False,
            max_log_prob_seq_len_forward: int = 1,
            max_log_prob_seq_len_update: int = 1,
            forward_backward_func: Callable = None,
            **kwargs):
        self.forward_backward_func = forward_backward_func
        self.micro_batch_size = micro_batch_size
        self.forward_micro_batch_size = forward_micro_batch_size
        self.vit_micro_batch_size = vit_micro_batch_size
        self.model = model
        self.optimizer = optimizer
        self.opt_param_scheduler = opt_param_scheduler
        self.beta = beta
        self.mini_batch_size_per_dp = mini_batch_size_per_dp
        self.epochs = epochs
        self.shuffle_mini_batch = shuffle_mini_batch
        self.stage = stage
        self.role = role
        self.kl_ctrl = kl_ctrl
        self.clip_ratio = clip_ratio
        self.temperature = temperature
        self.use_dynamic_bsz = use_dynamic_bsz
        self.max_log_prob_seq_len_forward = max_log_prob_seq_len_forward
        self.max_log_prob_seq_len_update = max_log_prob_seq_len_update
        self.loss_func: BaseLossFunc = LossFuncFactory.get_instance(self.stage, self.role)
        self.kwargs = kwargs

    @staticmethod
    def _split_batches(batch: Dict, batch_size: int, shuffle_mini_batch: bool, dim: int = 0, is_mini_batch: bool = False) -> List[Dict]:
        batches = []
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                split_values = torch.split(value, batch_size, dim)
            elif isinstance(value, List):
                num_batches = (len(value) + batch_size - 1) // batch_size
                if is_mini_batch:
                    split_values = [value[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]
                else:
                    split_values = [torch.concat(value[i * batch_size: (i + 1) * batch_size]) for i in range(num_batches)]
            for index, split_value in enumerate(split_values):
                if index >= len(batches):
                    batches.append({})
                batches[index][key] = split_value

        if shuffle_mini_batch:
            random.shuffle(batches)
        return batches
    
    @staticmethod
    def _split_batches_by_max_length(batch: Dict, max_length: int, shuffle_mini_batch: bool, is_mini_batch: bool = False, dim: int = 0) -> List[Dict]:
        actual_lengths = (batch["response_length"] + batch["input_ids_length"]).squeeze(1).cpu().numpy().tolist()
        partitions = rearrange_micro_batches(actual_lengths, max_length)
        batches = []
        for key,value in batch.items():
            if isinstance(value, torch.Tensor):
                for batch_idx, partition in enumerate(partitions):
                    if batch_idx >= len(batches):
                        batches.append({})
                    batches[batch_idx][key] = value[partition]
            elif isinstance(value, List):
                for batch_idx, partition in enumerate(partitions):
                    if batch_idx >= len(batches):
                        batches.append({})
                    if is_mini_batch:
                        batches[batch_idx][key] = [value[p] for p in partition]
                    else:
                        batches[batch_idx][key] = torch.concat([value[p] for p in partition])
        if shuffle_mini_batch:
            random.shuffle(batches)
        return batches, partitions

    def _forward_backward_batch(self, batch: Dict[str, torch.Tensor], forward_only: bool = False, vit_only: bool = False, llm_only: bool = False):
        if forward_only:
            micro_batch_size = self.vit_micro_batch_size if vit_only else self.forward_micro_batch_size
            split_seq_len = self.max_log_prob_seq_len_forward
        else:
            micro_batch_size = self.micro_batch_size
            split_seq_len = self.max_log_prob_seq_len_update

        if self.use_dynamic_bsz and not vit_only:
            batches,indices = self._split_batches_by_max_length(batch, split_seq_len, self.shuffle_mini_batch) 
        else:
            batches = self._split_batches(batch, batch_size=micro_batch_size,
                                      shuffle_mini_batch=self.shuffle_mini_batch)
        n_micro_batch = len(batches)
        seq_len = batches[0]['input_ids'].shape[1]

        self.loss_func.add_loss_meta_info(self.get_loss_meta_func())
        
        post_process = get_parallel_state().get_pipeline_model_parallel_world_size() == 1 or get_parallel_state().is_pipeline_last_stage()

        def forward_step(batch_iter, model):
            # input_ids, attention_mask, position_ids, process_batch = self._get_forward_batch_info(batch_iter)
            batch_dict = self._get_forward_batch_info(batch_iter, vit_only, llm_only)
            process_batch = batch_dict.pop('batch')
            output = model(**batch_dict)
            if isinstance(output, dict) and "logits" in output:
                output["logits"].div_(self.temperature)
            # output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            return output, partial(self.loss_func.compute_loss, batch=process_batch, forward_only=forward_only,
                                   max_log_prob_seq_len=split_seq_len, config_micro_batch_size=self.micro_batch_size)

        # batch should be a list of batches inside micro-batches
        losses_reduced = self.forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=iter(batches),
            model=self.model,
            num_microbatches=n_micro_batch,
            seq_length=seq_len,
            micro_batch_size=micro_batch_size,
            forward_only=forward_only,
            collect_non_loss_data=forward_only,
        )
        
        if self.use_dynamic_bsz and forward_only and not vit_only and post_process:
            losses_reduced_list = torch.cat(losses_reduced, dim=0)
            indices = list(itertools.chain.from_iterable(indices))
            revert_indices = get_reverse_idx(indices)
            losses_reduced = [losses_reduced_list[[idx,]] for idx in revert_indices]
        return losses_reduced

    def get_loss_meta_func(self) -> Dict:
        """
        获取具体的loss计算超参
        :return: loss计算超参。
        """
        return {}

    @staticmethod
    def _get_forward_batch_info(batch_iter, vit_only: bool = False, llm_only: bool = False):
        batch = next(batch_iter)
        if vit_only:
            batch['vit_only'] = vit_only
            batch['batch'] = batch
            return batch

        input_ids = batch['input_ids']
        # attention_mask_1d = generate_mask(input_ids, batch['prompt_length'] + batch['response_length']).to(
        #     input_ids.device)
        # position_ids = torch.tensor(generate_position_ids(input_ids)).to(input_ids.device)
        # attention_mask = get_tune_attention_mask(attention_mask_1d)
        # return input_ids, attention_mask, position_ids, batch

        delta_position_id = torch.arange(1, input_ids.size(1)-batch['attention_mask'].size(1) + 1, device=batch['position_ids'].device)
        batch_size = input_ids.size(0)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if batch['position_ids'].dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = batch['position_ids'][..., -1:] + delta_position_id
        position_ids = torch.cat([batch['position_ids'], response_position_ids], dim=-1)

        eos_mask = torch.zeros_like(batch['responses'], dtype=torch.bool)
        for token_id in [151645, 151643]:
            eos_mask |= batch['responses'].eq(token_id)

        eos_mask = eos_mask.long()
        eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
        eos_mask = torch.logical_not(eos_mask).to(batch['attention_mask'].dtype)
        attention_mask = torch.cat((batch['attention_mask'], eos_mask), dim=-1)

        batch['attention_mask'] = attention_mask
        position_ids = position_ids.permute(1, 0, 2).clone()

        batch['batch_size'], batch['seqlen'] = input_ids.shape

        input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

        # unpad the position_ids to align the rotary
        if position_ids.dim() == 3:
            position_ids_rmpad = (
                index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                .transpose(0, 1)
                .unsqueeze(1)
            )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
        else:
            position_ids_rmpad = index_first_axis(
                rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
            ).transpose(0, 1)

        # for compute the log_prob
        input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
        input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

        batch['input_ids'] = input_ids_rmpad
        batch['position_ids'] = position_ids_rmpad
        batch['attention_mask'] = None
        batch['indices'] = indices
        batch['input_ids_rmpad_rolled'] = input_ids_rmpad_rolled
        batch['vit_only'] = vit_only
        batch['llm_only'] = llm_only

        batch['batch'] = batch
        return batch

    def post_process_forward_backward_output(self, output: [torch.Tensor],
                                             batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        模型前反向计算结果后处理
        :param output: 模型前反向计算结果
        :param batch: 参与计算的batch数据
        :return: 模型前向计算结果。
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def forward(self, data: Dict, vit_only: bool = False, llm_only: bool = False) -> torch.Tensor:
        """
        模型前向计算
        :param data: 前向计算数据
        :return: 模型前向计算结果。
        """
        for k, v in data.items():
            if isinstance(v, List):
                data[k] = [t.to(next(self.model[0].parameters()).device) for t in v]
            else:
                data[k] = v.to(next(self.model[0].parameters()).device)
        for model_module in self.model:
            model_module.eval()
        with torch.no_grad():
            output = self._forward_backward_batch(data, forward_only=True, vit_only=vit_only, llm_only=llm_only)
            return self.post_process_forward_backward_output(output=output, batch=data)

    def update(self, data: Dict, kl_ctrl=None) -> Dict:
        """
        模型反向更新
        :param data: 反向更新数据
        :param kl_ctrl：KL散度计算controller
        :return: 模型反向计算结果。
        """
        self.kl_ctrl = kl_ctrl
        metrics = {}
        grad_norm_list = []
        for k, v in data.items():
            if v is not None:
                if isinstance(v, List):
                    data[k] = [t.to(next(self.model[0].parameters()).device) for t in v]
                else:
                    data[k] = v.to(next(self.model[0].parameters()).device)
        mini_batches = self._split_batches(data, batch_size=self.mini_batch_size_per_dp,
                                           shuffle_mini_batch=self.shuffle_mini_batch, dim=0, is_mini_batch=True)
        for model_module in self.model:
            model_module.train()
        for _ in range(self.epochs):
            for mini_batch in mini_batches:
                for model_chunk in self.model:
                    model_chunk.zero_grad_buffer()
                self.optimizer.zero_grad()
                metric_micro_batch = self._forward_backward_batch(mini_batch)
                update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()

                if update_successful:
                    increment = self.mini_batch_size_per_dp
                    self.opt_param_scheduler.step(increment=increment)
                grad_norm_list.append(grad_norm) 

                for metric in metric_micro_batch:
                    append_to_dict(metrics, metric)  # append the metric from this micro-batch to global metrics.

            # add empty cache after each compute
            # torch.cuda.empty_cache()
        grad_norm = sum(grad_norm_list) / len(grad_norm_list)
        metrics["grad_norm"] = grad_norm_list
        return metrics
