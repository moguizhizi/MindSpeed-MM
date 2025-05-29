#!/user/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from mindspeed_rl.utils.compute import compute_log_probs
from mindspeed_rl.utils.pad_process import truncate_middle_and_pad
from mindspeed_rl.utils.bert_padding import pad_input


class BaseLossFunc(ABC):
    def __init__(self):
        pass

    def add_loss_meta_info(self, meta_info: Dict):
        """
        添加计算loss所需要的超参信息，子类必须实现
        param: meta_info: 超参信息
        """
        pass

    @abstractmethod
    def compute_loss(self, output: torch.Tensor,
                     batch: Dict[str, torch.Tensor],
                     forward_only=False, 
                     max_log_prob_seq_len=0,
                     config_micro_batch_size=1,
                     non_loss_data=True) -> Tuple[torch.Tensor, Dict]:
        """
        计算损失函数，子类必须实现。
        :param output: 模型的输出 logits。
        :param batch: 输入数据，包含 responses、attention_mask 等。
        :param forward_only
        :param max_log_prob_seq_len 最大的log_prob长度
        :param config_micro_batch_size update的微批量大小
        :return: 损失值和统计信息。
        """
        pass

    @staticmethod
    def _get_compute_log_probs_input(output: torch.Tensor, batch: Dict[str, torch.Tensor]):
        if 'responses' not in batch:
            raise ValueError("The responses is None")
        responses = batch['responses']
        truncate_lengths = torch.cat([batch['prompt_length'], batch['prompt_length'] + batch['response_length']], dim=1) - 1
        logits = truncate_middle_and_pad(responses, output, truncate_lengths)
        return responses, logits

    def compute_log_probs(self, output: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = output['logits']

        padding_free = True
        if padding_free:
            logits_rmpad = output.squeeze(0)
            log_probs = compute_log_probs(logits=logits_rmpad, labels=batch['input_ids_rmpad_rolled'])
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=batch['indices'], batch=batch['batch_size'], seqlen=batch['seqlen']
            )
            log_probs = full_log_probs.squeeze(-1)[:, -batch['responses'].size(1) - 1 : -1]
            return log_probs

        responses, logits = self._get_compute_log_probs_input(output, batch)
        log_probs = compute_log_probs(logits, responses)
        return log_probs



