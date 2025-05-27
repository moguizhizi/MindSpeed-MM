# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import ray
import torch
from transformers import AutoTokenizer

from mindspeed_rl.models.rule_verifier import compute_verifier_score, math_compute_score
from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.trainer.utils.transfer_dock import pad_experience
from mindspeed_rl.utils.utils import get_least_common_multiple, get_current_dp_range_indexes

logger = Loggers("rule_reward")


# @ray.remote
class RuleReward(object):

    def initialize(self, megatron_config, rl_config, tokenizer):
        self.rl_config = rl_config
        self.megatron_config = megatron_config
        self.n_samples_per_prompt = rl_config.n_samples_per_prompt
        self.tokenizer = tokenizer
        self.hf_tokenizer = AutoTokenizer.from_pretrained(megatron_config.tokenizer_name_or_path,
                                                          trust_remote_code=True)

    def init_transfer_dock(self, td, mm_td):
        self.td = td
        self.mm_td = mm_td

    def compute_rm_score(self):
        experience_consumer_stage = 'rule_reward'
        experience_columns = ['prompts', 'responses', 'response_length', *self.megatron_config.dataset_additional_keys]
        experience_count = get_least_common_multiple(self.rl_config.verifier_micro_batch_size,
                                                     self.rl_config.n_samples_per_prompt)
        assign_batch_size = self.megatron_config.global_batch_size * self.rl_config.n_samples_per_prompt
        sorted_indexes = get_current_dp_range_indexes(experience_count=experience_count,
                                                      assign_batch_size=assign_batch_size) if self.rl_config.guarantee_order else None

        pad_token_id = self.tokenizer.pad if self.tokenizer.pad else self.tokenizer.eod
        while not ray.get(self.td.all_consumed.remote(experience_consumer_stage)):
            batch_data, index = ray.get(
                self.td.get_experience.remote(
                    experience_consumer_stage,
                    experience_columns,
                    experience_count,
                    indexes=sorted_indexes.pop(0) if self.rl_config.guarantee_order else None
                )
            )  # cpu数据

            if batch_data and index:
                batch_data = pad_experience(batch_data, pad_token_id) # multiple, tp_size
                if "categories" in batch_data.keys():
                    use_verifier_mask = batch_data["categories"][:, 0].squeeze().bool()
                    selected_index = [index[i] for i in range(len(index)) if use_verifier_mask[i]]
                    index = selected_index
                if not index:
                    continue
                if "categories" in batch_data.keys():
                    batch_data = {key: value[use_verifier_mask] if key != 'prompts' else value[
                        use_verifier_mask[::self.n_samples_per_prompt]] for key, value in batch_data.items()}
                ignore_token = self.tokenizer.pad if self.tokenizer.pad else self.tokenizer.eod

                token_level_rewards, metrics = compute_verifier_score(batch_data, self.megatron_config, self.rl_config,
                                                                      self.hf_tokenizer, ignore_token)
                
                for key, value in metrics.items():
                    ray.get(self.td.update_metrics.remote(key, value=value, cumulate=True))

                output = {"rm_scores": token_level_rewards, "token_level_rewards": token_level_rewards}
                self.td.put_experience.remote(data_dict=output, indexes=index)


@ray.remote
class MMRuleReward(RuleReward):
    def compute_rm_score(self):
        experience_consumer_stage = 'rule_reward'
        experience_columns = ['prompts', 'responses', 'response_length', *self.megatron_config.dataset_additional_keys]
        experience_count = get_least_common_multiple(self.megatron_config.micro_batch_size,
                                                     self.rl_config.n_samples_per_prompt)
        assign_batch_size = self.megatron_config.global_batch_size * self.rl_config.n_samples_per_prompt
        sorted_indexes = get_current_dp_range_indexes(experience_count=experience_count,
                                                      assign_batch_size=assign_batch_size) if self.rl_config.guarantee_order else None

        pad_token_id = self.hf_tokenizer.pad_token_id if self.hf_tokenizer.pad_token_id else self.hf_tokenizer.eos_token_id
        while not ray.get(self.td.all_consumed.remote(experience_consumer_stage)):
            batch_data, index = ray.get(
                self.td.get_experience.remote(
                    experience_consumer_stage,
                    experience_columns,
                    experience_count,
                    indexes=sorted_indexes.pop(0) if self.rl_config.guarantee_order else None
                )
            )  # cpu数据

            if batch_data and index:
                batch_data = pad_experience(batch_data, pad_token_id) # multiple, tp_size
                mm_columns = ray.get(self.mm_td.get_columns.remote(experience_consumer_stage))
                batch_mm_data = ray.get(self.mm_td.get_experience.remote(mm_columns, index))
                batch_data.update(batch_mm_data)

                reward_tensor = torch.zeros((batch_data['responses'].size(0), 1), dtype=torch.float32)
                original_shape = reward_tensor.shape
                responses = batch_data['responses']
                response_strs = self.hf_tokenizer.batch_decode(responses, skip_special_tokens=True)
                labels = [label for label in batch_data['labels'] for _ in range(self.n_samples_per_prompt)]

                for i, (response_str, label) in enumerate(zip(response_strs, labels)):
                    token_level_rewards = math_compute_score(response_str, label)
                    reward_tensor[i, 0] = token_level_rewards
                rm_scores = reward_tensor
                reward_tensor_reshaped = reward_tensor.reshape(-1, self.n_samples_per_prompt)
                reward_mean = reward_tensor_reshaped.mean(dim=1, keepdim=True)
                reward_std = reward_tensor_reshaped.std(dim=1, keepdim=True) + 1e-6  # 避免除零，verl eps是1e-6
                reward_tensor_normalized = (reward_tensor_reshaped - reward_mean) / reward_std
                reward_tensor = reward_tensor_normalized.reshape(original_shape)

                output = {"rm_scores": rm_scores, "token_level_rewards": reward_tensor}
                self.td.put_experience.remote(data_dict=output, indexes=index)
