# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import time
import dataclasses
import copy
from typing import Callable

import ray
from torch import nn
import torch
from transformers import AutoConfig

from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.utils.optimizer_module import OptimizerConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.config_cls.profiler_config import ProfilerConfig
from mindspeed_rl.models.actor_rollout_hybrid import ActorRolloutHybrid
from mindspeed_rl.models.rollout.vllm_engine import VLLMInferEngine
from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.workers.base_worker import BaseWorker
from mindspeed_rl.workers.resharding.megatron_sharding_manager import MegatronShardingManager, MegatronOffLoader
# from mindspeed_rl.utils.utils import num_floating_point_operations, get_least_common_multiple, get_attr_wrapped_model
from mindspeed_rl.utils.utils import num_floating_point_operations, get_least_common_multiple, get_attr_wrapped_model, mstx_timer_decorator, profiler_start, profiler_step
from mindspeed_rl.utils.pad_process import remove_padding_and_split_to_list, truncate_rows
from mindspeed_rl.utils.compute import get_parallel_state
from mindspeed_rl.trainer.utils.parallel_state import is_pipeline_last_stage, get_tensor_model_parallel_rank
from mindspeed_rl.trainer.utils.transfer_dock import unpack_mm_experience


class ActorHybridWorkerBase(BaseWorker):
    """
    ActorHybridWorker class. This class implements the hybrid worker logic for training and inference.

    Args:
        megatron_config: MegatronConfig Configuration for Megatron-LM (e.g., model parallelism settings).
        rl_config: RLConfig Configuration for reinforcement learning (e.g., PPO settings).
        generate_config: GenerateConfig Configuration for generation/inference (e.g., vLLM settings).
        model_provider: Callable Function to provide the model instance.
        initialize_func: Callable Function to initialize the model and environment.
        profiler_config: ProfilerConfig, Configuration for profiling.
        tokenizer: BaseTokenizer = None Object to retrieve the tokenizer.
        get_megatron_module: Callable = megatron_module from get_megatron_module.
        **kwargs: Additional parameters for base class argument passing.
    """

    def __init__(
            self,
            megatron_config: MegatronConfig,
            rl_config: RLConfig,
            generate_config: GenerateConfig,
            model_provider: Callable,
            initialize_func: Callable,
            profiler_config: ProfilerConfig = None,
            tokenizer: BaseTokenizer = None,
            get_megatron_module: Callable = None,
            **kwargs
    ):
        super().__init__(
            megatron_config,
            rl_config,
            generate_config,
            model_provider=model_provider,
            initialize_func=initialize_func,
            profiler_config=profiler_config,
            tokenizer=tokenizer,
            get_megatron_module=get_megatron_module,
            **kwargs
        )

        self.num_floating_point_operations_so_far = 0
        self.actor_hybrid = None
        self.actor_offloader = None
        self.prof_iteration = 1

    def initialize(self):
        self.setup_distributed_rank()
        self.model, self.optimizer, self.opt_param_scheduler = self._build_model_optimizer()
        self._set_no_sync_func()
        self.actor_offloader = MegatronOffLoader(self.model, self.optimizer)
        if self.generate_config.offload_train_optimizer:
            self.actor_offloader.offload_optimizer()
        if self.generate_config.offload_train_grad:
            self.actor_offloader.offload_grad()
        if self.generate_config.offload_train_param:
            self.actor_offloader.offload_param()

        self.inference_model = self._build_rollout()
        self.sharding_manager = self._build_sharding_manager()

        self.actor_hybrid = ActorRolloutHybrid(
            self.model,
            optimizer=self.optimizer,
            opt_param_scheduler=self.opt_param_scheduler,
            inference_model=self.inference_model,
            sharding_manager=self.sharding_manager,
            beta=self.rl_config.beta,
            mini_batch_size_per_dp=self.rl_config.mini_batch_size
                                   // self.parallel_state.get_data_parallel_world_size(),
            epochs=self.rl_config.epochs,
            shuffle_mini_batch=self.rl_config.shuffle_mini_batch,
            generate_config=self.generate_config,
            stage=self.megatron_config.stage,
            forward_backward_func=self.forward_backward_func,
            clip_ratio=self.rl_config.clip_ratio,
            micro_batch_size=self.megatron_config.micro_batch_size,
            temperature=self.generate_config.sampling_config["temperature"],
            use_dynamic_bsz=self.megatron_config.use_dynamic_bsz,
            max_log_prob_seq_len_forward=self.megatron_config.max_log_prob_seq_len_forward,
            max_log_prob_seq_len_update=self.megatron_config.max_log_prob_seq_len_update,
            forward_micro_batch_size=self.megatron_config.actor_log_prob_micro_batch_size,
            vit_micro_batch_size=self.megatron_config.actor_image_embeds_micro_batch_size
        )
        self.empty_cache()

    def init_transfer_dock(self, td, mm_td):
        self.td = td
        self.mm_td = mm_td
        self.empty_cache()

    def get_iteration(self):
        return self.args.iteration

    def get_consumed_train_samples(self):
        return self.args.consumed_train_samples

    @mstx_timer_decorator
    def update(self, kl_ctrl=None, skip_actor_log_prob=False):
        if skip_actor_log_prob:
            self.sharding_manager.enter_forward_mode()
        start_sharding_enter_train = time.time()
        self.sharding_manager.enter_train_mode()
        sharding_train_interval = time.time() - start_sharding_enter_train

        self.args.curr_iteration = self.iteration

        experience_consumer_stage = 'actor_train'

        experience_columns = ['responses', 'advantages', 'old_log_prob',
                             'ref_log_prob', 'input_ids', 'response_length', 'prompt_length', 'attention_mask', 'position_ids', 'input_ids_length']

        experience_count = self.megatron_config.global_batch_size // self.parallel_state.get_data_parallel_world_size()
        if skip_actor_log_prob:
            experience_columns.remove('old_log_prob')

        #get lr
        learning_rate = None
        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr']
        ray.get(self.td.update_metrics.remote(key='grpo/lr', value=learning_rate))
        sorted_indexes = self.get_dp_range_indexes(experience_count,
                                                   use_vllm=False) if self.rl_config.guarantee_order else None
        actor_update_profiler = profiler_start(self.profiler_config, role="actor_update",
                                        profiler_iteration=self.prof_iteration)
        start_time_defined = False
        count = 0
        while self.all_consumed(experience_consumer_stage, sorted_indexes) > 0:
            batch_data, index = self.dispatch_transfer_dock_data(experience_consumer_stage,
                                                                 experience_columns,
                                                                 experience_count,
                                                                 self.megatron_config.tensor_model_parallel_size,
                                                                 indexes=sorted_indexes.pop(
                                                                     0) if self.rl_config.guarantee_order else None,
                                                                 get_n_samples=False)
            if not start_time_defined:
                start_time = time.time()
                start_time_defined = True
            if batch_data and index:
                metrics = self.actor_hybrid.update_actor(batch_data, kl_ctrl)

                # self.empty_cache()
                self.args.consumed_train_samples += self.megatron_config.global_batch_size // self.rl_config.n_samples_per_prompt
                self.num_floating_point_operations_so_far += num_floating_point_operations(self.args,
                                                                                           self.megatron_config.global_batch_size)
                if self.parallel_state.is_pipeline_last_stage() and self.parallel_state.get_tensor_model_parallel_rank() == 0:
                    ray.get(self.td.update_metrics.remote(value=metrics, cumulate=True))
                    ray.get(
                        self.td.update_metrics.remote(
                            "timing/update",
                            value=[round(time.time(), 4), round(start_time, 4)],
                            cumulate=True
                        )
                    )

        self.iteration += 1
        profiler_step(actor_update_profiler)
        self.prof_iteration += 1
        start_sharding_exit_train = time.time()
        self.sharding_manager.exit_train_mode()
        sharding_train_interval += (time.time() - start_sharding_exit_train)
        ray.get(
            self.td.update_metrics.remote(
                "timing/resharding_to_train",
                value=[sharding_train_interval],
                cumulate=True
            )
        )

    def save_ckpt(self, iteration: int):
        self.save_checkpoint(iteration, self.model, self.optimizer, self.opt_param_scheduler,
                             self.num_floating_point_operations_so_far)

    @mstx_timer_decorator
    def generate_sequences(self):
        start_sharding_enter_infer = time.time()
        self.sharding_manager.enter_infer_mode()
        sharding_infer_interval = time.time() - start_sharding_enter_infer

        experience_consumer_stage = 'actor_rollout'
        experience_columns = ['prompts', 'prompt_length', 'input_ids', 'input_ids_length']
        experience_count = get_least_common_multiple(self.generate_config.micro_batch_size,
                                                     self.rl_config.n_samples_per_prompt)

        pad_token_id = self.tokenizer.pad if self.tokenizer.pad else self.tokenizer.eod
        sorted_indexes = self.get_dp_range_indexes(experience_count,
                                                   use_vllm=True) if self.rl_config.guarantee_order else None
        actor_generate_profiler = profiler_start(self.profiler_config, role="actor_generate",
                                                 profiler_iteration=self.prof_iteration)
        start_time_defined = False
        while self.all_consumed(experience_consumer_stage, sorted_indexes, use_vllm=True) > 0:
            batch_data, index = self.dispatch_transfer_dock_data(
                experience_consumer_stage,
                experience_columns,
                experience_count,
                tp_size=self.megatron_config.tensor_model_parallel_size,
                indexes=sorted_indexes.pop(0) if self.rl_config.guarantee_order else None,
                use_vllm=True
            )
            if not start_time_defined:
                start_time = time.time()
                start_time_defined = True
            if batch_data and index:
                indexes = list(range(0, experience_count, self.rl_config.n_samples_per_prompt))
                if self.rl_config.reuse_image_embeds:
                    prompts_data = batch_data['input_ids'][indexes]
                    prompt_length_data = batch_data['input_ids_length'][indexes]
                    # preprocess, remove padding
                    prompts = truncate_rows(prompts_data, prompt_length_data, left_pad=True)
                else:
                    prompts_data = batch_data['prompts'][indexes]
                    prompt_length_data = batch_data['prompt_length'][indexes]
                    # preprocess, remove padding
                    prompts = truncate_rows(prompts_data, prompt_length_data)
                prompts_list = [prompt.numpy().tolist() for prompt in prompts]

                responses_pad_right = self.actor_hybrid.generate_sequences(prompts_list, batch_data)
                responses = remove_padding_and_split_to_list(responses_pad_right, self.tokenizer.eod, pad_token_id)

                responses_length = [torch.tensor([len(response)]) for response in responses]

                prompts_data = prompts
                prompts = []
                for prompt in batch_data['input_ids'][indexes].unbind():
                    for _ in range(self.rl_config.n_samples_per_prompt):
                        prompts.append(copy.deepcopy(prompt))

                input_ids_list = []
                for prompt, response in zip(prompts, responses):
                    input_ids_list.append(torch.cat((prompt.cpu(), response.cpu()), dim=0))

                outputs = {
                    'responses': responses,
                    'input_ids': input_ids_list,
                    'response_length': responses_length
                }
                self.collect_transfer_dock_data(outputs, index, use_vllm=True)
                end_time = time.time()
                ray.get(
                        self.td.update_metrics.remote(
                            "timing/rollout",
                            value=[round(end_time, 4), round(start_time, 4)],
                            cumulate=True
                        )
                )

        profiler_step(actor_generate_profiler)
        start_sharding_exit_infer = time.time()
        self.sharding_manager.exit_infer_mode()
        sharding_infer_interval += (time.time() - start_sharding_exit_infer)
        ray.get(
            self.td.update_metrics.remote(
                "timing/resharding_to_infer",
                value=[sharding_infer_interval],
                cumulate=True
            )
        )

    @mstx_timer_decorator
    def compute_image_embeds(self):
        experience_consumer_stage = 'actor_image_embeds'
        experience_columns = ['input_ids']
        experience_count = get_least_common_multiple(self.megatron_config.actor_image_embeds_micro_batch_size,
                                                     self.rl_config.n_samples_per_prompt)
        sorted_indexes = self.get_dp_range_indexes(experience_count,
                                                   use_vllm=False) if self.rl_config.guarantee_order else None

        start_time_defined = False
        while self.all_consumed(experience_consumer_stage, sorted_indexes) > 0:
            batch_data, index = self.dispatch_transfer_dock_data(experience_consumer_stage,
                                                                 experience_columns,
                                                                 experience_count,
                                                                 tp_size=self.megatron_config.tensor_model_parallel_size,
                                                                 indexes=sorted_indexes.pop(
                                                                     0) if self.rl_config.guarantee_order else None,
                                                                 get_n_samples=True)
            if not start_time_defined:
                start_time = time.time()
                start_time_defined = True
            if batch_data and index:
                indexes = list(range(0, experience_count, self.rl_config.n_samples_per_prompt))
                batch_data['input_ids'] = batch_data['input_ids'][indexes]
                output, batch = self.actor_hybrid.compute_image_embeds(batch_data)
                if self.parallel_state.is_pipeline_last_stage(ignore_virtual=True):

                    data = {
                                "vit_embeds": output[0].squeeze(1).cpu(),
                                "image_grid_thw": batch_data['image_grid_thw'],
                                "image_num": batch_data['image_num'],
                                "video_num": batch_data['video_num']
                            }

                    data = unpack_mm_experience(data)

                    output = {'vit_embeds': data['vit_embeds']}
                    index = [i // self.rl_config.n_samples_per_prompt for i in index[::self.rl_config.n_samples_per_prompt]]
                    self.collect_transfer_dock_mm_data(output, index)
                end_time = time.time()
                ray.get(
                        self.td.update_metrics.remote(
                            "timing/vit_image_emb",
                            value=[round(end_time, 4), round(start_time, 4)],
                            cumulate=True
                        )
                )

    @mstx_timer_decorator
    def compute_log_prob(self):
        self.sharding_manager.enter_forward_mode()

        experience_consumer_stage = 'actor_log_prob'
        experience_columns = ['input_ids', 'responses', 'response_length', 'prompt_length', 'attention_mask', 'position_ids', 'input_ids_length']
        experience_count = get_least_common_multiple(self.megatron_config.actor_log_prob_micro_batch_size,
                                                     self.rl_config.n_samples_per_prompt)
        sorted_indexes = self.get_dp_range_indexes(experience_count,
                                                   use_vllm=False) if self.rl_config.guarantee_order else None
        actor_compute_log_prob_profiler = profiler_start(self.profiler_config, role="actor_compute_log_prob",
                                                    profiler_iteration=self.prof_iteration)
        start_time_defined = False
        while self.all_consumed(experience_consumer_stage, sorted_indexes) > 0:
            batch_data, index = self.dispatch_transfer_dock_data(experience_consumer_stage,
                                                                 experience_columns,
                                                                 experience_count,
                                                                 tp_size=self.megatron_config.tensor_model_parallel_size,
                                                                 indexes=sorted_indexes.pop(
                                                                     0) if self.rl_config.guarantee_order else None,
                                                                 get_n_samples=False)
            if not start_time_defined:
                start_time = time.time()
                start_time_defined = True
            if batch_data and index:
                output, batch = self.actor_hybrid.compute_log_prob(batch_data, self.rl_config.reuse_image_embeds)
                if self.parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # only on last rank. It should be on every tp rank
                    log_probs = torch.cat(output, dim=0)  # (bs, seq_size)
                    log_probs = log_probs.to(torch.float32)
                    log_probs = truncate_rows(log_probs, batch['response_length'])
                    output = {'old_log_prob': log_probs}
                    self.collect_transfer_dock_data(output, index)
                end_time = time.time()
                ray.get(
                        self.td.update_metrics.remote(
                            "timing/old_log_p",
                            value=[round(end_time, 4), round(start_time, 4)],
                            cumulate=True
                        )
                )
                ray.get(
                        self.td.update_metrics.remote(
                            "end_time/old_log_p",
                            value=[round(end_time, 4)],
                            cumulate=True
                        )
                )
        profiler_step(actor_compute_log_prob_profiler)

        # self.empty_cache()

    def _build_model_optimizer(self):
        actor_module, optimizer, opt_param_scheduler = self.setup_model_and_optimizer(
            self.model_provider, self.model_type.encoder_or_decoder)

        self.iteration = self.get_iteration()

        return actor_module, optimizer, opt_param_scheduler

    def _build_rollout(self):
        self.actor_model_config = AutoConfig.from_pretrained(
            self.megatron_config.tokenizer_name_or_path, trust_remote_code=self.generate_config.trust_remote_code)

        sampling_config = {"num_completions": self.rl_config.n_samples_per_prompt,
                        #    "best_of": self.rl_config.n_samples_per_prompt, # 对齐verl实现
                           **self.generate_config.sampling_config}

        rollout = VLLMInferEngine(
            tokenizer_name_or_path=self.megatron_config.tokenizer_name_or_path,
            train_tensor_parallel_size=self.megatron_config.tensor_model_parallel_size,
            train_pipeline_parallel_size=self.megatron_config.pipeline_model_parallel_size,
            train_expert_parallel_size=self.megatron_config.expert_model_parallel_size,
            infer_tensor_parallel_size=self.generate_config.infer_tensor_parallel_size,
            infer_pipeline_parallel_size=self.generate_config.infer_pipeline_parallel_size,
            infer_expert_parallel_size=self.generate_config.infer_expert_parallel_size,
            megatron_config=self.megatron_config,
            sampling_config=sampling_config,
            enable_prefix_caching=self.generate_config.enable_prefix_caching,
            num_scheduler_steps=self.generate_config.num_scheduler_steps,
            max_num_seqs=self.generate_config.max_num_seqs,
            max_model_len=self.generate_config.max_model_len,
            dtype=self.generate_config.dtype,
            gpu_memory_utilization=self.generate_config.gpu_memory_utilization,
            trust_remote_code=self.generate_config.trust_remote_code,
            limit_mm_image_per_prompt=self.generate_config.limit_mm_image_per_prompt,
            limit_mm_video_per_prompt=self.generate_config.limit_mm_video_per_prompt
        )

        return rollout

    def _build_sharding_manager(self):
        # perform weight resharding between actor and rollout
        sharding_manager = MegatronShardingManager(
            megatron_model=self.model,
            model_config=self.actor_model_config,
            infer_tensor_parallel_size=self.generate_config.infer_tensor_parallel_size,
            infer_pipeline_parallel_size=self.generate_config.infer_pipeline_parallel_size,
            infer_expert_parallel_size=self.generate_config.infer_expert_parallel_size,
            num_layer_list=self.megatron_config.num_layer_list,
            moe_tp_extend_ep=self.megatron_config.moe_tp_extend_ep,
            parallel_state=self.parallel_state,
            inference_engine=self.inference_model,
            optimizer=self.optimizer,
            optimizer_offload=self.generate_config.offload_train_optimizer,
            grad_offload=self.generate_config.offload_train_grad,
            train_param_offload=self.generate_config.offload_train_param,
            enable_validate=self.rl_config.enable_sharding_validate,
            megatron_offloader=self.actor_offloader
        )
        return sharding_manager

    def _set_no_sync_func(self):
        config = get_attr_wrapped_model(self.model[0], 'config', allow_none=False)

        config.grad_scale_func = self.optimizer.scale_loss

        if isinstance(self.model[0], self.distributed_data_parallel) and self.megatron_config.overlap_grad_reduce:
            if config.no_sync_func is not None:
                raise ValueError('When overlap_grad_reduce is True, config.no_sync_func must be None; '
                    'a custom no_sync_func is not supported when overlapping grad-reduce')
            config.no_sync_func = [model_chunk.no_sync for model_chunk in self.model]
            if len(self.model) == 1:
                config.no_sync_func = config.no_sync_func[0]

        config.finalize_model_grads_func = self.finalize_model_grads


@ray.remote(resources={"NPU": 0.7})
class ActorHybridWorker(ActorHybridWorkerBase):
    pass