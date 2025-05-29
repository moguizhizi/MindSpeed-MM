# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import dataclasses
from typing import Callable

import time
import ray
import torch

from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.config_cls.profiler_config import ProfilerConfig
from mindspeed_rl.utils.tokenizer import BaseTokenizer
from mindspeed_rl.workers.resharding.megatron_sharding_manager import MegatronOffLoader
from mindspeed_rl.utils.utils import mstx_timer_decorator, profiler_start, profiler_step

from mindspeed_rl.workers.actor_hybrid_worker import ActorHybridWorkerBase
from mindspeed_rl.workers.reference_woker import ReferenceWorkerBase
from mindspeed_rl.workers.reward_woker import RewardWorkerBase
from mindspeed_rl.models.reference import Reference
from mindspeed_rl.models.reward import Reward


@ray.remote(resources={"NPU": 0.7})
class IntegratedWorker(ActorHybridWorkerBase, ReferenceWorkerBase, RewardWorkerBase):
    """
    IntegratedWorker class. This class implements the integrated worker for the Actor, Reference and Reward Worker.

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

        # We use Actor as main worker, so only do init for Actor here.
        ActorHybridWorkerBase.__init__(
            self,
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

        self.update_micro_batch_size = rl_config.update_micro_batch_size

        self.reference = None
        self.ref_model = None
        self.ref_manager = None
        self.integrated_profiler = None


    def initialize(self):

        # Based on Actor
        ActorHybridWorkerBase.initialize(self)
        self.integrated_profiler = profiler_start(self.profiler_config, "integrated")

        # Add Reference
        self.ref_model = self.get_model(self.model_provider, self.model_type, wrap_with_ddp=False)
        ref_model_load_path = getattr(
            self.rl_config.integrated_mode_config, "ref_model_load_path", None
        ) if self.rl_config.integrated_mode_config is not None else None
        self.load_checkpoint_with_path(self.ref_model, ref_model_load_path, ckpt_only=True)
        self.ref_manager = MegatronOffLoader(self.ref_model, wrap_with_ddp=False)
        self.ref_manager.offload_param()
        self.reference = Reference(
            self.ref_model,
            beta=self.rl_config.beta,
            mini_batch_size=self.rl_config.mini_batch_size,
            epochs=self.rl_config.epochs,
            shuffle_mini_batch=self.rl_config.shuffle_mini_batch,
            generate_config=self.generate_config,
            stage=self.megatron_config.stage,
            forward_backward_func=self.forward_backward_func,
            micro_batch_size=self.megatron_config.ref_log_prob_micro_batch_size,
            temperature=self.generate_config.sampling_config["temperature"],
            use_dynamic_bsz=self.megatron_config.use_dynamic_bsz,
            max_log_prob_seq_len_forward=self.megatron_config.max_log_prob_seq_len_forward,
            max_log_prob_seq_len_update=self.megatron_config.max_log_prob_seq_len_update,
            forward_micro_batch_size=self.megatron_config.ref_log_prob_micro_batch_size
        )

    @mstx_timer_decorator
    def compute_ref_log_prob(self):
        start_onload_time = time.time()
        self.ref_manager.onload_param()
        end_onload_time = time.time()
        ray.get(
            self.td.update_metrics.remote(
                "timing/onload", 
                value=[round(end_onload_time, 4), round(start_onload_time, 4)],
                cumulate=True
            )
        ) 

        compute_log_prob_profiler = profiler_start(self.profiler_config, role="reference_compute_log_prob",
                                            profiler_iteration=self.prof_iteration)
        
        ReferenceWorkerBase.compute_ref_log_prob(self)
        
        profiler_step(compute_log_prob_profiler)

        start_offload_time = time.time()
        self.ref_manager.offload_param()
        end_offload_time = time.time()
        ray.get(
            self.td.update_metrics.remote(
                "timing/offload",
                value=[round(end_offload_time, 4), round(start_offload_time, 4)],
                cumulate=True
            )
        )

    def update(self, kl_ctrl=None, skip_actor_log_prob=False):
        # set update mbs
        update_mbs = self.update_micro_batch_size
        mbs = self.actor_hybrid.train_actor.micro_batch_size

        args = self.get_args()

        if update_mbs is not None:
            self.actor_hybrid.train_actor.micro_batch_size = update_mbs
            args.micro_batch_size = update_mbs

        ActorHybridWorkerBase.update(self, kl_ctrl, skip_actor_log_prob)
        
        profiler_step(self.integrated_profiler)
        args.micro_batch_size = mbs
        self.actor_hybrid.train_actor.micro_batch_size = mbs

    def load_checkpoint_with_path(self, model, path, ckpt_only=False):
        """Load model checkpoint from a specified path with flexible control.

        Args:
            model: The model to load checkpoint into.
            path: Path to the checkpoint file/directory. If None, use the path in megatron args.
            ckpt_only: If True, only loads model weights (skips optimizer/RNG states).
        """

        # Backup original arguments if needed
        original_args = {
            'no_load_optim': getattr(self.get_args(), "no_load_optim", None),
            'no_load_rng': getattr(self.get_args(), "no_load_rng", None),
            'load': getattr(self.get_args(), "load", None),
            'iteration': getattr(self.get_args(), "iteration", None),
            'finetune': getattr(self.get_args(), "finetune", None),
            'consumed_train_samples': getattr(self.get_args(), "consumed_train_samples", None),
            'consumed_valid_samples': getattr(self.get_args(), "consumed_valid_samples", None),
        } if ckpt_only or path else {}

        if ckpt_only:
            self._set_args({
                "no_load_optim": True,
                "no_load_rng": True,
                "finetune": True,
                'consumed_train_samples': 0,
                'consumed_valid_samples': 0
            })

        if path is not None:
            self._set_args({"load": path})

        self.load_checkpoint(model, None, None)

        if original_args:
            self._set_args(original_args)

    def _set_args(self, arg_dict):
        for key, value in arg_dict.items():
            if hasattr(self.get_args(), key):
                setattr(self.get_args(), key, value)

