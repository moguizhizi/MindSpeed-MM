# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.

import os
from contextlib import contextmanager

import gc
import ray
import torch
import torch.distributed
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig
from vllm import LLM, SamplingParams
from mindspeed_rl.utils.utils import mstx_timer_decorator

def dummy_compile(*compile_args, **compile_kwargs):
    def decorate(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    return decorate


torch.jit.script = dummy_compile
torch.compile = dummy_compile
# from vllm import LLM, SamplingParams
import mindspeed_rl.vllm_patch

from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.models.base.base_inference_engine import BaseInferEngine
from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.models.rollout.vllm_adapter.vllm_parallel_state import initialize_parallel_state
from mindspeed_rl.models.rollout.vllm_adapter.megatron_weight_loaders import (
    load_megatron_weights,
    update_megatron_weight_loader,
    InferParallelConfig
)
from mindspeed_rl.utils import get_tokenizer

logger = Loggers("vllm_engine")


class VLLMInferEngine(BaseInferEngine):
    def __init__(
            self,
            tokenizer_name_or_path: str,
            train_tensor_parallel_size: int,
            train_pipeline_parallel_size: int,
            train_expert_parallel_size: int,
            infer_tensor_parallel_size: int,
            infer_pipeline_parallel_size: int,
            infer_expert_parallel_size: int,
            megatron_config: MegatronConfig,
            sampling_config: dict,
            prompt_type: str = None,
            prompt_type_path: str = None,
            enable_prefix_caching: bool = False,
            num_scheduler_steps: int = 1,
            max_num_seqs: int = 1,
            max_model_len: int = 2048,
            dtype: str = "bfloat16",
            gpu_memory_utilization: float = 0.5,
            trust_remote_code: bool = True,
            load_format: str = "megatron",
            limit_mm_image_per_prompt: int = 1,
            limit_mm_video_per_prompt: int = 0,
            **kwargs
    ):
        """
        Initialize the VLLM inference engine.

        Args:
            tokenizer_name_or_path (str): Path or name of the tokenizer.
            train_tensor_parallel_size (int): Tensor parallel size during training.
            train_pipeline_parallel_size (int): Pipeline parallel size during training.
            train_expert_parallel_size (int): Expert parallel size during training.
            infer_tensor_parallel_size (int): Tensor parallel size during inference.
            infer_pipeline_parallel_size (int): Pipeline parallel size during inference.
            infer_expert_parallel_size (int): Expert parallel size during inference.
            sampling_config (dict): Configuration for text generation sampling.
            enable_prefix_caching (bool): Whether to enable prefix caching.
            num_scheduler_steps (int): Num scheduler steps. Default is 1.
            max_num_seqs (int): Maximum number of sequences to process simultaneously. Default is 1.
            max_model_len (int): Maximum model length (in tokens). Default is 2048.
            dtype (str): Data type for model weights. Default is "bfloat16".
            gpu_memory_utilization (float): GPU memory utilization factor. Default is 0.5.
            trust_remote_code (bool): Whether to trust remote code (e.g., for custom tokenizers).
            **kwargs: Additional keyword arguments.
        """
        # Call the parent class's __init__ method
        super().__init__(
            tokenizer_name_or_path=tokenizer_name_or_path,
            prompt_type=prompt_type,
            prompt_type_path=prompt_type_path,
            train_tensor_parallel_size=train_tensor_parallel_size,
            train_pipeline_parallel_size=train_pipeline_parallel_size,
            train_expert_parallel_size=train_expert_parallel_size,
            infer_tensor_parallel_size=infer_tensor_parallel_size,
            infer_pipeline_parallel_size=infer_pipeline_parallel_size,
            infer_expert_parallel_size=infer_expert_parallel_size,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code
        )
        # Additional initialization logic for VLLMInferEngine
        # Initialize sampling parameters from SamplingConfig
        self.sampling_config = sampling_config
        try:
            self.sampling_params = SamplingParams(
                n=sampling_config.get('num_completions', 1),
                logprobs=sampling_config.get('logprobs', 1),
                max_tokens=sampling_config.get('max_tokens', 128),
                # best_of=sampling_config.get('best_of', 2),
                top_p=sampling_config.get('top_p', 1.0),
                top_k=sampling_config.get('top_k', 50),
                min_p=sampling_config.get('min_p', 0.0),
                temperature=sampling_config.get('temperature', 0.2),
                detokenize=sampling_config.get('detokenize', False),
                seed=sampling_config.get('seed', None)
            )
        except Exception as e:
            raise ValueError(f"Error creating SamplingParams from dictionary") from e

        self.hf_config = AutoConfig.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=trust_remote_code
        )

        self.tokenizer = get_tokenizer(tokenizer_name_or_path,
                                       prompt_type=prompt_type, prompt_type_path=prompt_type_path)
        self.pad_token_id = (
            self.tokenizer.tokenizer.pad_token_id if self.tokenizer.tokenizer.pad_token_id is not None
            else self.tokenizer.tokenizer.eos_token_id)

        # Set up local rank using the helper function
        self.local_rank = get_local_rank()

        # Initialize parallel state if tensor parallel size is specified
        if train_tensor_parallel_size is not None:
            num_tp_per_train_tp = train_tensor_parallel_size // infer_tensor_parallel_size
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            initialize_parallel_state(
                infer_tensor_model_parallel_size=infer_tensor_parallel_size,
                train_tensor_model_parallel_size=train_tensor_parallel_size,
                infer_pipeline_model_parallel_size=infer_pipeline_parallel_size,
                train_pipeline_model_parallel_size=train_pipeline_parallel_size,
            )

        if load_format == "megatron":
            update_megatron_weight_loader()

        torch.jit.script = dummy_compile
        torch.compile = dummy_compile

        limit_mm_per_prompt_dict = {}
        if limit_mm_image_per_prompt > 0:
            limit_mm_per_prompt_dict['image'] = limit_mm_image_per_prompt
        if limit_mm_video_per_prompt > 0:
            limit_mm_per_prompt_dict['video'] = limit_mm_video_per_prompt

        # Initialize the LLM engine
        self.llm = LLM(
            model=tokenizer_name_or_path,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=infer_tensor_parallel_size,
            load_format="dummy" if load_format == "megatron" else "auto",
            distributed_executor_backend="external_launcher",
            enable_prefix_caching=enable_prefix_caching,
            num_scheduler_steps=num_scheduler_steps,
            dtype=dtype,
            enforce_eager=False,
            skip_tokenizer_init=False,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            seed=self.sampling_config.get('seed', None),
            limit_mm_per_prompt=limit_mm_per_prompt_dict
        )

        self.model = self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()

        self.cpu_model = {}
        for name, params in self.model.named_parameters():
            self.cpu_model[name] = torch.empty_like(params, device="cpu")

        if load_format == "megatron":
            self.free_cache_engine()
            self.offload_model_weights()

    def init_cache_engine(self):
        if self.llm.llm_engine.model_executor.driver_worker.worker.cache_engine is None:
            self.llm.llm_engine.model_executor.driver_worker.worker._init_cache_engine()

    def free_cache_engine(self):
        ctx = self.llm.llm_engine.model_executor.driver_worker.worker.compilation_config.static_forward_context
        from vllm.attention import AttentionType

        layer_need_kv_cache = []
        for layer_name in ctx:
            if ctx[layer_name].attn_type in (AttentionType.DECODER, AttentionType.ENCODER_DECODER):
                layer_need_kv_cache.append(layer_name)

        pipeline_parallel_size = self.llm.llm_engine.vllm_config.parallel_config.pipeline_parallel_size
        for layer_name in layer_need_kv_cache:
            kv_cache = []
            for _ in range(pipeline_parallel_size):
                kv_cache.append(torch.tensor([]))
            ctx[layer_name].kv_cache = kv_cache

        self.llm.llm_engine.model_executor.driver_worker.worker.cache_engine = None
        self.llm.llm_engine.model_executor.driver_worker.worker.gpu_cache = None

        if hasattr(self.model,'model') and hasattr(self.model.model.layers[0].self_attn, "attn"):
            for i in range(self.model.model.start_layer, self.model.model.end_layer):
                attn_impl = self.model.model.layers[i].self_attn.attn.impl
                if hasattr(attn_impl, "key_cache"):
                    attn_impl.key_cache = None
                    attn_impl.value_cache = None
        elif hasattr(self.model,'language_model') and hasattr(self.model.language_model.model.layers[0].self_attn, "attn"):
            for i in range(self.model.language_model.model.start_layer, self.model.language_model.model.end_layer):
                attn_impl = self.model.language_model.model.layers[i].self_attn.attn.impl
                if hasattr(attn_impl, "key_cache"):
                    attn_impl.key_cache = None
                    attn_impl.value_cache = None

        self.gpu_cache = None

        gc.collect()
        torch.cuda.empty_cache()

    def offload_model_weights(self):
        for name, params in self.model.named_parameters():
            params.data = self.cpu_model[name]

    def sync_model_weights(self, params, load_format='megatron'):
        infer_parallel_config = InferParallelConfig(self.infer_tensor_parallel_size, self.infer_pipeline_parallel_size,
                                                    self.infer_expert_parallel_size)
        load_megatron_weights(params,
                              self.model,
                              infer_parallel_config,
                              self.hf_config)
        if hasattr(self.model,'model') and hasattr(self.model.model.layers[0].self_attn, "mla_attn"):
            self._process_mla()

    def _process_mla(self):
        for i in range(self.model.model.start_layer, self.model.model.end_layer):
            mla = self.model.model.layers[i].self_attn.mla_attn.impl
            if hasattr(mla, "w_kc"):
                mla.w_kc = None
                mla.w_vc = None

    @torch.no_grad()
    @mstx_timer_decorator
    def generate_sequences(self, prompts, images,**kwargs):
        self.init_cache_engine()
        if "vit_embeds" in images:
            prompts = [{"prompt_token_ids": prompt,
                        "multi_modal_data": {"image_embeds": image_embeds, "image_grid_thw": grid_thw}}
                       for prompt, image_embeds, grid_thw in
                       zip(prompts, images['vit_embeds'], images['image_grid_thw'])]
            if torch.sum(images['video_num']).item() > 0:
                from megatron.training import get_args
                # replace video_token_id with image_token_id when using vit_embeds
                for p in prompts:
                    p['prompt_token_ids'] = list(map(lambda x: get_args().mm.model.image_token_id if x == get_args().mm.model.video_token_id else x, p['prompt_token_ids']))
        else:
            if torch.sum(images['image_num']).item() > 0:
                prompts = [{"prompt_token_ids": prompt, "multi_modal_data": {"image": image}} for prompt, image in
                           zip(prompts, images['image'])]
            else:
                prompts = [{"prompt_token_ids": prompt, "multi_modal_data": {"video": video},
                            'mm_processor_kwargs': {'fps': fps.squeeze().tolist()}} for prompt, video, fps in
                           zip(prompts, images['video'], images['video_fps'])]

        with self.update_sampling_params(**kwargs):
            response = self.llm.generate(
                prompts=prompts,
                sampling_params=self.sampling_params,
                use_tqdm=False
            )
            outs = self._post_process_outputs(response)
        self.free_cache_engine()
        return outs

    def _post_process_outputs(self, request_outputs):
        output_token_ids = []
        logprobs = []
        for request_output in request_outputs:  # List[RequestOutput]
            outputs = request_output.outputs
            for output in outputs:  # List[CompletionOutput], usually len == 1
                output_token_ids.append(torch.tensor(output.token_ids))
                logprobs_dicts = output.logprobs
                if logprobs_dicts is None:
                    continue

                logprob = []
                for logprobs_dict, token_id in zip(logprobs_dicts, output.token_ids):
                    logprob.append(logprobs_dict[token_id].logprob)
                logprobs.append(torch.tensor(logprob))

        output_token_ids = pad_sequence(output_token_ids, batch_first=True,
                                        padding_value=self.pad_token_id)
        if len(logprobs) > 0:
            logprobs = pad_sequence(logprobs, batch_first=True,
                                    padding_value=self.pad_token_id)
        return output_token_ids, logprobs

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def chat(self, conversation, sampling_params=None):
        outputs = self.llm.chat(
            conversation,
            sampling_params=sampling_params if sampling_params else self.sampling_params,
            use_tqdm=False)
        return outputs


def get_local_rank() -> int:
    """
    Determine the local rank based on the runtime context.
    - If launched via `torchrun`, the `LOCAL_RANK` environment variable is used.
    - If launched via `ray`, the rank is obtained from the ray runtime context.
    - If neither is available, defaults to 0 (for testing or single-process scenarios).

    Returns:
        int: The local rank of the current process.
    """
    # Check if launched via torchrun (LOCAL_RANK is set)
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])

    # Check if launched via ray
    try:
        # Get the local rank from ray's runtime context
        local_rank_str = ray.get_runtime_context().get_accelerator_ids()["NPU"][0]
        os.environ["LOCAL_RANK"] = local_rank_str
        return int(local_rank_str)

    except Exception as e:
        logger.warning("Warning: Failed to get local rank from ray runtime context. Error: {}".format(e))

    # Default to 0 (for testing or single-process scenarios)
    logger.warning("Warning: Unable to determine local rank. Defaulting to 0.")
    return 0
