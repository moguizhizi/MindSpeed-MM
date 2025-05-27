# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import os
import math
from functools import partial

import torch

from megatron.training import get_args
from megatron.training.global_vars import set_args
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.training.utils import print_rank_0

from mindspeed_mm.data.data_utils.constants import (
    VIDEO,
    PROMPT_IDS,
    PROMPT_MASK,
    VIDEO_MASK,
    VIDEO_REJECTED,
    SCORE,
    SCORE_REJECTED
)
from mindspeed_mm.tasks.rl.dpo.dpo_trainer import DPOTrainer
from mindspeed_mm.tasks.rl.dpo.stepvideo_dpo_model import StepVideoDPOModel
from mindspeed_mm.tasks.rl.utils import read_json_file, find_probability


class StepVideoDPOTrainer(DPOTrainer):
    """
    A trainer class for Direct Preference Optimization (DPO).

    This class provides methods for model initialize, computing losses and metrics, and training.
    """

    def __init__(
        self,
        train_valid_test_dataset_provider,
        model_type,
        process_non_loss_data_func=None,
        extra_args_provider=None,
        args_defaults=None,
    ):
        """
        Initializes the DPOTrainer instance.

        Sets up the instance variables for the model provider, actual micro batch size,
        and initializes the DPO model.
        """
        super().__init__(
            train_valid_test_dataset_provider,
            model_type,
            process_non_loss_data_func,
            extra_args_provider,
            args_defaults
        )

        args = get_args()
        self.histgram = read_json_file(args.mm.model.dpo.histgram_path)
        self.alpha = args.mm.model.dpo.weight_alpha
        self.beta = args.mm.model.dpo.weight_beta if args.mm.model.dpo.weight_beta else self.histgram['max_num'] / self.histgram['total_num']
        self.dpo_beta = args.mm.model.dpo.loss_beta
        self.args.actual_micro_batch_size = self.args.micro_batch_size * 4
        self.disable_dropout()

    def disable_dropout(self):
        """
        disable dropout
        """
        args_ = get_args()
        args_.attention_dropout = 0.0
        args_.hidden_dropout = 0.0
        args_.retro_encoder_hidden_dropout = 0.0
        args_.retro_encoder_attention_dropout = 0.0
        set_args(args_)

    @staticmethod
    def get_batch(data_iterator):
        """Generate a batch."""
        if data_iterator is not None:
            batch = next(data_iterator)
        else:
            raise ValueError("Data iterator is None. Unable to retrieve batch.")

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(torch.cuda.current_device())

        video = batch.pop(VIDEO, None)
        prompt_ids = batch.pop(PROMPT_IDS, None)
        video_mask = batch.pop(VIDEO_MASK, None)
        prompt_mask = batch.pop(PROMPT_MASK, None)
        video_lose = batch.pop(VIDEO_REJECTED, None)
        score = batch.pop(SCORE, 1.0)
        score_lose = batch.pop(SCORE_REJECTED, 1.0)

        args = get_args()

        video = video.to(args.params_dtype)
        video_lose = video_lose.to(args.params_dtype)

        return video, video_lose, prompt_ids, None, prompt_mask, score, score_lose

    def model_provider(self, **kwargs):
        args = get_args()
        print_rank_0("building StepVideoDPO model ...")
        self.hyper_model = StepVideoDPOModel(args.mm.model)
        return self.hyper_model

    def forward_step(self, data_iterator, model):
        """DPO Forward training step.

        Args:
            data_iterator : Input data iterator
            model : vlm model
        """
        # Get the batch.
        video, video_lose, prompt_ids, video_mask, prompt_mask, score, score_lose = self.get_batch(data_iterator)

        output_tensor, latents, noised_latents, noise, timesteps = model(video=video, video_lose=video_lose, prompt_ids=prompt_ids, video_mask=video_mask, prompt_mask=prompt_mask)

        return output_tensor, partial(self.loss_func, latents, noised_latents, noise, timesteps, score, score_lose)

    def loss_func(
        self,
        latents: torch.Tensor,
        noised_latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        score_win: torch.Tensor,
        score_lose: torch.Tensor,
        output_tensor: torch.Tensor
    ):
        args = get_args()
        actor_output, refer_output = torch.chunk(output_tensor, 2, dim=0)
        refer_output = refer_output.detach()

        loss, metrics = self.get_batch_loss_metrics(actor_output, refer_output,
                latents=latents, noised_latents=noised_latents, timesteps=timesteps, noise=noise, video_mask=None, score_win=score_win, score_lose=score_lose)

        if args.check_for_nan_in_loss_and_grad:
            global_rank = torch.distributed.get_rank()
            if loss.isnan():
                raise ValueError(f'Rank {global_rank}: found NaN in local forward loss calculation. '
                                 f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')

        # Reduce loss for logging.
        metrics['dpo loss'] = average_losses_across_data_parallel_group([loss])
        for key in metrics.keys():
            metrics[key] = average_losses_across_data_parallel_group([metrics[key]])

        return loss, metrics

    def get_batch_loss_metrics(
        self,
        actor_output,
        refer_output,
        **kwargs
    ):
        metrics = {}

        # compute L2
        actor_chosen_loss, actor_rejected_loss, actor_chosen_loss_avg = self._compute_log_probs(actor_output, **kwargs)

        refer_chosen_loss, refer_rejected_loss, *_ = self._compute_log_probs(refer_output, **kwargs)
        # compute DPO loss
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            actor_chosen_loss,
            actor_rejected_loss,
            refer_chosen_loss,
            refer_rejected_loss,
        )
        pair_prob = math.sqrt(find_probability(kwargs['score_win'], self.histgram) * find_probability(kwargs['score_lose'], self.histgram))
        weight_pair = math.pow((self.beta / max(pair_prob, 1e-3)), self.alpha)
        losses = losses * weight_pair

        sft_loss = -actor_chosen_loss_avg
        if self.args.pref_ftx > 1e-6:
            losses += self.args.pref_ftx * sft_loss

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        prefix = ""
        metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.detach().mean()

        return losses.mean(), metrics

    def _compute_log_probs(self, output, **kwargs):
        """
        Computes the sum log probabilities of the labels under given logits if loss_type.
        Otherwise, the average log probabilities.
        Assuming IGNORE_INDEX is all negative numbers, the default is -100.

        Args:
            all_logits: The logits tensor.

        Returns:
            A tuple containing the log probabilities and other tensors.
        """
        # # SNR is determined by snr_gamma in config and has been multiplied in training_losses.
        latents, noised_latents, timesteps, noise, video_mask = kwargs['latents'], kwargs['noised_latents'], kwargs['timesteps'], kwargs['noise'], kwargs['video_mask']
        
        l2_loss = self.hyper_model.diffusion.training_losses(model_output=output, x_start=latents, x_t=noised_latents, t=timesteps, noise=noise, mask=None)

        chosen_l2_losses, rejected_l2_losses = torch.chunk(- self.dpo_beta * timesteps * l2_loss, 2, dim=0)

        all_results = (chosen_l2_losses, rejected_l2_losses, chosen_l2_losses)

        return all_results