import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from mindspeed_mm.models.ae.losses.discriminator import weights_init, NLayerDiscriminator3D
from mindspeed_mm.models.ae.losses.lpips import LPIPS


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


def hinge_d_loss_with_exemplar_weights(logits_real, logits_fake, weights):
    if weights.shape[0] != logits_real.shape[0] or weights.shape[0] != logits_fake.shape[0]:
        raise ValueError("The first dimension of weights, logits_real, and logits_fake must be the same.")

    loss_real = torch.mean(F.relu(1.0 - logits_real), dim=[1, 2, 3])
    loss_fake = torch.mean(F.relu(1.0 + logits_fake), dim=[1, 2, 3])
    loss_real = (weights * loss_real).sum() / weights.sum()
    loss_fake = (weights * loss_fake).sum() / weights.sum()
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def measure_perplexity(predicted_indices, n_embed):
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use


def l1(x, y):
    return torch.abs(x - y)


def l2(x, y):
    return torch.pow((x - y), 2)


class LPIPSWithDiscriminator3D(nn.Module):
    def __init__(
        self,
        perceptual_from_pretrained,
        discrim_start,
        logvar_init=0.0,
        kl_weight=1.0,
        pixelloss_weight=1.0,
        perceptual_weight=1.0,
        discrim_num_layers=4,
        discrim_in_channels=3,
        discrim_factor=1.0,
        discrim_weight=1.0,
        use_actnorm=False,
        discrim_conditional=False,
        discrim_loss="hinge",
        learn_logvar: bool = False,
        wavelet_weight=0.01,
        loss_type: str = "l1",
        use_dropout: bool = True,
        **kwargs
    ):

        super().__init__()
        if discrim_loss not in ["hinge", "vanilla"]:
            raise ValueError(f"discrim_loss must in ['hinge', 'vanilla'], but got {discrim_loss}!")
        self.wavelet_weight = wavelet_weight
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS(perceptual_from_pretrained, use_dropout).eval()
        self.perceptual_weight = perceptual_weight
        self.logvar = nn.Parameter(
            torch.full((), logvar_init), requires_grad=learn_logvar
        )
        self.discriminator = NLayerDiscriminator3D(
            input_nc=discrim_in_channels, n_layers=discrim_num_layers, use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator_iter_start = discrim_start
        self.discrim_loss = hinge_d_loss if discrim_loss == "hinge" else vanilla_d_loss
        self.discrim_factor = discrim_factor
        self.discriminator_weight = discrim_weight
        self.discrim_conditional = discrim_conditional
        self.loss_func = l1 if loss_type == "l1" else l2

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        layer = last_layer if last_layer is not None else self.last_layer[0]

        nll_grads = torch.autograd.grad(nll_loss, layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        optimizer_idx,
        global_step,
        split="train",
        weights=None,
        last_layer=None,
        cond=None,
    ):
        if optimizer_idx not in [0, 1]:
            raise ValueError(f"optimizer_idx must be equal to 0 or 1, but got {optimizer_idx}!")
        
        bs = inputs.shape[0]
        t = inputs.shape[2]
        if optimizer_idx == 0:
            inputs = rearrange(inputs, "b c t h w -> (b t) c h w").contiguous()
            reconstructions = rearrange(
                reconstructions, "b c t h w -> (b t) c h w"
            ).contiguous()
            rec_loss = self.loss_func(inputs, reconstructions)
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs, reconstructions)
                rec_loss = rec_loss + self.perceptual_weight * p_loss
            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            weighted_nll_loss = nll_loss
            if weights is not None:
                weighted_nll_loss = weights * nll_loss
            weighted_nll_loss = (
                torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            )
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
            kl_loss = posteriors.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            inputs = rearrange(inputs, "(b t) c h w -> b c t h w", t=t).contiguous()
            reconstructions = rearrange(
                reconstructions, "(b t) c h w -> b c t h w", t=t
            ).contiguous()

            logits_fake = self.discriminator(reconstructions)
            g_loss = -torch.mean(logits_fake)
            if global_step >= self.discriminator_iter_start:
                if self.discrim_factor > 0.0:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                else:
                    d_weight = torch.tensor(1.0)
            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0, requires_grad=True)

            discrim_factor = adopt_weight(
                self.discrim_factor, global_step, threshold=self.discriminator_iter_start
            )
            loss = (
                weighted_nll_loss
                + self.kl_weight * kl_loss
                + d_weight * discrim_factor * g_loss
            )
            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/logvar".format(split): self.logvar.detach(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): weighted_nll_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/discrim_factor".format(split): torch.tensor(discrim_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            }
            return loss, log
        else:
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            discrim_factor = adopt_weight(
                self.discrim_factor, global_step, threshold=self.discriminator_iter_start
            )

            d_loss = discrim_factor * self.discrim_loss(logits_real, logits_fake)

            log = {
                "{}/discrim_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log