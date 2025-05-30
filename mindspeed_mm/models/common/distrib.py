import torch
import numpy as np


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False, dim=1):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean, device=self.parameters.device, dtype=self.parameters.dtype)

    def sample(self, generator=None):
        x = self.mean + self.std * torch.randn(self.mean.shape, device=self.parameters.device,
                                               dtype=self.parameters.dtype, generator=generator)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                       + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                       torch.pow(self.mean - other.mean, 2) / other.var
                       + self.var / other.var - 1.0 - self.logvar + other.logvar, dim=[1, 2, 3])

    def nll(self, sample, dims=(1, 2, 3)):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar
               + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean
