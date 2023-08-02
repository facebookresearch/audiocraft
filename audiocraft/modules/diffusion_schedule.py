# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for Noise Schedule, defines diffusion process, reverse process and data processor.
"""

from collections import namedtuple
import random
import typing as tp
import julius
import torch

TrainingItem = namedtuple("TrainingItem", "noisy noise step")


def betas_from_alpha_bar(alpha_bar):
    alphas = torch.cat([torch.Tensor([alpha_bar[0]]), alpha_bar[1:]/alpha_bar[:-1]])
    return 1 - alphas


class SampleProcessor(torch.nn.Module):
    def project_sample(self, x: torch.Tensor):
        """Project the original sample to the 'space' where the diffusion will happen."""
        return x

    def return_sample(self, z: torch.Tensor):
        """Project back from diffusion space to the actual sample space."""
        return z


class MultiBandProcessor(SampleProcessor):
    """
    MultiBand sample processor. The input audio is splitted across
    frequency bands evenly distributed in mel-scale.

    Each band will be rescaled to match the power distribution
    of Gaussian noise in that band, using online metrics
    computed on the first few samples.

    Args:
        n_bands (int): Number of mel-bands to split the signal over.
        sample_rate (int): Sample rate of the audio.
        num_samples (int): Number of samples to use to fit the rescaling
            for each band. The processor won't be stable
            until it has seen that many samples.
        power_std (float or list/tensor): The rescaling factor computed to match the
            power of Gaussian noise in each band is taken to
            that power, i.e. `1.` means full correction of the energy
            in each band, and values less than `1` means only partial
            correction. Can be used to balance the relative importance
            of low vs. high freq in typical audio signals.
    """
    def __init__(self, n_bands: int = 8, sample_rate: float = 24_000,
                 num_samples: int = 10_000, power_std: tp.Union[float, tp.List[float], torch.Tensor] = 1.):
        super().__init__()
        self.n_bands = n_bands
        self.split_bands = julius.SplitBands(sample_rate, n_bands=n_bands)
        self.num_samples = num_samples
        self.power_std = power_std
        if isinstance(power_std, list):
            assert len(power_std) == n_bands
            power_std = torch.tensor(power_std)
        self.register_buffer('counts', torch.zeros(1))
        self.register_buffer('sum_x', torch.zeros(n_bands))
        self.register_buffer('sum_x2', torch.zeros(n_bands))
        self.register_buffer('sum_target_x2', torch.zeros(n_bands))
        self.counts: torch.Tensor
        self.sum_x: torch.Tensor
        self.sum_x2: torch.Tensor
        self.sum_target_x2: torch.Tensor

    @property
    def mean(self):
        mean = self.sum_x / self.counts
        return mean

    @property
    def std(self):
        std = (self.sum_x2 / self.counts - self.mean**2).clamp(min=0).sqrt()
        return std

    @property
    def target_std(self):
        target_std = self.sum_target_x2 / self.counts
        return target_std

    def project_sample(self, x: torch.Tensor):
        assert x.dim() == 3
        bands = self.split_bands(x)
        if self.counts.item() < self.num_samples:
            ref_bands = self.split_bands(torch.randn_like(x))
            self.counts += len(x)
            self.sum_x += bands.mean(dim=(2, 3)).sum(dim=1)
            self.sum_x2 += bands.pow(2).mean(dim=(2, 3)).sum(dim=1)
            self.sum_target_x2 += ref_bands.pow(2).mean(dim=(2, 3)).sum(dim=1)
        rescale = (self.target_std / self.std.clamp(min=1e-12)) ** self.power_std  # same output size
        bands = (bands - self.mean.view(-1, 1, 1, 1)) * rescale.view(-1, 1, 1, 1)
        return bands.sum(dim=0)

    def return_sample(self, x: torch.Tensor):
        assert x.dim() == 3
        bands = self.split_bands(x)
        rescale = (self.std / self.target_std) ** self.power_std
        bands = bands * rescale.view(-1, 1, 1, 1) + self.mean.view(-1, 1, 1, 1)
        return bands.sum(dim=0)


class NoiseSchedule:
    """Noise schedule for diffusion.

    Args:
        beta_t0 (float): Variance of the first diffusion step.
        beta_t1 (float): Variance of the last diffusion step.
        beta_exp (float): Power schedule exponent
        num_steps (int): Number of diffusion step.
        variance (str): choice of the sigma value for the denoising eq. Choices: "beta" or "beta_tilde"
        clip (float): clipping value for the denoising steps
        rescale (float): rescaling value to avoid vanishing signals unused by default (i.e 1)
        repartition (str): shape of the schedule only power schedule is supported
        sample_processor (SampleProcessor): Module that normalize data to match better the gaussian distribution
        noise_scale (float): Scaling factor for the noise
    """
    def __init__(self, beta_t0: float = 1e-4, beta_t1: float = 0.02, num_steps: int = 1000, variance: str = 'beta',
                 clip: float = 5., rescale: float = 1., device='cuda', beta_exp: float = 1,
                 repartition: str = "power", alpha_sigmoid: dict = {}, n_bands: tp.Optional[int] = None,
                 sample_processor: SampleProcessor = SampleProcessor(), noise_scale: float = 1.0, **kwargs):

        self.beta_t0 = beta_t0
        self.beta_t1 = beta_t1
        self.variance = variance
        self.num_steps = num_steps
        self.clip = clip
        self.sample_processor = sample_processor
        self.rescale = rescale
        self.n_bands = n_bands
        self.noise_scale = noise_scale
        assert n_bands is None
        if repartition == "power":
            self.betas = torch.linspace(beta_t0 ** (1 / beta_exp), beta_t1 ** (1 / beta_exp), num_steps,
                                        device=device, dtype=torch.float) ** beta_exp
        else:
            raise RuntimeError('Not implemented')
        self.rng = random.Random(1234)

    def get_beta(self, step: tp.Union[int, torch.Tensor]):
        if self.n_bands is None:
            return self.betas[step]
        else:
            return self.betas[:, step]  # [n_bands, len(step)]

    def get_initial_noise(self, x: torch.Tensor):
        if self.n_bands is None:
            return torch.randn_like(x)
        return torch.randn((x.size(0), self.n_bands, x.size(2)))

    def get_alpha_bar(self, step: tp.Optional[tp.Union[int, torch.Tensor]] = None) -> torch.Tensor:
        """Return 'alpha_bar', either for a given step, or as a tensor with its value for each step."""
        if step is None:
            return (1 - self.betas).cumprod(dim=-1)  # works for simgle and multi bands
        if type(step) is int:
            return (1 - self.betas[:step + 1]).prod()
        else:
            return (1 - self.betas).cumprod(dim=0)[step].view(-1, 1, 1)

    def get_training_item(self, x: torch.Tensor, tensor_step: bool = False) -> TrainingItem:
        """Create a noisy data item for diffusion model training:

        Args:
            x (torch.Tensor): clean audio data torch.tensor(bs, 1, T)
            tensor_step (bool): If tensor_step = false, only one step t is sample,
                the whole batch is diffused to the same step and t is int.
                If tensor_step = true, t is a tensor of size (x.size(0),)
                every element of the batch is diffused to a independently sampled.
        """
        step: tp.Union[int, torch.Tensor]
        if tensor_step:
            bs = x.size(0)
            step = torch.randint(0, self.num_steps, size=(bs,), device=x.device)
        else:
            step = self.rng.randrange(self.num_steps)
        alpha_bar = self.get_alpha_bar(step)  # [batch_size, n_bands, 1]

        x = self.sample_processor.project_sample(x)
        noise = torch.randn_like(x)
        noisy = (alpha_bar.sqrt() / self.rescale) * x + (1 - alpha_bar).sqrt() * noise * self.noise_scale
        return TrainingItem(noisy, noise, step)

    def generate(self, model: torch.nn.Module, initial: tp.Optional[torch.Tensor] = None,
                 condition: tp.Optional[torch.Tensor] = None, return_list: bool = False):
        """Full ddpm reverse process.

        Args:
            model (nn.Module): Diffusion model.
            initial (tensor): Initial Noise.
            condition (tensor): Input conditionning Tensor (e.g. encodec compressed representation).
            return_list (bool): Whether to return the whole process or only the sampled point.
        """
        alpha_bar = self.get_alpha_bar(step=self.num_steps - 1)
        current = initial
        iterates = [initial]
        for step in range(self.num_steps)[::-1]:
            with torch.no_grad():
                estimate = model(current, step, condition=condition).sample
            alpha = 1 - self.betas[step]
            previous = (current - (1 - alpha) / (1 - alpha_bar).sqrt() * estimate) / alpha.sqrt()
            previous_alpha_bar = self.get_alpha_bar(step=step - 1)
            if step == 0:
                sigma2 = 0
            elif self.variance == 'beta':
                sigma2 = 1 - alpha
            elif self.variance == 'beta_tilde':
                sigma2 = (1 - previous_alpha_bar) / (1 - alpha_bar) * (1 - alpha)
            elif self.variance == 'none':
                sigma2 = 0
            else:
                raise ValueError(f'Invalid variance type {self.variance}')

            if sigma2 > 0:
                previous += sigma2**0.5 * torch.randn_like(previous) * self.noise_scale
            if self.clip:
                previous = previous.clamp(-self.clip, self.clip)
            current = previous
            alpha_bar = previous_alpha_bar
            if step == 0:
                previous *= self.rescale
            if return_list:
                iterates.append(previous.cpu())

        if return_list:
            return iterates
        else:
            return self.sample_processor.return_sample(previous)

    def generate_subsampled(self, model: torch.nn.Module, initial: torch.Tensor, step_list: tp.Optional[list] = None,
                            condition: tp.Optional[torch.Tensor] = None, return_list: bool = False):
        """Reverse process that only goes through Markov chain states in step_list."""
        if step_list is None:
            step_list = list(range(1000))[::-50] + [0]
        alpha_bar = self.get_alpha_bar(step=self.num_steps - 1)
        alpha_bars_subsampled = (1 - self.betas).cumprod(dim=0)[list(reversed(step_list))].cpu()
        betas_subsampled = betas_from_alpha_bar(alpha_bars_subsampled)
        current = initial * self.noise_scale
        iterates = [current]
        for idx, step in enumerate(step_list[:-1]):
            with torch.no_grad():
                estimate = model(current, step, condition=condition).sample * self.noise_scale
            alpha = 1 - betas_subsampled[-1 - idx]
            previous = (current - (1 - alpha) / (1 - alpha_bar).sqrt() * estimate) / alpha.sqrt()
            previous_alpha_bar = self.get_alpha_bar(step_list[idx + 1])
            if step == step_list[-2]:
                sigma2 = 0
                previous_alpha_bar = torch.tensor(1.0)
            else:
                sigma2 = (1 - previous_alpha_bar) / (1 - alpha_bar) * (1 - alpha)
            if sigma2 > 0:
                previous += sigma2**0.5 * torch.randn_like(previous) * self.noise_scale
            if self.clip:
                previous = previous.clamp(-self.clip, self.clip)
            current = previous
            alpha_bar = previous_alpha_bar
            if step == 0:
                previous *= self.rescale
            if return_list:
                iterates.append(previous.cpu())
        if return_list:
            return iterates
        else:
            return self.sample_processor.return_sample(previous)
