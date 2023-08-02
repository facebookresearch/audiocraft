# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import flashy
import torch
from torch import autograd


class Balancer:
    """Loss balancer.

    The loss balancer combines losses together to compute gradients for the backward.
    Given `y = f(...)`, and a number of losses `l1(y, ...)`, `l2(y, ...)`, with `...`
    not having any dependence on `f`, the balancer can efficiently normalize the partial gradients
    `d l1 / d y`, `d l2 / dy` before summing them in order to achieve a desired ratio between
    the losses. For instance if `weights = {'l1': 2, 'l2': 1}`, 66% of the gradient
    going into `f(...)` will come from `l1` on average, and 33% from `l2`. This allows for an easy
    interpration of the weights even if the intrisic scale of `l1`, `l2` ... is unknown.

    Noting `g1 = d l1 / dy`, etc., the balanced gradient `G` will be
    (with `avg` an exponential moving average over the updates),

        G = sum_i total_norm * g_i / avg(||g_i||) * w_i / sum(w_i)

    If `balance_grads` is False, this is deactivated, and instead the gradient will just be the
    standard sum of the partial gradients with the given weights.

    A call to the backward method of the balancer will compute the the partial gradients,
    combining all the losses and potentially rescaling the gradients,
    which can help stabilize the training and reason about multiple losses with varying scales.
    The obtained gradient with respect to `y` is then back-propagated to `f(...)`.

    Expected usage:

        weights = {'loss_a': 1, 'loss_b': 4}
        balancer = Balancer(weights, ...)
        losses: dict = {}
        losses['loss_a'] = compute_loss_a(x, y)
        losses['loss_b'] = compute_loss_b(x, y)
        if model.training():
            effective_loss = balancer.backward(losses, x)

    Args:
        weights (dict[str, float]): Weight coefficient for each loss. The balancer expect the losses keys
            from the backward method to match the weights keys to assign weight to each of the provided loss.
        balance_grads (bool): Whether to rescale gradients so that weights reflect the fraction of the
            overall gradient, rather than a constant multiplier.
        total_norm (float): Reference norm when rescaling gradients, ignored otherwise.
        emay_decay (float): EMA decay for averaging the norms.
        per_batch_item (bool): Whether to compute the averaged norm per batch item or not. This only holds
            when rescaling the gradients.
        epsilon (float): Epsilon value for numerical stability.
        monitor (bool): If True, stores in `self.metrics` the relative ratio between the norm of the gradients
            coming from each loss, when calling `backward()`.
    """
    def __init__(self, weights: tp.Dict[str, float], balance_grads: bool = True, total_norm: float = 1.,
                 ema_decay: float = 0.999, per_batch_item: bool = True, epsilon: float = 1e-12,
                 monitor: bool = False):
        self.weights = weights
        self.per_batch_item = per_batch_item
        self.total_norm = total_norm or 1.
        self.averager = flashy.averager(ema_decay or 1.)
        self.epsilon = epsilon
        self.monitor = monitor
        self.balance_grads = balance_grads
        self._metrics: tp.Dict[str, tp.Any] = {}

    @property
    def metrics(self):
        return self._metrics

    def backward(self, losses: tp.Dict[str, torch.Tensor], input: torch.Tensor) -> torch.Tensor:
        """Compute the backward and return the effective train loss, e.g. the loss obtained from
        computing the effective weights. If `balance_grads` is True, the effective weights
        are the one that needs to be applied to each gradient to respect the desired relative
        scale of gradients coming from each loss.

        Args:
            losses (Dict[str, torch.Tensor]): dictionary with the same keys as `self.weights`.
            input (torch.Tensor): the input of the losses, typically the output of the model.
                This should be the single point of dependence between the losses
                and the model being trained.
        """
        norms = {}
        grads = {}
        for name, loss in losses.items():
            # Compute partial derivative of the less with respect to the input.
            grad, = autograd.grad(loss, [input], retain_graph=True)
            if self.per_batch_item:
                # We do not average the gradient over the batch dimension.
                dims = tuple(range(1, grad.dim()))
                norm = grad.norm(dim=dims, p=2).mean()
            else:
                norm = grad.norm(p=2)
            norms[name] = norm
            grads[name] = grad

        count = 1
        if self.per_batch_item:
            count = len(grad)
        # Average norms across workers. Theoretically we should average the
        # squared norm, then take the sqrt, but it worked fine like that.
        avg_norms = flashy.distrib.average_metrics(self.averager(norms), count)
        # We approximate the total norm of the gradient as the sums of the norms.
        # Obviously this can be very incorrect if all gradients are aligned, but it works fine.
        total = sum(avg_norms.values())

        self._metrics = {}
        if self.monitor:
            # Store the ratio of the total gradient represented by each loss.
            for k, v in avg_norms.items():
                self._metrics[f'ratio_{k}'] = v / total

        total_weights = sum([self.weights[k] for k in avg_norms])
        assert total_weights > 0.
        desired_ratios = {k: w / total_weights for k, w in self.weights.items()}

        out_grad = torch.zeros_like(input)
        effective_loss = torch.tensor(0., device=input.device, dtype=input.dtype)
        for name, avg_norm in avg_norms.items():
            if self.balance_grads:
                # g_balanced = g / avg(||g||) * total_norm * desired_ratio
                scale = desired_ratios[name] * self.total_norm / (self.epsilon + avg_norm)
            else:
                # We just do regular weighted sum of the gradients.
                scale = self.weights[name]
            out_grad.add_(grads[name], alpha=scale)
            effective_loss += scale * losses[name].detach()
        # Send the computed partial derivative with respect to the output of the model to the model.
        input.backward(out_grad)
        return effective_loss
