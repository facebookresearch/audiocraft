# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import random

import torch

from audiocraft.adversarial import (
    AdversarialLoss,
    get_adv_criterion,
    get_real_criterion,
    get_fake_criterion,
    FeatureMatchingLoss,
    MultiScaleDiscriminator,
)


class TestAdversarialLoss:

    def test_adversarial_single_multidiscriminator(self):
        adv = MultiScaleDiscriminator()
        optimizer = torch.optim.Adam(
            adv.parameters(),
            lr=1e-4,
        )
        loss, loss_real, loss_fake = get_adv_criterion('mse'), get_real_criterion('mse'), get_fake_criterion('mse')
        adv_loss = AdversarialLoss(adv, optimizer, loss, loss_real, loss_fake)

        B, C, T = 4, 1, random.randint(1000, 5000)
        real = torch.randn(B, C, T)
        fake = torch.randn(B, C, T)

        disc_loss = adv_loss.train_adv(fake, real)
        assert isinstance(disc_loss, torch.Tensor) and isinstance(disc_loss.item(), float)

        loss, loss_feat = adv_loss(fake, real)
        assert isinstance(loss, torch.Tensor) and isinstance(loss.item(), float)
        # we did not specify feature loss
        assert loss_feat.item() == 0.

    def test_adversarial_feat_loss(self):
        adv = MultiScaleDiscriminator()
        optimizer = torch.optim.Adam(
            adv.parameters(),
            lr=1e-4,
        )
        loss, loss_real, loss_fake = get_adv_criterion('mse'), get_real_criterion('mse'), get_fake_criterion('mse')
        feat_loss = FeatureMatchingLoss()
        adv_loss = AdversarialLoss(adv, optimizer, loss, loss_real, loss_fake, feat_loss)

        B, C, T = 4, 1, random.randint(1000, 5000)
        real = torch.randn(B, C, T)
        fake = torch.randn(B, C, T)

        loss, loss_feat = adv_loss(fake, real)

        assert isinstance(loss, torch.Tensor) and isinstance(loss.item(), float)
        assert isinstance(loss_feat, torch.Tensor) and isinstance(loss.item(), float)


class TestGeneratorAdversarialLoss:

    def test_hinge_generator_adv_loss(self):
        adv_loss = get_adv_criterion(loss_type='hinge')

        t0 = torch.randn(1, 2, 0)
        t1 = torch.FloatTensor([1.0, 2.0, 3.0])

        assert adv_loss(t0).item() == 0.0
        assert adv_loss(t1).item() == -2.0

    def test_mse_generator_adv_loss(self):
        adv_loss = get_adv_criterion(loss_type='mse')

        t0 = torch.randn(1, 2, 0)
        t1 = torch.FloatTensor([1.0, 1.0, 1.0])
        t2 = torch.FloatTensor([2.0, 5.0, 5.0])

        assert adv_loss(t0).item() == 0.0
        assert adv_loss(t1).item() == 0.0
        assert adv_loss(t2).item() == 11.0


class TestDiscriminatorAdversarialLoss:

    def _disc_loss(self, loss_type: str, fake: torch.Tensor, real: torch.Tensor):
        disc_loss_real = get_real_criterion(loss_type)
        disc_loss_fake = get_fake_criterion(loss_type)

        loss = disc_loss_fake(fake) + disc_loss_real(real)
        return loss

    def test_hinge_discriminator_adv_loss(self):
        loss_type = 'hinge'
        t0 = torch.FloatTensor([0.0, 0.0, 0.0])
        t1 = torch.FloatTensor([1.0, 2.0, 3.0])

        assert self._disc_loss(loss_type, t0, t0).item() == 2.0
        assert self._disc_loss(loss_type, t1, t1).item() == 3.0

    def test_mse_discriminator_adv_loss(self):
        loss_type = 'mse'

        t0 = torch.FloatTensor([0.0, 0.0, 0.0])
        t1 = torch.FloatTensor([1.0, 1.0, 1.0])

        assert self._disc_loss(loss_type, t0, t0).item() == 1.0
        assert self._disc_loss(loss_type, t1, t0).item() == 2.0


class TestFeatureMatchingLoss:

    def test_features_matching_loss_base(self):
        ft_matching_loss = FeatureMatchingLoss()
        length = random.randrange(1, 100_000)
        t1 = torch.randn(1, 2, length)

        loss = ft_matching_loss([t1], [t1])
        assert isinstance(loss, torch.Tensor)
        assert loss.item() == 0.0

    def test_features_matching_loss_raises_exception(self):
        ft_matching_loss = FeatureMatchingLoss()
        length = random.randrange(1, 100_000)
        t1 = torch.randn(1, 2, length)
        t2 = torch.randn(1, 2, length + 1)

        with pytest.raises(AssertionError):
            ft_matching_loss([], [])

        with pytest.raises(AssertionError):
            ft_matching_loss([t1], [t1, t1])

        with pytest.raises(AssertionError):
            ft_matching_loss([t1], [t2])

    def test_features_matching_loss_output(self):
        loss_nonorm = FeatureMatchingLoss(normalize=False)
        loss_layer_normed = FeatureMatchingLoss(normalize=True)

        length = random.randrange(1, 100_000)
        t1 = torch.randn(1, 2, length)
        t2 = torch.randn(1, 2, length)

        assert loss_nonorm([t1, t2], [t1, t2]).item() == 0.0
        assert loss_layer_normed([t1, t2], [t1, t2]).item() == 0.0

        t3 = torch.FloatTensor([1.0, 2.0, 3.0])
        t4 = torch.FloatTensor([2.0, 10.0, 3.0])

        assert loss_nonorm([t3], [t4]).item() == 3.0
        assert loss_nonorm([t3, t3], [t4, t4]).item() == 6.0

        assert loss_layer_normed([t3], [t4]).item() == 3.0
        assert loss_layer_normed([t3, t3], [t4, t4]).item() == 3.0
