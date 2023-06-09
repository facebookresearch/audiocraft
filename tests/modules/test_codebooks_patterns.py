# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from audiocraft.modules.codebooks_patterns import (
    DelayedPatternProvider,
    ParallelPatternProvider,
    Pattern,
    UnrolledPatternProvider,
)


class TestParallelPatternProvider:

    @pytest.mark.parametrize("n_q", [1, 4, 32])
    @pytest.mark.parametrize("timesteps", [0, 1, 16, 100])
    def test_get_pattern(self, n_q: int, timesteps: int):
        provider = ParallelPatternProvider(n_q)
        pattern = provider.get_pattern(timesteps)
        # + 1 to account for 1st step
        assert len(pattern.layout) == timesteps + 1

    @pytest.mark.parametrize("n_q", [1, 4, 32])
    @pytest.mark.parametrize("timesteps", [8, 16, 100])
    def test_pattern_content(self, n_q: int, timesteps: int):
        provider = ParallelPatternProvider(n_q)
        pattern = provider.get_pattern(timesteps)
        for s, v in enumerate(pattern.layout):
            for i, code in enumerate(v):
                assert i == code.q
                assert code.t == s - 1  # account for the 1st empty step

    @pytest.mark.parametrize("n_q", [1, 4, 32])
    @pytest.mark.parametrize("timesteps", [8, 16, 100])
    def test_pattern_max_delay(self, n_q: int, timesteps: int):
        provider = ParallelPatternProvider(n_q)
        pattern = provider.get_pattern(timesteps)
        assert pattern.max_delay == 0
        assert len(pattern.valid_layout) == len(pattern.layout) - pattern.max_delay


class TestDelayedPatternProvider:

    @pytest.mark.parametrize("n_q", [1, 4, 32])
    @pytest.mark.parametrize("timesteps", [0, 1, 16, 100])
    def test_get_pattern(self, n_q: int, timesteps: int):
        delays = [
            list(range(n_q)),
            [0] + [1] * (n_q - 1),
            [0] + [4] * (n_q - 1),
        ]
        for delay in delays:
            provider = DelayedPatternProvider(n_q, delay)
            pattern = provider.get_pattern(timesteps)
            # + 1 to account for 1st step
            assert len(pattern.layout) == timesteps + max(delay) + 1

    @pytest.mark.parametrize("n_q", [1, 4, 32])
    @pytest.mark.parametrize("timesteps", [8, 16, 100])
    def test_pattern_content(self, n_q: int, timesteps: int):
        provider = DelayedPatternProvider(n_q)
        pattern = provider.get_pattern(timesteps)
        for s, v in enumerate(pattern.layout):
            for i, code in enumerate(v):
                assert i == code.q
                assert code.t == max(0, s - code.q - 1)

    @pytest.mark.parametrize("timesteps", [8, 16, 100])
    @pytest.mark.parametrize("delay", [[0, 1, 2, 3], [0, 1, 1, 1], [0, 3, 3, 3], [0, 3]])
    def test_pattern_max_delay(self, timesteps: int, delay: list):
        provider = DelayedPatternProvider(len(delay), delay)
        pattern = provider.get_pattern(timesteps)
        assert pattern.max_delay == max(delay)
        assert len(pattern.valid_layout) == len(pattern.layout) - pattern.max_delay


class TestUnrolledPatternProvider:

    @pytest.mark.parametrize("timesteps", [0, 1, 16])
    @pytest.mark.parametrize("flattening", [[0, 1, 2], [0, 1, 1]])
    @pytest.mark.parametrize("delays", [[0, 0, 0], [0, 5, 5]])
    def test_get_pattern(self, timesteps: int, flattening: list, delays: list):
        n_q = len(flattening)
        max_delay = max(delays)
        provider = UnrolledPatternProvider(n_q, flattening, delays)
        pattern = provider.get_pattern(timesteps)
        assert len(pattern.layout) == provider.num_virtual_steps(timesteps) + max_delay

    @pytest.mark.parametrize("timesteps", [0, 1, 16])
    @pytest.mark.parametrize("flattening", [[0, 1, 2], [0, 1, 1]])
    @pytest.mark.parametrize("delays", [[0, 0, 0], [0, 5, 5]])
    def test_pattern_max_delay(self, timesteps: int, flattening: list, delays: list):
        n_q = len(flattening)
        max_delay = max(delays)
        provider = UnrolledPatternProvider(n_q, flattening, delays)
        pattern = provider.get_pattern(timesteps)
        assert pattern.max_delay == max_delay


class TestPattern:

    def ref_build_pattern_sequence(self, z: torch.Tensor, pattern: Pattern, special_token: int):
        """Reference method to build the sequence from the pattern without using fancy scatter."""
        bs, n_q, T = z.shape
        z = z.cpu().numpy()
        assert n_q == pattern.n_q
        assert T <= pattern.timesteps
        inp = torch.full((bs, n_q, len(pattern.layout)), special_token, dtype=torch.long).numpy()
        inp[:] = special_token
        for s, v in enumerate(pattern.layout):
            for (t, q) in v:
                if t < T:
                    inp[:, q, s] = z[:, q, t]
        return torch.from_numpy(inp)

    def ref_revert_pattern_sequence(self, z: torch.Tensor, pattern: Pattern, special_token: int):
        """Reference method to revert the sequence from the pattern without using fancy scatter."""
        z = z.cpu().numpy()
        bs, n_q, S = z.shape
        assert pattern.n_q == n_q
        inp = torch.full((bs, pattern.n_q, pattern.timesteps), special_token, dtype=torch.long).numpy()
        inp[:] = special_token
        for s, v in enumerate(pattern.layout):
            for (t, q) in v:
                if t < pattern.timesteps:
                    inp[:, q, t] = z[:, q, s]
        return torch.from_numpy(inp)

    def ref_revert_pattern_logits(self, z: torch.Tensor, pattern: Pattern, special_token: float):
        """Reference method to revert the logits from the pattern without using fancy scatter."""
        z = z.cpu().numpy()
        bs, card, n_q, S = z.shape
        assert pattern.n_q == n_q
        ref_layout = pattern.layout
        inp = torch.full((bs, card, pattern.n_q, pattern.timesteps), special_token, dtype=torch.float).numpy()
        inp[:] = special_token
        for s, v in enumerate(ref_layout[1:]):
            if s < S:
                for (t, q) in v:
                    if t < pattern.timesteps:
                        inp[:, :, q, t] = z[:, :, q, s]
        return torch.from_numpy(inp)

    def _get_pattern_providers(self, n_q: int):
        pattern_provider_1 = ParallelPatternProvider(n_q)
        pattern_provider_2 = DelayedPatternProvider(n_q, list(range(n_q)))
        pattern_provider_3 = DelayedPatternProvider(n_q, [0] + [1] * (n_q - 1))
        pattern_provider_4 = UnrolledPatternProvider(
            n_q, flattening=list(range(n_q)), delays=[0] * n_q
        )
        pattern_provider_5 = UnrolledPatternProvider(
            n_q, flattening=[0] + [1] * (n_q - 1), delays=[0] * n_q
        )
        pattern_provider_6 = UnrolledPatternProvider(
            n_q, flattening=[0] + [1] * (n_q - 1), delays=[0] + [5] * (n_q - 1)
        )
        return [
            pattern_provider_1,
            pattern_provider_2,
            pattern_provider_3,
            pattern_provider_4,
            pattern_provider_5,
            pattern_provider_6,
        ]

    @pytest.mark.parametrize("n_q", [1, 4, 32])
    @pytest.mark.parametrize("timesteps", [16, 72])
    def test_build_pattern_sequence(self, n_q: int, timesteps: int):
        bs = 2
        card = 256
        special_token = card

        pattern_providers = self._get_pattern_providers(n_q)
        for pattern_provider in pattern_providers:
            pattern = pattern_provider.get_pattern(timesteps)
            # we can correctly build the sequence from the pattern
            z = torch.randint(0, card, (bs, n_q, timesteps))
            ref_res = self.ref_build_pattern_sequence(z, pattern, special_token)
            res, indexes, mask = pattern.build_pattern_sequence(z, special_token)
            assert (res == ref_res).float().mean() == 1.0

            # expected assertion fails on the number of timesteps
            invalid_timesteps = [timesteps + 1]
            if pattern.num_sequence_steps != pattern.timesteps:
                invalid_timesteps.append(pattern.num_sequence_steps)
            for i_timesteps in invalid_timesteps:
                z2 = torch.randint(0, card, (bs, n_q, i_timesteps))
                with pytest.raises(AssertionError):
                    pattern.build_pattern_sequence(z2, special_token)

            # expected assertion fails on the number of codebooks
            invalid_qs = [0, n_q - 1, n_q + 1]
            for i_q in invalid_qs:
                z3 = torch.randint(0, card, (bs, i_q, timesteps))
                with pytest.raises(AssertionError):
                    pattern.build_pattern_sequence(z3, special_token)

    @pytest.mark.parametrize("n_q", [1, 4, 32])
    @pytest.mark.parametrize("timesteps", [16, 72])
    def test_revert_pattern_sequence(self, n_q: int, timesteps: int):
        bs = 2
        card = 256
        special_token = card

        pattern_providers = self._get_pattern_providers(n_q)
        for pattern_provider in pattern_providers:
            pattern = pattern_provider.get_pattern(timesteps)
            # this works assuming previous tests are successful
            z = torch.randint(0, card, (bs, n_q, timesteps))
            s = self.ref_build_pattern_sequence(z, pattern, special_token)
            ref_out = self.ref_revert_pattern_sequence(s, pattern, special_token)
            # ensure our reference script retrieve the original sequence
            assert z.shape == ref_out.shape
            assert (z == ref_out).float().mean() == 1.0
            # now we can test the scatter version
            out, indexes, mask = pattern.revert_pattern_sequence(s, special_token)
            assert out.shape == ref_out.shape
            assert (out == ref_out).float().mean() == 1.0

    @pytest.mark.parametrize("n_q", [1, 4, 32])
    @pytest.mark.parametrize("timesteps", [16, 72])
    @pytest.mark.parametrize("card", [1, 2, 256, 1024])
    def test_revert_pattern_logits(self, n_q: int, timesteps: int, card: int):
        bs = 2
        special_token = card
        logits_special_token = float('nan')

        pattern_providers = self._get_pattern_providers(n_q)
        for pattern_provider in pattern_providers:
            pattern = pattern_provider.get_pattern(timesteps)
            # this works assuming previous tests are successful
            z = torch.randint(0, card, (bs, n_q, timesteps))
            s = self.ref_build_pattern_sequence(z, pattern, special_token)
            logits = torch.randn((bs, card, n_q, s.shape[-1]))
            ref_out = self.ref_revert_pattern_logits(logits, pattern, logits_special_token)
            # ensure our reference script retrieve the original sequence
            assert ref_out.shape == torch.Size([bs, card, n_q, timesteps])
            # now we can test the scatter version
            out, indexes, mask = pattern.revert_pattern_logits(logits, logits_special_token)
            assert out.shape == ref_out.shape
            assert (out == ref_out).float().mean() == 1.0
