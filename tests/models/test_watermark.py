# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from audiocraft.models.watermark import AudioSeal
from tests.common_utils.wav_utils import get_white_noise


class TestWatermarkModel:

    def test_base(self):
        sr = 16_000
        duration = 1.0
        wav = get_white_noise(1, int(sr * duration)).unsqueeze(0)
        wm = AudioSeal.get_pretrained(name="base")

        secret_message = torch.randint(0, 2, (1, 16), dtype=torch.int32)
        watermarked_wav = wm(wav, message=secret_message, sample_rate=sr, alpha=0.8)
        result = wm.detect_watermark(watermarked_wav)

        detected = (
            torch.count_nonzero(torch.gt(result[:, 1, :], 0.5)) / result.shape[-1]
        )
        detect_prob = detected.cpu().item()  # type: ignore

        assert detect_prob >= 0.0
