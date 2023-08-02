# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import json
import logging
from pathlib import Path
import tempfile
import typing as tp
import subprocess
import shutil

import torch
import torchaudio

logger = logging.getLogger(__name__)


class ViSQOL:
    """ViSQOL wrapper to run ViSQOL from Python using a pre-installed binary.

    To learn more about ViSQOL and how to build ViSQOL binary using bazel, please refer to the
    instructions available in the open source repository: https://github.com/google/visqol

    ViSQOL is capable of running in two modes:

    Audio Mode:
        When running in audio mode, input signals must have a 48kHz sample rate. Input should be resampled to 48kHz.
        Input signals can be multi-channel, but they will be down-mixed to mono for performing the comparison.
        Audio mode uses support vector regression, with the maximum range at ~4.75.

    Speech Mode:
        When running in speech mode, ViSQOL uses a wideband model. It therefore expects input sample rates of 16kHz.
            Input should be resampled to 16kHz.
        As part of the speech mode processing, a root mean square implementation for voice activity detection
            is performed on the reference signal to determine what parts of the signal have voice activity and
            should therefore be included in the comparison. The signal is normalized before performing the voice
            activity detection.
        Input signals can be multi-channel, but they will be down-mixed to mono for performing the comparison.
        Speech mode is scaled to have a maximum MOS of 5.0 to match previous version behavior.

    For more details, check the guidelines: https://github.com/google/visqol#general-guidelines-for-input

    Args:
        visqol_bin (str): Path to the ViSQOL binary.
        mode (str): ViSQOL computation mode, expecting "audio" or "speech".
        model (str): Name of the model to use for similarity to quality model.
        debug (bool): Whether to also get debug metrics from ViSQOL or not.
    """
    SAMPLE_RATES_MODES = {"audio": 48_000, "speech": 16_000}
    ALLOWED_SAMPLE_RATES = frozenset(SAMPLE_RATES_MODES.values())

    def __init__(self, bin: tp.Union[Path, str], mode: str = "audio",
                 model: str = "libsvm_nu_svr_model.txt", debug: bool = False):
        assert bin is not None and Path(bin).exists(), f"Could not find ViSQOL binary in specified path: {bin}"
        self.visqol_bin = str(bin)
        self.visqol_mode = mode
        self.target_sr = self._get_target_sr(self.visqol_mode)
        self.model = model
        self.debug = debug
        assert Path(self.visqol_model).exists(), \
            f"Could not find the specified model in ViSQOL install: {self.visqol_model}"

    def _get_target_sr(self, mode: str) -> int:
        # returns target sampling rate for the corresponding ViSQOL mode.
        if mode not in ViSQOL.SAMPLE_RATES_MODES:
            raise ValueError(
                f"Unsupported mode! Allowed are: {', '.join(ViSQOL.SAMPLE_RATES_MODES.keys())}"
            )
        return ViSQOL.SAMPLE_RATES_MODES[mode]

    def _prepare_files(
        self, ref_sig: torch.Tensor, deg_sig: torch.Tensor, sr: int, target_sr: int, pad_with_silence: bool = False
    ):
        # prepare files for ViSQOL evaluation.
        assert target_sr in ViSQOL.ALLOWED_SAMPLE_RATES
        assert len(ref_sig) == len(deg_sig), (
            "Expects same number of ref and degraded inputs",
            f" but ref len {len(ref_sig)} != deg len {len(deg_sig)}"
        )
        # resample audio if needed
        if sr != target_sr:
            transform = torchaudio.transforms.Resample(sr, target_sr)
            pad = int(0.5 * target_sr)
            rs_ref = []
            rs_deg = []
            for i in range(len(ref_sig)):
                rs_ref_i = transform(ref_sig[i])
                rs_deg_i = transform(deg_sig[i])
                if pad_with_silence:
                    rs_ref_i = torch.nn.functional.pad(rs_ref_i, (pad, pad), mode='constant', value=0)
                    rs_deg_i = torch.nn.functional.pad(rs_deg_i, (pad, pad), mode='constant', value=0)
                rs_ref.append(rs_ref_i)
                rs_deg.append(rs_deg_i)
            ref_sig = torch.stack(rs_ref)
            deg_sig = torch.stack(rs_deg)
        # save audio chunks to tmp dir and create csv
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            tmp_input_csv_path = tmp_dir / "input.csv"
            tmp_results_csv_path = tmp_dir / "results.csv"
            tmp_debug_json_path = tmp_dir / "debug.json"
            with open(tmp_input_csv_path, "w") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["reference", "degraded"])
                for i in range(len(ref_sig)):
                    tmp_ref_filename = tmp_dir / f"ref_{i}.wav"
                    tmp_deg_filename = tmp_dir / f"deg_{i}.wav"
                    torchaudio.save(
                        tmp_ref_filename,
                        torch.clamp(ref_sig[i], min=-0.99, max=0.99),
                        sample_rate=target_sr,
                        bits_per_sample=16,
                        encoding="PCM_S"
                    )
                    torchaudio.save(
                        tmp_deg_filename,
                        torch.clamp(deg_sig[i], min=-0.99, max=0.99),
                        sample_rate=target_sr,
                        bits_per_sample=16,
                        encoding="PCM_S"
                    )
                    csv_writer.writerow([str(tmp_ref_filename), str(tmp_deg_filename)])
            return tmp_dir, tmp_input_csv_path, tmp_results_csv_path, tmp_debug_json_path
        except Exception as e:
            logger.error("Exception occurred when preparing files for ViSQOL: %s", e)
            return tmp_dir, None, None, None

    def _flush_files(self, tmp_dir: tp.Union[Path, str]):
        # flush tmp files used to compute ViSQOL.
        shutil.rmtree(str(tmp_dir))

    def _collect_moslqo_score(self, results_csv_path: tp.Union[Path, str]) -> float:
        # collect results for each evaluated pair and return averaged moslqo score.
        with open(results_csv_path, "r") as csv_file:
            reader = csv.DictReader(csv_file)
            moslqo_scores = [float(row["moslqo"]) for row in reader]
            if len(moslqo_scores) > 0:
                return sum(moslqo_scores) / len(moslqo_scores)
            else:
                return 0.0

    def _collect_debug_data(self, debug_json_path: tp.Union[Path, str]) -> dict:
        # collect debug data for the visqol inference.
        with open(debug_json_path, "r") as f:
            data = json.load(f)
            return data

    @property
    def visqol_model(self):
        return f'{self.visqol_bin}/model/{self.model}'

    def _run_visqol(
        self,
        input_csv_path: tp.Union[Path, str],
        results_csv_path: tp.Union[Path, str],
        debug_csv_path: tp.Optional[tp.Union[Path, str]],
    ):
        input_csv_path = str(input_csv_path)
        results_csv_path = str(results_csv_path)
        debug_csv_path = str(debug_csv_path)
        cmd = [
            f'{self.visqol_bin}/bazel-bin/visqol',
            '--batch_input_csv', f'{input_csv_path}',
            '--results_csv', f'{results_csv_path}'
        ]
        if debug_csv_path is not None:
            cmd += ['--output_debug', f'{debug_csv_path}']
        if self.visqol_mode == "speech":
            cmd += ['--use_speech_mode']
        cmd += ['--similarity_to_quality_model', f'{self.visqol_model}']
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode:
            logger.error("Error with visqol: \n %s \n %s", result.stdout.decode(), result.stderr.decode())
            raise RuntimeError("Error while executing visqol")
        result.check_returncode()

    def __call__(
        self,
        ref_sig: torch.Tensor,
        deg_sig: torch.Tensor,
        sr: int,
        pad_with_silence: bool = False,
    ):
        """Calculate the ViSQOL metric for a pair of audio signals at a given sample rate.
        Args:
            ref_sig (torch.Tensor): Reference signals as [B, C, T].
            deg_sig (torch.Tensor): Degraded signals as [B, C, T].
            sr (int): Sample rate of the two audio signals.
            pad_with_silence (bool): Whether to pad the file with silences as recommended
                in visqol guidelines (see: https://github.com/google/visqol#general-guidelines-for-input).
        Returns:
            float: The ViSQOL score or mean score for the batch.
        """
        logger.debug(f"Calculating visqol with mode={self.visqol_mode} on {len(ref_sig)} samples")
        tmp_dir, input_csv, results_csv, debug_json = self._prepare_files(
            ref_sig, deg_sig, sr, self.target_sr, pad_with_silence
        )
        try:
            if input_csv and results_csv:
                self._run_visqol(
                    input_csv,
                    results_csv,
                    debug_json if self.debug else None,
                )
                mosqol = self._collect_moslqo_score(results_csv)
                return mosqol
            else:
                raise RuntimeError("Something unexpected happened when running VISQOL!")
        except Exception as e:
            logger.error("Exception occurred when running ViSQOL: %s", e)
        finally:
            self._flush_files(tmp_dir)
