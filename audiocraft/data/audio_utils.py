# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Various utilities for audio convertion (pcm format, sample rate and channels),
and volume normalization."""
import io
import logging
import re
import sys
import typing as tp

import julius
import torch
import torchaudio

logger = logging.getLogger(__name__)


def convert_audio_channels(wav: torch.Tensor, channels: int = 2) -> torch.Tensor:
    """Convert audio to the given number of channels.

    Args:
        wav (torch.Tensor): Audio wave of shape [B, C, T].
        channels (int): Expected number of channels as output.
    Returns:
        torch.Tensor: Downmixed or unchanged audio wave [B, C, T].
    """
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, and the stream has multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file has
        # a single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file has
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav


def convert_audio(wav: torch.Tensor, from_rate: float,
                  to_rate: float, to_channels: int) -> torch.Tensor:
    """Convert audio to new sample rate and number of audio channels."""
    wav = julius.resample_frac(wav, int(from_rate), int(to_rate))
    wav = convert_audio_channels(wav, to_channels)
    return wav


def normalize_loudness(wav: torch.Tensor, sample_rate: int, loudness_headroom_db: float = 14,
                       loudness_compressor: bool = False, energy_floor: float = 2e-3):
    """Normalize an input signal to a user loudness in dB LKFS.
    Audio loudness is defined according to the ITU-R BS.1770-4 recommendation.

    Args:
        wav (torch.Tensor): Input multichannel audio data.
        sample_rate (int): Sample rate.
        loudness_headroom_db (float): Target loudness of the output in dB LUFS.
        loudness_compressor (bool): Uses tanh for soft clipping.
        energy_floor (float): anything below that RMS level will not be rescaled.
    Returns:
        torch.Tensor: Loudness normalized output data.
    """
    energy = wav.pow(2).mean().sqrt().item()
    if energy < energy_floor:
        return wav
    transform = torchaudio.transforms.Loudness(sample_rate)
    input_loudness_db = transform(wav).item()
    # calculate the gain needed to scale to the desired loudness level
    delta_loudness = -loudness_headroom_db - input_loudness_db
    gain = 10.0 ** (delta_loudness / 20.0)
    output = gain * wav
    if loudness_compressor:
        output = torch.tanh(output)
    assert output.isfinite().all(), (input_loudness_db, wav.pow(2).mean().sqrt())
    return output


def _clip_wav(wav: torch.Tensor, log_clipping: bool = False, stem_name: tp.Optional[str] = None) -> None:
    """
    Utility function to clip the audio with logging if specified.
    """
    max_scale = wav.abs().max()
    if log_clipping and max_scale > 1:
        clamp_prob = (wav.abs() > 1).float().mean().item()
        print(f"CLIPPING {stem_name or ''} happening with proba (a bit of clipping is okay):",
              clamp_prob, "maximum scale: ", max_scale.item(), file=sys.stderr)
    wav.clamp_(-1, 1)


def normalize_audio(wav: torch.Tensor, normalize: bool = True,
                    strategy: str = 'peak', peak_clip_headroom_db: float = 1,
                    rms_headroom_db: float = 18, loudness_headroom_db: float = 14,
                    loudness_compressor: bool = False, log_clipping: bool = False,
                    sample_rate: tp.Optional[int] = None,
                    stem_name: tp.Optional[str] = None) -> torch.Tensor:
    """Normalize the audio according to the prescribed strategy (see after).

    Args:
        wav (torch.Tensor): Audio data.
        normalize (bool): if `True` (default), normalizes according to the prescribed
            strategy (see after). If `False`, the strategy is only used in case clipping
            would happen.
        strategy (str): Can be either 'clip', 'peak', or 'rms'. Default is 'peak',
            i.e. audio is normalized by its largest value. RMS normalizes by root-mean-square
            with extra headroom to avoid clipping. 'clip' just clips.
        peak_clip_headroom_db (float): Headroom in dB when doing 'peak' or 'clip' strategy.
        rms_headroom_db (float): Headroom in dB when doing 'rms' strategy. This must be much larger
            than the `peak_clip` one to avoid further clipping.
        loudness_headroom_db (float): Target loudness for loudness normalization.
        loudness_compressor (bool): If True, uses tanh based soft clipping.
        log_clipping (bool): If True, basic logging on stderr when clipping still
            occurs despite strategy (only for 'rms').
        sample_rate (int): Sample rate for the audio data (required for loudness).
        stem_name (str, optional): Stem name for clipping logging.
    Returns:
        torch.Tensor: Normalized audio.
    """
    scale_peak = 10 ** (-peak_clip_headroom_db / 20)
    scale_rms = 10 ** (-rms_headroom_db / 20)
    if strategy == 'peak':
        rescaling = (scale_peak / wav.abs().max())
        if normalize or rescaling < 1:
            wav = wav * rescaling
    elif strategy == 'clip':
        wav = wav.clamp(-scale_peak, scale_peak)
    elif strategy == 'rms':
        mono = wav.mean(dim=0)
        rescaling = scale_rms / mono.pow(2).mean().sqrt()
        if normalize or rescaling < 1:
            wav = wav * rescaling
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    elif strategy == 'loudness':
        assert sample_rate is not None, "Loudness normalization requires sample rate."
        wav = normalize_loudness(wav, sample_rate, loudness_headroom_db, loudness_compressor)
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    else:
        assert wav.abs().max() < 1
        assert strategy == '' or strategy == 'none', f"Unexpected strategy: '{strategy}'"
    return wav


def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """
    Convert audio to float 32 bits PCM format.
    Args:
        wav (torch.tensor): Input wav tensor
    Returns:
        same wav in float32 PCM format
    """
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / 2**15
    elif wav.dtype == torch.int32:
        return wav.float() / 2**31
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def i16_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to int 16 bits PCM format.

    ..Warning:: There exist many formula for doing this conversion. None are perfect
    due to the asymmetry of the int16 range. One either have possible clipping, DC offset,
    or inconsistencies with f32_pcm. If the given wav doesn't have enough headroom,
    it is possible that `i16_pcm(f32_pcm)) != Identity`.
    Args:
        wav (torch.tensor): Input wav tensor
    Returns:
        same wav in float16 PCM format
    """
    if wav.dtype.is_floating_point:
        assert wav.abs().max() <= 1
        candidate = (wav * 2 ** 15).round()
        if candidate.max() >= 2 ** 15:  # clipping would occur
            candidate = (wav * (2 ** 15 - 1)).round()
        return candidate.short()
    else:
        assert wav.dtype == torch.int16
        return wav


def compress(wav: torch.Tensor, sr: int,
             target_format: tp.Literal["mp3", "ogg", "flac"] = "mp3",
             bitrate: str = "128k") -> tp.Tuple[torch.Tensor, int]:
    """Convert audio wave form to a specified lossy format: mp3, ogg, flac

    Args:
        wav (torch.Tensor): Input wav tensor.
        sr (int): Sampling rate.
        target_format (str): Compression format (e.g., 'mp3').
        bitrate (str): Bitrate for compression.

    Returns:
        Tuple of compressed WAV tensor and sampling rate.
    """

    # Extract the bit rate from string (e.g., '128k')
    match = re.search(r"\d+(\.\d+)?", str(bitrate))
    parsed_bitrate = float(match.group()) if match else None
    assert parsed_bitrate, f"Invalid bitrate specified (got {parsed_bitrate})"
    try:
        # Create a virtual file instead of saving to disk
        buffer = io.BytesIO()

        torchaudio.save(
            buffer, wav, sr, format=target_format, bits_per_sample=parsed_bitrate,
        )
        # Move to the beginning of the file
        buffer.seek(0)
        compressed_wav, sr = torchaudio.load(buffer)
        return compressed_wav, sr

    except RuntimeError:
        logger.warning(
            f"compression failed skipping compression: {format} {parsed_bitrate}"
        )
        return wav, sr


def get_mp3(wav_tensor: torch.Tensor, sr: int, bitrate: str = "128k") -> torch.Tensor:
    """Convert a batch of audio files to MP3 format, maintaining the original shape.

    This function takes a batch of audio files represented as a PyTorch tensor, converts
    them to MP3 format using the specified bitrate, and returns the batch in the same
    shape as the input.

    Args:
        wav_tensor (torch.Tensor): Batch of audio files represented as a tensor.
            Shape should be (batch_size, channels, length).
        sr (int): Sampling rate of the audio.
        bitrate (str): Bitrate for MP3 conversion, default is '128k'.

    Returns:
        torch.Tensor: Batch of audio files converted to MP3 format, with the same
            shape as the input tensor.
    """
    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape

    # Flatten tensor for conversion and move to CPU
    wav_tensor_flat = wav_tensor.view(1, -1).cpu()

    # Convert to MP3 format with specified bitrate
    wav_tensor_flat, _ = compress(wav_tensor_flat, sr, bitrate=bitrate)

    # Reshape back to original batch format and trim or pad if necessary
    wav_tensor = wav_tensor_flat.view(batch_size, channels, -1)
    compressed_length = wav_tensor.shape[-1]
    if compressed_length > original_length:
        wav_tensor = wav_tensor[:, :, :original_length]  # Trim excess frames
    elif compressed_length < original_length:
        padding = torch.zeros(
            batch_size, channels, original_length - compressed_length, device=device
        )
        wav_tensor = torch.cat((wav_tensor, padding), dim=-1)  # Pad with zeros

    # Move tensor back to the original device
    return wav_tensor.to(device)


def get_aac(
    wav_tensor: torch.Tensor,
    sr: int,
    bitrate: str = "128k",
    lowpass_freq: tp.Optional[int] = None,
) -> torch.Tensor:
    """Converts a batch of audio tensors to AAC format and then back to tensors.

    This function first saves the input tensor batch as WAV files, then uses FFmpeg to convert
    these WAV files to AAC format. Finally, it loads the AAC files back into tensors.

    Args:
        wav_tensor (torch.Tensor): A batch of audio files represented as a tensor.
                                   Shape should be (batch_size, channels, length).
        sr (int): Sampling rate of the audio.
        bitrate (str): Bitrate for AAC conversion, default is '128k'.
        lowpass_freq (Optional[int]): Frequency for a low-pass filter. If None, no filter is applied.

    Returns:
        torch.Tensor: Batch of audio files converted to AAC and back, with the same
                      shape as the input tensor.
    """
    import tempfile
    import subprocess

    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape

    # Parse the bitrate value from the string
    match = re.search(r"\d+(\.\d+)?", bitrate)
    parsed_bitrate = (
        match.group() if match else "128"
    )  # Default to 128 if parsing fails

    # Flatten tensor for conversion and move to CPU
    wav_tensor_flat = wav_tensor.view(1, -1).cpu()

    with tempfile.NamedTemporaryFile(
        suffix=".wav"
    ) as f_in, tempfile.NamedTemporaryFile(suffix=".aac") as f_out:
        input_path, output_path = f_in.name, f_out.name

        # Save the tensor as a WAV file
        torchaudio.save(input_path, wav_tensor_flat, sr, backend="ffmpeg")

        # Prepare FFmpeg command for AAC conversion
        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ar",
            str(sr),
            "-b:a",
            f"{parsed_bitrate}k",
            "-c:a",
            "aac",
        ]
        if lowpass_freq is not None:
            command += ["-cutoff", str(lowpass_freq)]
        command.append(output_path)

        try:
            # Run FFmpeg and suppress output
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Load the AAC audio back into a tensor
            aac_tensor, _ = torchaudio.load(output_path, backend="ffmpeg")
        except Exception as exc:
            raise RuntimeError(
                "Failed to run command " ".join(command)} "
                "(Often this means ffmpeg is not installed or the encoder is not supported, "
                "make sure you installed an older version ffmpeg<5)"
            ) from exc

    original_length_flat = batch_size * channels * original_length
    compressed_length_flat = aac_tensor.shape[-1]

    # Trim excess frames
    if compressed_length_flat > original_length_flat:
        aac_tensor = aac_tensor[:, :original_length_flat]

    # Pad the shortedn frames
    elif compressed_length_flat < original_length_flat:
        padding = torch.zeros(
            1, original_length_flat - compressed_length_flat, device=device
        )
        aac_tensor = torch.cat((aac_tensor, padding), dim=-1)

    # Reshape and adjust length to match original tensor
    wav_tensor = aac_tensor.view(batch_size, channels, -1)
    compressed_length = wav_tensor.shape[-1]

    assert compressed_length == original_length, (
        "AAC-compressed audio does not have the same frames as original one. "
        "One reason can be ffmpeg is not  installed and used as proper backed "
        "for torchaudio, or the AAC encoder is not correct. Run "
        "`torchaudio.utils.ffmpeg_utils.get_audio_encoders()` and make sure we see entry for"
        "AAC in the output."
    )
    return wav_tensor.to(device)
