# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Audio IO methods are defined in this module (info, read, write),
We rely on av library for faster read when possible, otherwise on torchaudio.
"""

from dataclasses import dataclass
from pathlib import Path
import logging
import typing as tp

import numpy as np
import soundfile
import torch
from torch.nn import functional as F

import av
import subprocess as sp

from .audio_utils import f32_pcm, normalize_audio


_av_initialized = False


def _init_av():
    global _av_initialized
    if _av_initialized:
        return
    logger = logging.getLogger('libav.mp3')
    logger.setLevel(logging.ERROR)
    _av_initialized = True


@dataclass(frozen=True)
class AudioFileInfo:
    sample_rate: int
    duration: float
    channels: int


def _av_info(filepath: tp.Union[str, Path]) -> AudioFileInfo:
    _init_av()
    with av.open(str(filepath)) as af:
        stream = af.streams.audio[0]
        sample_rate = stream.codec_context.sample_rate
        duration = float(stream.duration * stream.time_base)
        channels = stream.channels
        return AudioFileInfo(sample_rate, duration, channels)


def _soundfile_info(filepath: tp.Union[str, Path]) -> AudioFileInfo:
    info = soundfile.info(filepath)
    return AudioFileInfo(info.samplerate, info.duration, info.channels)


def audio_info(filepath: tp.Union[str, Path]) -> AudioFileInfo:
    # torchaudio no longer returns useful duration informations for some formats like mp3s.
    filepath = Path(filepath)
    if filepath.suffix in ['.flac', '.ogg']:  # TODO: Validate .ogg can be safely read with av_info
        # ffmpeg has some weird issue with flac.
        return _soundfile_info(filepath)
    else:
        return _av_info(filepath)


def _av_read(filepath: tp.Union[str, Path], seek_time: float = 0, duration: float = -1.) -> tp.Tuple[torch.Tensor, int]:
    """FFMPEG-based audio file reading using PyAV bindings.
    Soundfile cannot read mp3 and av_read is more efficient than torchaudio.

    Args:
        filepath (str or Path): Path to audio file to read.
        seek_time (float): Time at which to start reading in the file.
        duration (float): Duration to read from the file. If set to -1, the whole file is read.
    Returns:
        tuple of torch.Tensor, int: Tuple containing audio data and sample rate
    """
    _init_av()
    with av.open(str(filepath)) as af:
        stream = af.streams.audio[0]
        sr = stream.codec_context.sample_rate
        num_frames = int(sr * duration) if duration >= 0 else -1
        frame_offset = int(sr * seek_time)
        # we need a small negative offset otherwise we get some edge artifact
        # from the mp3 decoder.
        af.seek(int(max(0, (seek_time - 0.1)) / stream.time_base), stream=stream)
        frames = []
        length = 0
        for frame in af.decode(streams=stream.index):
            current_offset = int(frame.rate * frame.pts * frame.time_base)
            strip = max(0, frame_offset - current_offset)
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != stream.channels:
                buf = buf.view(-1, stream.channels).t()
            buf = buf[:, strip:]
            frames.append(buf)
            length += buf.shape[1]
            if num_frames > 0 and length >= num_frames:
                break
        assert frames
        # If the above assert fails, it is likely because we seeked past the end of file point,
        # in which case ffmpeg returns a single frame with only zeros, and a weird timestamp.
        # This will need proper debugging, in due time.
        wav = torch.cat(frames, dim=1)
        assert wav.shape[0] == stream.channels
        if num_frames > 0:
            wav = wav[:, :num_frames]
        return f32_pcm(wav), sr


def audio_read(filepath: tp.Union[str, Path], seek_time: float = 0.,
               duration: float = -1.0, pad: bool = False) -> tp.Tuple[torch.Tensor, int]:
    """Read audio by picking the most appropriate backend tool based on the audio format.

    Args:
        filepath (str or Path): Path to audio file to read.
        seek_time (float): Time at which to start reading in the file.
        duration (float): Duration to read from the file. If set to -1, the whole file is read.
        pad (bool): Pad output audio if not reaching expected duration.
    Returns:
        tuple of torch.Tensor, int: Tuple containing audio data and sample rate.
    """
    fp = Path(filepath)
    if fp.suffix in ['.flac', '.ogg']:  # TODO: check if we can safely use av_read for .ogg
        # There is some bug with ffmpeg and reading flac
        info = _soundfile_info(filepath)
        frames = -1 if duration <= 0 else int(duration * info.sample_rate)
        frame_offset = int(seek_time * info.sample_rate)
        wav, sr = soundfile.read(filepath, start=frame_offset, frames=frames, dtype=np.float32)
        assert info.sample_rate == sr, f"Mismatch of sample rates {info.sample_rate} {sr}"
        wav = torch.from_numpy(wav).t().contiguous()
        if len(wav.shape) == 1:
            wav = torch.unsqueeze(wav, 0)
    else:
        wav, sr = _av_read(filepath, seek_time, duration)
    if pad and duration > 0:
        expected_frames = int(duration * sr)
        wav = F.pad(wav, (0, expected_frames - wav.shape[-1]))
    return wav, sr


def _piping_to_ffmpeg(out_path: tp.Union[str, Path], wav: torch.Tensor, sample_rate: int, flags: tp.List[str]):
    # ffmpeg is always installed and torchaudio is a bit unstable lately, so let's bypass it entirely.
    assert wav.dim() == 2, wav.shape
    command = [
        'ffmpeg',
        '-loglevel', 'error',
        '-y', '-f', 'f32le', '-ar', str(sample_rate), '-ac', str(wav.shape[0]),
        '-i', '-'] + flags + [str(out_path)]
    input_ = f32_pcm(wav).t().detach().cpu().numpy().tobytes()
    sp.run(command, input=input_, check=True)


def audio_write(stem_name: tp.Union[str, Path],
                wav: torch.Tensor, sample_rate: int,
                format: str = 'wav', mp3_rate: int = 320, ogg_rate: tp.Optional[int] = None,
                normalize: bool = True, strategy: str = 'peak', peak_clip_headroom_db: float = 1,
                rms_headroom_db: float = 18, loudness_headroom_db: float = 14,
                loudness_compressor: bool = False,
                log_clipping: bool = True, make_parent_dir: bool = True,
                add_suffix: bool = True) -> Path:
    """Convenience function for saving audio to disk. Returns the filename the audio was written to.

    Args:
        stem_name (str or Path): Filename without extension which will be added automatically.
        wav (torch.Tensor): Audio data to save.
        sample_rate (int): Sample rate of audio data.
        format (str): Either "wav", "mp3", "ogg", or "flac".
        mp3_rate (int): kbps when using mp3s.
        ogg_rate (int): kbps when using ogg/vorbis. If not provided, let ffmpeg decide for itself.
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
        loudness_compressor (bool): Uses tanh for soft clipping when strategy is 'loudness'.
         when strategy is 'loudness' log_clipping (bool): If True, basic logging on stderr when clipping still
            occurs despite strategy (only for 'rms').
        make_parent_dir (bool): Make parent directory if it doesn't exist.
    Returns:
        Path: Path of the saved audio.
    """
    assert wav.dtype.is_floating_point, "wav is not floating point"
    if wav.dim() == 1:
        wav = wav[None]
    elif wav.dim() > 2:
        raise ValueError("Input wav should be at most 2 dimension.")
    assert wav.isfinite().all()
    wav = normalize_audio(wav, normalize, strategy, peak_clip_headroom_db,
                          rms_headroom_db, loudness_headroom_db, loudness_compressor,
                          log_clipping=log_clipping, sample_rate=sample_rate,
                          stem_name=str(stem_name))
    if format == 'mp3':
        suffix = '.mp3'
        flags = ['-f', 'mp3', '-c:a', 'libmp3lame', '-b:a', f'{mp3_rate}k']
    elif format == 'wav':
        suffix = '.wav'
        flags = ['-f', 'wav', '-c:a', 'pcm_s16le']
    elif format == 'ogg':
        suffix = '.ogg'
        flags = ['-f', 'ogg', '-c:a', 'libvorbis']
        if ogg_rate is not None:
            flags += ['-b:a', f'{ogg_rate}k']
    elif format == 'flac':
        suffix = '.flac'
        flags = ['-f', 'flac']
    else:
        raise RuntimeError(f"Invalid format {format}. Only wav or mp3 are supported.")
    if not add_suffix:
        suffix = ''
    path = Path(str(stem_name) + suffix)
    if make_parent_dir:
        path.parent.mkdir(exist_ok=True, parents=True)
    try:
        _piping_to_ffmpeg(path, wav, sample_rate, flags)
    except Exception:
        if path.exists():
            # we do not want to leave half written files around.
            path.unlink()
        raise
    return path


def get_spec(y, sr=16000, n_fft=4096, hop_length=128, dur=8) -> np.ndarray:
    """Get the mel-spectrogram from the raw audio.

    Args:
        y (numpy array): raw input
        sr (int): Sampling rate
        n_fft (int): Number of samples per FFT. Default is 2048.
        hop_length (int): Number of samples between successive frames. Default is 512.
        dur (float): Maxium duration to get the spectrograms
    Returns:
        spectro histogram as a numpy array
    """
    import librosa
    import librosa.display

    spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db


def save_spectrograms(
    ys: tp.List[np.ndarray],
    sr: int,
    path: str,
    names: tp.List[str],
    n_fft: int = 4096,
    hop_length: int = 128,
    dur: float = 8.0,
):
    """Plot a spectrogram for an audio file.

    Args:
        ys: List of audio spectrograms
        sr (int): Sampling rate of the audio file. Default is 22050 Hz.
        path (str): Path to the plot file.
        names: name of each spectrogram plot
        n_fft (int): Number of samples per FFT. Default is 2048.
        hop_length (int): Number of samples between successive frames. Default is 512.
        dur (float): Maxium duration to plot the spectrograms

    Returns:
        None (plots the spectrogram using matplotlib)
    """
    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import librosa.display

    if not names:
        names = ["Ground Truth", "Audio Watermarked", "Watermark"]
    ys = [wav[: int(dur * sr)] for wav in ys]  # crop
    assert len(names) == len(
        ys
    ), f"There are {len(ys)} wavs but {len(names)} names ({names})"

    # Set matplotlib stuff
    BIGGER_SIZE = 10
    SMALLER_SIZE = 8
    linewidth = 234.8775  # linewidth in pt

    plt.rc("font", size=BIGGER_SIZE, family="serif")  # controls default text sizes
    plt.rcParams["font.family"] = "DeJavu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALLER_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)
    height = 1.6 * linewidth / 72.0
    fig, ax = plt.subplots(
        nrows=len(ys),
        ncols=1,
        sharex=True,
        figsize=(linewidth / 72.0, height),
    )
    fig.tight_layout()

    # Plot the spectrogram

    for i, ysi in enumerate(ys):
        spectrogram_db = get_spec(ysi, sr=sr, n_fft=n_fft, hop_length=hop_length)
        if i == 0:
            cax = fig.add_axes(
                [
                    ax[0].get_position().x1 + 0.01,  # type: ignore
                    ax[-1].get_position().y0,
                    0.02,
                    ax[0].get_position().y1 - ax[-1].get_position().y0,
                ]
            )
            fig.colorbar(
                mpl.cm.ScalarMappable(
                    norm=mpl.colors.Normalize(
                        np.min(spectrogram_db), np.max(spectrogram_db)
                    ),
                    cmap="magma",
                ),
                ax=ax,
                orientation="vertical",
                format="%+2.0f dB",
                cax=cax,
            )
        librosa.display.specshow(
            spectrogram_db,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            ax=ax[i],
        )
        ax[i].set(title=names[i])
        ax[i].yaxis.set_label_text(None)
        ax[i].label_outer()
    fig.savefig(path, bbox_inches="tight")
    plt.close()
