"""AudioCraft model wrapper for the web UI."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from einops import rearrange

from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio
from audiocraft.models import AudioGen, MusicGen, MultiBandDiffusion
from audiocraft.models.encodec import InterleaveStereoCompressionModel


class JobStatus(str, Enum):
    PENDING = "pending"
    LOADING = "loading"
    GENERATING = "generating"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class GenerationJob:
    id: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = "Queued"
    audio_path: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)


MUSIC_MODELS = [
    "facebook/musicgen-small",
    "facebook/musicgen-medium",
    "facebook/musicgen-melody",
    "facebook/musicgen-stereo-small",
    "facebook/musicgen-stereo-medium",
    "facebook/musicgen-stereo-melody",
]

SOUND_MODELS = [
    "facebook/audiogen-medium",
]


class ModelManager:
    """Lazy-loads and caches AudioCraft models."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._music_model: Optional[MusicGen] = None
        self._music_model_name: Optional[str] = None
        self._sound_model: Optional[AudioGen] = None
        self._sound_model_name: Optional[str] = None
        self._mbd: Optional[MultiBandDiffusion] = None
        self._lock = threading.Lock()
        self.jobs: dict[str, GenerationJob] = {}

    @property
    def device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        return self.jobs.get(job_id)

    def _load_music_model(self, name: str) -> MusicGen:
        with self._lock:
            if self._music_model is None or self._music_model_name != name:
                if self._music_model is not None:
                    del self._music_model
                    torch.cuda.empty_cache()
                self._music_model = MusicGen.get_pretrained(name, device=self.device)
                self._music_model_name = name
            return self._music_model

    def _load_sound_model(self, name: str) -> AudioGen:
        with self._lock:
            if self._sound_model is None or self._sound_model_name != name:
                if self._sound_model is not None:
                    del self._sound_model
                    torch.cuda.empty_cache()
                self._sound_model = AudioGen.get_pretrained(name, device=self.device)
                self._sound_model_name = name
            return self._sound_model

    def _load_mbd(self) -> MultiBandDiffusion:
        with self._lock:
            if self._mbd is None:
                self._mbd = MultiBandDiffusion.get_mbd_musicgen()
            return self._mbd

    def _update_job(self, job: GenerationJob, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(job, key, value)

    def _progress_callback(self, job: GenerationJob):
        def callback(generated: int, total: int) -> None:
            if total > 0:
                self._update_job(
                    job,
                    progress=min(generated / total, 0.99),
                    message=f"Generating… {generated}/{total} tokens",
                    status=JobStatus.GENERATING,
                )

        return callback

    def start_music_job(
        self,
        prompt: str,
        model: str,
        duration: float,
        top_k: int,
        top_p: float,
        temperature: float,
        cfg_coef: float,
        melody: Optional[tuple[int, np.ndarray]] = None,
        use_diffusion: bool = False,
    ) -> str:
        job_id = str(uuid.uuid4())
        job = GenerationJob(id=job_id)
        self.jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_music_job,
            args=(job, prompt, model, duration, top_k, top_p, temperature, cfg_coef, melody, use_diffusion),
            daemon=True,
        )
        thread.start()
        return job_id

    def start_sound_job(
        self,
        prompt: str,
        model: str,
        duration: float,
        top_k: int,
        top_p: float,
        temperature: float,
        cfg_coef: float,
    ) -> str:
        job_id = str(uuid.uuid4())
        job = GenerationJob(id=job_id)
        self.jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_sound_job,
            args=(job, prompt, model, duration, top_k, top_p, temperature, cfg_coef),
            daemon=True,
        )
        thread.start()
        return job_id

    def _run_music_job(
        self,
        job: GenerationJob,
        prompt: str,
        model: str,
        duration: float,
        top_k: int,
        top_p: float,
        temperature: float,
        cfg_coef: float,
        melody: Optional[tuple[int, np.ndarray]],
        use_diffusion: bool,
    ) -> None:
        try:
            self._update_job(job, status=JobStatus.LOADING, message="Loading model…", progress=0.05)
            music_model = self._load_music_model(model)
            music_model.set_generation_params(
                duration=duration,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                cfg_coef=cfg_coef,
            )
            music_model.set_custom_progress_callback(self._progress_callback(job))

            processed_melody = None
            if melody is not None:
                sr, audio = melody
                wav = torch.from_numpy(audio).to(music_model.device).float().t()
                if wav.dim() == 1:
                    wav = wav[None]
                wav = wav[..., : int(sr * duration)]
                processed_melody = convert_audio(wav, sr, 32000, 1)

            self._update_job(job, status=JobStatus.GENERATING, message="Generating music…", progress=0.1)

            if processed_melody is not None:
                outputs = music_model.generate_with_chroma(
                    descriptions=[prompt],
                    melody_wavs=[processed_melody],
                    melody_sample_rate=32000,
                    progress=True,
                    return_tokens=use_diffusion,
                )
            else:
                outputs = music_model.generate(
                    [prompt],
                    progress=True,
                    return_tokens=use_diffusion,
                )

            if use_diffusion:
                self._update_job(job, message="Running MultiBand Diffusion…", progress=0.85)
                mbd = self._load_mbd()
                tokens = outputs[1]
                if isinstance(music_model.compression_model, InterleaveStereoCompressionModel):
                    left, right = music_model.compression_model.get_left_right_codes(tokens)
                    tokens = torch.cat([left, right])
                decoded = mbd.tokens_to_wav(tokens)
                if isinstance(music_model.compression_model, InterleaveStereoCompressionModel):
                    assert decoded.shape[1] == 1
                    decoded = rearrange(decoded, "(s b) c t -> b (s c) t", s=2)
                output = decoded[0]
            else:
                output = outputs[0]

            output = output.detach().cpu().float()
            out_path = self.output_dir / f"{job.id}.wav"
            audio_write(
                str(out_path.with_suffix("")),
                output,
                music_model.sample_rate,
                strategy="loudness",
                loudness_headroom_db=16,
                loudness_compressor=True,
                add_suffix=False,
            )
            self._update_job(
                job,
                status=JobStatus.COMPLETE,
                progress=1.0,
                message="Done",
                audio_path=str(out_path),
            )
        except Exception as exc:
            self._update_job(
                job,
                status=JobStatus.ERROR,
                error=str(exc),
                message="Generation failed",
            )

    def _run_sound_job(
        self,
        job: GenerationJob,
        prompt: str,
        model: str,
        duration: float,
        top_k: int,
        top_p: float,
        temperature: float,
        cfg_coef: float,
    ) -> None:
        try:
            self._update_job(job, status=JobStatus.LOADING, message="Loading model…", progress=0.05)
            sound_model = self._load_sound_model(model)
            sound_model.set_generation_params(
                duration=duration,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                cfg_coef=cfg_coef,
            )
            sound_model.set_custom_progress_callback(self._progress_callback(job))

            self._update_job(job, status=JobStatus.GENERATING, message="Generating sound…", progress=0.1)
            outputs = sound_model.generate([prompt], progress=True)
            output = outputs[0].detach().cpu().float()

            out_path = self.output_dir / f"{job.id}.wav"
            audio_write(
                str(out_path.with_suffix("")),
                output,
                sound_model.sample_rate,
                strategy="loudness",
                loudness_headroom_db=16,
                loudness_compressor=True,
                add_suffix=False,
            )
            self._update_job(
                job,
                status=JobStatus.COMPLETE,
                progress=1.0,
                message="Done",
                audio_path=str(out_path),
            )
        except Exception as exc:
            self._update_job(
                job,
                status=JobStatus.ERROR,
                error=str(exc),
                message="Generation failed",
            )
