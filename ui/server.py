"""AudioCraft unified web UI server."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ui.generator import MUSIC_MODELS, SOUND_MODELS, JobStatus, ModelManager

UI_DIR = Path(__file__).parent
STATIC_DIR = UI_DIR / "static"
OUTPUT_DIR = UI_DIR / "outputs"

manager = ModelManager(OUTPUT_DIR)

app = FastAPI(
    title="AudioCraft Studio",
    description="Web UI for MusicGen and AudioGen",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=512)
    model: str
    duration: float = Field(10.0, ge=1.0, le=120.0)
    top_k: int = Field(250, ge=0)
    top_p: float = Field(0.0, ge=0.0, le=1.0)
    temperature: float = Field(1.0, ge=0.0)
    cfg_coef: float = Field(3.0, ge=0.0)
    use_diffusion: bool = False


class JobResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    id: str
    status: str
    progress: float
    message: str
    audio_url: Optional[str] = None
    error: Optional[str] = None


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": manager.device,
        "music_models": MUSIC_MODELS,
        "sound_models": SOUND_MODELS,
    }


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    audio_url = None
    if job.status == JobStatus.COMPLETE and job.audio_path:
        audio_url = f"/api/audio/{job.id}.wav"

    return JobStatusResponse(
        id=job.id,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        audio_url=audio_url,
        error=job.error,
    )


@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/wav", filename=filename)


@app.post("/api/generate/music", response_model=JobResponse)
async def generate_music(
    prompt: str = Form(...),
    model: str = Form(...),
    duration: float = Form(10.0),
    top_k: int = Form(250),
    top_p: float = Form(0.0),
    temperature: float = Form(1.0),
    cfg_coef: float = Form(3.0),
    use_diffusion: bool = Form(False),
    melody: Optional[UploadFile] = File(None),
):
    if model not in MUSIC_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    melody_data = None
    if melody is not None and melody.filename:
        contents = await melody.read()
        with tempfile.NamedTemporaryFile(suffix=Path(melody.filename).suffix) as tmp:
            tmp.write(contents)
            tmp.flush()
            data, sr = sf.read(tmp.name, always_2d=True)
            melody_data = (sr, data.T.astype(np.float32))

    job_id = manager.start_music_job(
        prompt=prompt.strip(),
        model=model,
        duration=duration,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        cfg_coef=cfg_coef,
        melody=melody_data,
        use_diffusion=use_diffusion,
    )
    return JobResponse(job_id=job_id)


@app.post("/api/generate/sound", response_model=JobResponse)
def generate_sound(body: GenerateRequest):
    if body.model not in SOUND_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {body.model}")

    job_id = manager.start_sound_job(
        prompt=body.prompt.strip(),
        model=body.model,
        duration=body.duration,
        top_k=body.top_k,
        top_p=body.top_p,
        temperature=body.temperature,
        cfg_coef=body.cfg_coef,
    )
    return JobResponse(job_id=job_id)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
