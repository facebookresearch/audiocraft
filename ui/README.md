# AudioCraft Studio

A unified web UI for [AudioCraft](https://github.com/facebookresearch/audiocraft) — generate music with **MusicGen** and environmental sounds with **AudioGen**.

## Features

- **MusicGen** — text-to-music with optional melody conditioning and MultiBand Diffusion decoder
- **AudioGen** — text-to-sound generation
- Real-time job progress polling
- Waveform visualization and WAV download
- Studio-inspired dark UI

## Requirements

- Python 3.9+
- PyTorch 2.1.0 (CUDA recommended; 16GB+ VRAM for medium models)
- AudioCraft installed from the repo root
- `ffmpeg` on your PATH

## Setup

From the repository root:

```bash
# Install AudioCraft (if not already)
python -m pip install -e .

# Install UI dependencies
python -m pip install -r ui/requirements.txt
```

## Run

```bash
python ui/run.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

Options:

```bash
python ui/run.py --host 0.0.0.0 --port 8080
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Device info and available models |
| `/api/generate/music` | POST | Start music generation (multipart form) |
| `/api/generate/sound` | POST | Start sound generation (JSON body) |
| `/api/jobs/{id}` | GET | Poll generation status |
| `/api/audio/{filename}` | GET | Download generated WAV |

## Notes

- First generation downloads model weights from Hugging Face (can take several minutes).
- CPU inference works but is very slow; a GPU is strongly recommended.
- Generated files are stored in `ui/outputs/` and cleaned up manually.

## Existing Gradio demos

The repo also includes original Gradio demos:

```bash
python demos/musicgen_app.py
python demos/magnet_app.py
```
