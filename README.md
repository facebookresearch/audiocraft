# AudioCraft
![docs badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_docs/badge.svg)
![linter badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_linter/badge.svg)
![tests badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_tests/badge.svg)

AudioCraft is a PyTorch library for deep learning research on audio generation. AudioCraft contains inference and training code
for two state-of-the-art AI generative models producing high-quality audio: AudioGen and MusicGen.


## Installation

### Using UV (Recommended)

AudioCraft now uses [UV](https://github.com/astral-sh/uv) for fast and reliable dependency management. UV is 10-100x faster than pip and provides better dependency resolution.

First, install UV:
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install AudioCraft:
```bash
# Clone the repository
git clone https://github.com/facebookresearch/audiocraft.git
cd audiocraft

# Install with UV (recommended)
uv sync                    # Install dependencies and package in development mode
uv sync --extra dev        # Include development dependencies
uv sync --extra wm         # Include watermarking dependencies
uv sync --extra all        # Include all optional dependencies

# Run commands with UV
uv run python your_script.py
uv run pytest
uv run audiocraft/train.py ...
```

### Traditional pip Installation

AudioCraft requires Python 3.8+, PyTorch 2.1.0. To install AudioCraft with pip:

```shell
# Best to make sure you have torch installed first, in particular before installing xformers.
# Don't run this if you already have PyTorch installed.
python -m pip install 'torch==2.1.0'
# You might need the following before trying to install the packages
python -m pip install setuptools wheel
# Then proceed to one of the following
python -m pip install -U audiocraft  # stable release
python -m pip install -U git+https://git@github.com/facebookresearch/audiocraft#egg=audiocraft  # bleeding edge
python -m pip install -e .  # or if you cloned the repo locally (mandatory if you want to train).
python -m pip install -e '.[wm]'  # if you want to train a watermarking model
```

We also recommend having `ffmpeg` installed, either through your system or Anaconda:
```bash
sudo apt-get install ffmpeg
# Or if you are using Anaconda or Miniconda
conda install "ffmpeg<5" -c conda-forge
```

## Development with UV

AudioCraft includes a comprehensive UV-based development setup. Use the provided Makefile for common tasks:

```bash
# Development setup
make dev-setup          # Set up development environment
make install            # Install dependencies
make dev-install        # Install with dev dependencies

# Code quality
make format             # Format code with black and isort
make lint               # Run linting (flake8, mypy)
make test               # Run tests
make check              # Run all checks (format, lint, test)

# Dependencies
make lock               # Update lock file
make upgrade            # Upgrade all dependencies
make clean              # Clean build artifacts

# Demos
make demo-musicgen      # Run MusicGen demo
make demo-audiogen      # Run AudioGen demo
make demo-magnet        # Run MAGNeT demo
```

### Adding Dependencies

With UV, adding new dependencies is simple:

```bash
# Add runtime dependency
uv add requests

# Add development dependency
uv add --dev pytest-cov

# Add optional dependency
uv add --optional watermarking audioseal

# Remove dependency
uv remove requests
```

### Running Scripts

UV provides a fast way to run scripts in isolated environments:

```bash
# Run Python scripts
uv run python scripts/your_script.py

# Run training
uv run audiocraft/train.py solvers/musicgen/musicgen_base_32khz

# Run Jupyter notebooks
uv run jupyter notebook demos/musicgen_demo.ipynb
```

## Models

At the moment, AudioCraft contains the training code and inference code for:
* [MusicGen](./docs/MUSICGEN.md): A state-of-the-art controllable text-to-music model.
* [AudioGen](./docs/AUDIOGEN.md): A state-of-the-art text-to-sound model.
* [EnCodec](./docs/ENCODEC.md): A state-of-the-art high fidelity neural audio codec.
* [Multi Band Diffusion](./docs/MBD.md): An EnCodec compatible decoder using diffusion.
* [MAGNeT](./docs/MAGNET.md): A state-of-the-art non-autoregressive model for text-to-music and text-to-sound.
* [AudioSeal](./docs/WATERMARKING.md): A state-of-the-art audio watermarking.
* [MusicGen Style](./docs/MUSICGEN_STYLE.md): A state-of-the-art text-and-style-to-music model.
* [JASCO](./docs/JASCO.md): "High quality text-to-music model conditioned on chords, melodies and drum tracks"


## Training code

AudioCraft contains PyTorch components for deep learning research in audio and training pipelines for the developed models.
For a general introduction of AudioCraft design principles and instructions to develop your own training pipeline, refer to
the [AudioCraft training documentation](./docs/TRAINING.md).

For reproducing existing work and using the developed training pipelines, refer to the instructions for each specific model
that provides pointers to configuration, example grids and model/task-specific information and FAQ.


## API documentation

We provide some [API documentation](https://facebookresearch.github.io/audiocraft/api_docs/audiocraft/index.html) for AudioCraft.


## FAQ

#### Is the training code available?

Yes! We provide the training code for [EnCodec](./docs/ENCODEC.md), [MusicGen](./docs/MUSICGEN.md),[Multi Band Diffusion](./docs/MBD.md) and [JASCO](./docs/JASCO.md).

#### Where are the models stored?

Hugging Face stored the model in a specific location, which can be overridden by setting the `AUDIOCRAFT_CACHE_DIR` environment variable for the AudioCraft models.
In order to change the cache location of the other Hugging Face models, please check out the [Hugging Face Transformers documentation for the cache setup](https://huggingface.co/docs/transformers/installation#cache-setup).
Finally, if you use a model that relies on Demucs (e.g. `musicgen-melody`) and want to change the download location for Demucs, refer to the [Torch Hub documentation](https://pytorch.org/docs/stable/hub.html#where-are-my-downloaded-models-saved).


## License
* The code in this repository is released under the MIT license as found in the [LICENSE file](LICENSE).
* The models weights in this repository are released under the CC-BY-NC 4.0 license as found in the [LICENSE_weights file](LICENSE_weights).


## Citation

For the general framework of AudioCraft, please cite the following.
```
@inproceedings{copet2023simple,
    title={Simple and Controllable Music Generation},
    author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```

When referring to a specific model, please cite as mentioned in the model specific README, e.g
[./docs/MUSICGEN.md](./docs/MUSICGEN.md), [./docs/AUDIOGEN.md](./docs/AUDIOGEN.md), etc.
