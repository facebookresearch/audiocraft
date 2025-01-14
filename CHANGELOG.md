# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.4.0a2] - 2025-01-14

Add training and inference code for JASCO (https://arxiv.org/abs/2406.10970) along with the [hf checkpoints](https://huggingface.co/facebook/jasco-chords-drums-melody-1B).

## [1.4.0a1] - 2024-06-03

Adding new metric PesqMetric ([Perceptual Evaluation of Speech Quality](https://doi.org/10.5281/zenodo.6549559))

Adding multiple audio augmentation functions: generating pink noises, up-/downsampling, low-/highpass filtering, banpass filtering, smoothing, duck masking, boosting. All are wrapped in the `audiocraft.utils.audio_effects.AudioEffects` and can be called with the API `audiocraft.utils.audio_effects.select_audio_effects`.

Add training code for AudioSeal (https://arxiv.org/abs/2401.17264) along with the [hf checkpoints]( https://huggingface.co/facebook/audioseal).

## [1.3.0] - 2024-05-02

Adding the MAGNeT model (https://arxiv.org/abs/2401.04577) along with hf checkpoints and a gradio demo app.

Typo fixes.

Fixing setup.py to install only audiocraft, not the unit tests and scripts.

Fix FSDP support with PyTorch 2.1.0. 

## [1.2.0] - 2024-01-11

Adding stereo models.

Fixed the commitment loss, which was until now only applied to the first RVQ layer.

Removed compression model state from the LM checkpoints, for consistency, it
should always be loaded from the original `compression_model_checkpoint`.


## [1.1.0] - 2023-11-06

Not using torchaudio anymore when writing audio files, relying instead directly on the commandline ffmpeg. Also not using it anymore for reading audio files, for similar reasons.

Fixed DAC support with non default number of codebooks.

Fixed bug when `two_step_cfg` was overriden when calling `generate()`.

Fixed samples being always prompted with audio, rather than having both prompted and unprompted.

**Backward incompatible change:** A `torch.no_grad` around the computation of the conditioning made its way in the public release.
	The released models were trained without this. Those impact linear layers applied to the output of the T5 or melody conditioners.
	We removed it, so you might need to retrain models.

**Backward incompatible change:** Fixing wrong sample rate in CLAP (WARNING if you trained model with CLAP before).

**Backward incompatible change:** Renamed VALLEPattern to CoarseFirstPattern, as it was wrongly named. Probably no one
	retrained a model with this pattern, so hopefully this won't impact you!


## [1.0.0] - 2023-09-07

Major revision, added training code for EnCodec, AudioGen, MusicGen, and MultiBandDiffusion.
Added pretrained model for AudioGen and MultiBandDiffusion.

## [0.0.2] - 2023-08-01

Improved demo, fixed top p (thanks @jnordberg).

Compressor tanh on output to avoid clipping with some style (especially piano).
Now repeating the conditioning periodically if it is too short.

More options when launching Gradio app locally (thanks @ashleykleynhans).

Testing out PyTorch 2.0 memory efficient attention.

Added extended generation (infinite length) by slowly moving the windows.
Note that other implementations exist: https://github.com/camenduru/MusicGen-colab.

## [0.0.1] - 2023-06-09

Initial release, with model evaluation only.
