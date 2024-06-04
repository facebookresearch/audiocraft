# AudioSeal: Proactive Localized Watermarking

AudioCraft provides the training code and models for AudioSeal, a method for speech localized watermarking [Proactive Detection of Voice Cloning with Localized Watermarking][arxiv], with state-of-the-art robustness and detector speed. It jointly trains a generator that embeds a watermark in the audio, and a detector that detects the watermarked fragments in longer audios, even in the presence of editing.

## Installation and setup

To install AudioSeal, follow [Installation](https://github.com/facebookresearch/audioseal) guidelines in the AudioSeal repo. Note that currently the training of AudioSeal models requires a bleeding-edge version (`1.4.0a1`) of AudioCraft, which you obtain by installing from the source code instead of PyPi. _NOTE_: Since we use AAC augmentation in our training loop, you need to install ffmpeg, or it will not work (See Section "Installation" in [README](../README.md)).

Make sure you follow [steps for basic training setup](TRAINING.md) before starting.

## API
Check the [Github repository](https://github.com/facebookresearch/audioseal) for more details.

## Training

The [WatermarkSolver](../audiocraft/solvers/watermark.py) implements the AudioSeal's training pipeline. It joins the generator and detector that wrap
`audioseal.AudioSealWM` and `audioseal.AudioSealDetector` respectively. For the training recipe, see [config/solver/watermark/robustness.yaml](../config/solver/watermark/robustness.yaml).

For illustration, we use the three example audios in `datasets`, with datasourc definition in [dset/audio/example.yaml](../config/dset/audio/example.yaml) (Please read [DATASET](./DATASETS.md) to understand AudioCraft's dataset structure.)

To run the Watermarking training pipeline locally:

```bash
dora run solver=watermark/robustness dset=audio/example
```

you can override model / experiment parameters here directly like:

```bash
dora run solver=watermark/robustness dset=audio/example sample_rate=24000
```

If you want to run in debug mode:

```bash
python3 -m pdb -c c -m dora run solver=watermark/robustness dset=audio/example
```