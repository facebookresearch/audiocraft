# MAGNeT: Masked Audio Generation using a Single Non-Autoregressive Transformer

AudioCraft provides the code and models for MAGNeT, [Masked Audio Generation using a Single Non-Autoregressive Transformer][arxiv].

MAGNeT is a text-to-music and text-to-sound model capable of generating high-quality audio samples conditioned on text descriptions.
It is a masked generative non-autoregressive Transformer trained over a 32kHz EnCodec tokenizer with 4 codebooks sampled at 50 Hz. 
Unlike prior work on masked generative audio Transformers, such as [SoundStorm](https://arxiv.org/abs/2305.09636) and [VampNet](https://arxiv.org/abs/2307.04686), 
MAGNeT doesn't require semantic token conditioning, model cascading or audio prompting, and employs a full text-to-audio using a single non-autoregressive Transformer.

Check out our [sample page][magnet_samples] or test the available demo!

We use 16K hours of licensed music to train MAGNeT. Specifically, we rely on an internal dataset
of 10K high-quality music tracks, and on the ShutterStock and Pond5 music data.


## Model Card

See [the model card](../model_cards/MAGNET_MODEL_CARD.md).


## Installation

Please follow the AudioCraft installation instructions from the [README](../README.md).

AudioCraft requires a GPU with at least 16 GB of memory for running inference with the medium-sized models (~1.5B parameters).

## Usage

We currently offer two ways to interact with MAGNeT:
1. You can use the gradio demo locally by running [`python -m demos.magnet_app --share`](../demos/magnet_app.py).
2. You can play with MAGNeT by running the jupyter notebook at [`demos/magnet_demo.ipynb`](../demos/magnet_demo.ipynb) locally (if you have a GPU).

## API

We provide a simple API and 6 pre-trained models. The pre trained models are:
- `facebook/magnet-small-10secs`: 300M model, text to music, generates 10-second samples - [🤗 Hub](https://huggingface.co/facebook/magnet-small-10secs)
- `facebook/magnet-medium-10secs`: 1.5B model, text to music, generates 10-second samples - [🤗 Hub](https://huggingface.co/facebook/magnet-medium-10secs)
- `facebook/magnet-small-30secs`: 300M model, text to music, generates 30-second samples - [🤗 Hub](https://huggingface.co/facebook/magnet-small-30secs)
- `facebook/magnet-medium-30secs`: 1.5B model, text to music, generates 30-second samples - [🤗 Hub](https://huggingface.co/facebook/magnet-medium-30secs)
- `facebook/audio-magnet-small`: 300M model, text to sound-effect - [🤗 Hub](https://huggingface.co/facebook/audio-magnet-small)
- `facebook/audio-magnet-medium`: 1.5B model, text to sound-effect - [🤗 Hub](https://huggingface.co/facebook/audio-magnet-medium)

In order to use MAGNeT locally **you must have a GPU**. We recommend 16GB of memory, especially for 
the medium size models. 

See after a quick example for using the API.

```python
import torchaudio
from audiocraft.models import MAGNeT
from audiocraft.data.audio import audio_write

model = MAGNeT.get_pretrained('facebook/magnet-small-10secs')
descriptions = ['disco beat', 'energetic EDM', 'funky groove']
wav = model.generate(descriptions)  # generates 3 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
```

## 🤗 Transformers Usage

Coming soon...

## Training

The [MagnetSolver](../audiocraft/solvers/magnet.py) implements MAGNeT's training pipeline.
It defines a masked generation task over multiple streams of discrete tokens
extracted from a pre-trained EnCodec model (see [EnCodec documentation](./ENCODEC.md)
for more details on how to train such model).

Note that **we do NOT provide any of the datasets** used for training MAGNeT.
We provide a dummy dataset containing just a few examples for illustrative purposes.

Please read first the [TRAINING documentation](./TRAINING.md), in particular the Environment Setup section.


### Example configurations and grids

We provide configurations to reproduce the released models and our research.
MAGNeT solvers configuration are available in [config/solver/magnet](../config/solver/magnet),
in particular:
* MAGNeT model for text-to-music:
[`solver=magnet/magnet_32khz`](../config/solver/magnet/magnet_32khz.yaml)
* MAGNeT model for text-to-sound:
[`solver=magnet/audio_magnet_16khz`](../config/solver/magnet/audio_magnet_16khz.yaml)

We provide 3 different scales, e.g. `model/lm/model_scale=small` (300M), or `medium` (1.5B), and `large` (3.3B).

Please find some example grids to train MAGNeT at
[audiocraft/grids/magnet](../audiocraft/grids/magnet/).

```shell
# text-to-music
dora grid magnet.magnet_32khz --dry_run --init

# text-to-sound
dora grid magnet.audio_magnet_16khz --dry_run --init

# Remove the `--dry_run --init` flags to actually schedule the jobs once everything is setup.
```

### dataset and metadata
Learn more in the [datasets section](./DATASETS.md).

#### Music Models
MAGNeT's underlying dataset is an AudioDataset augmented with music-specific metadata.
The MAGNeT dataset implementation expects the metadata to be available as `.json` files
at the same location as the audio files. 

#### Sound Models
Audio-MAGNeT's underlying dataset is an AudioDataset augmented with description metadata.
The Audio-MAGNeT dataset implementation expects the metadata to be available as `.json` files
at the same location as the audio files or through specified external folder.

### Audio tokenizers

See [MusicGen](./MUSICGEN.md)

### Fine tuning existing models

You can initialize your model to one of the pretrained models by using the `continue_from` argument, in particular

```bash
# Using pretrained MAGNeT model.
dora run solver=magnet/magnet_32khz model/lm/model_scale=medium continue_from=//pretrained/facebook/magnet-medium-10secs conditioner=text2music

# Using another model you already trained with a Dora signature SIG.
dora run solver=magnet/magnet_32khz model/lm/model_scale=medium continue_from=//sig/SIG conditioner=text2music

# Or providing manually a path
dora run solver=magnet/magnet_32khz model/lm/model_scale=medium continue_from=/checkpoints/my_other_xp/checkpoint.th
```

**Warning:** You are responsible for selecting the other parameters accordingly, in a way that make it compatible
    with the model you are fine tuning. Configuration is NOT automatically inherited from the model you continue from. In particular make sure to select the proper `conditioner` and `model/lm/model_scale`.

**Warning:** We currently do not support fine tuning a model with slightly different layers. If you decide
 to change some parts, like the conditioning or some other parts of the model, you are responsible for manually crafting a checkpoint file from which we can safely run `load_state_dict`.
 If you decide to do so, make sure your checkpoint is saved with `torch.save` and contains a dict
    `{'best_state': {'model': model_state_dict_here}}`. Directly give the path to `continue_from` without a `//pretrained/` prefix.

### Evaluation stage
For the 6 pretrained MAGNeT models, objective metrics could be reproduced using the following grids:

```shell
# text-to-music
REGEN=1 dora grid magnet.magnet_pretrained_32khz_eval --dry_run --init

# text-to-sound
REGEN=1 dora grid magnet.audio_magnet_pretrained_16khz_eval --dry_run --init

# Remove the `--dry_run --init` flags to actually schedule the jobs once everything is setup.
```

See [MusicGen](./MUSICGEN.md) for more details. 

### Generation stage

See [MusicGen](./MUSICGEN.md)

### Playing with the model

Once you have launched some experiments, you can easily get access
to the Solver with the latest trained model using the following snippet.

```python
from audiocraft.solvers.magnet import MagnetSolver

solver = MagnetSolver.get_eval_solver_from_sig('SIG', device='cpu', batch_size=8)
solver.model
solver.dataloaders
```

### Importing / Exporting models

We do not support currently loading a model from the Hugging Face implementation or exporting to it.
If you want to export your model in a way that is compatible with `audiocraft.models.MAGNeT`
API, you can run:

```python
from audiocraft.utils import export
from audiocraft import train
xp = train.main.get_xp_from_sig('SIG_OF_LM')
export.export_lm(xp.folder / 'checkpoint.th', '/checkpoints/my_audio_lm/state_dict.bin')
# You also need to bundle the EnCodec model you used !!
## Case 1) you trained your own
xp_encodec = train.main.get_xp_from_sig('SIG_OF_ENCODEC')
export.export_encodec(xp_encodec.folder / 'checkpoint.th', '/checkpoints/my_audio_lm/compression_state_dict.bin')
## Case 2) you used a pretrained model. Give the name you used without the //pretrained/ prefix.
## This will actually not dump the actual model, simply a pointer to the right model to download.
export.export_pretrained_compression_model('facebook/encodec_32khz', '/checkpoints/my_audio_lm/compression_state_dict.bin')
```

Now you can load your custom model with:
```python
import audiocraft.models
magnet = audiocraft.models.MAGNeT.get_pretrained('/checkpoints/my_audio_lm/')
```


### Learn more

Learn more about AudioCraft training pipelines in the [dedicated section](./TRAINING.md).

## FAQ

#### What are top-k, top-p, temperature and classifier-free guidance?

Check out [@FurkanGozukara tutorial](https://github.com/FurkanGozukara/Stable-Diffusion/blob/main/Tutorials/AI-Music-Generation-Audiocraft-Tutorial.md#more-info-about-top-k-top-p-temperature-and-classifier-free-guidance-from-chatgpt).

#### Should I use FSDP or autocast ?

The two are mutually exclusive (because FSDP does autocast on its own).
You can use autocast up to 1.5B (medium), if you have enough RAM on your GPU.
FSDP makes everything more complex but will free up some memory for the actual
activations by sharding the optimizer state.

## Citation
```
@misc{ziv2024masked,
      title={Masked Audio Generation using a Single Non-Autoregressive Transformer}, 
      author={Alon Ziv and Itai Gat and Gael Le Lan and Tal Remez and Felix Kreuk and Alexandre Défossez and Jade Copet and Gabriel Synnaeve and Yossi Adi},
      year={2024},
      eprint={2401.04577},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

## License

See license information in the [model card](../model_cards/MAGNET_MODEL_CARD.md).

[arxiv]: https://arxiv.org/abs/2401.04577
[magnet_samples]: https://pages.cs.huji.ac.il/adiyoss-lab/MAGNeT/
