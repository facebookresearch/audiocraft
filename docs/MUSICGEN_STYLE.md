# MusicGen-Style: Audio Conditioning for Music Generation via Discrete Bottleneck Features

AudioCraft provides the code and models for MusicGen-Style, [Audio Conditioning for Music Generation via Discrete Bottleneck Features][arxiv].

MusicGen-Style is a text-and-audio-to-music model that can be conditioned on textual and audio data (thanks to a style conditioner). 
The style conditioner takes as input a music excerpt of a few seconds (between 1.5 and 4.5) extracts some features that are used by the model to generate music in the same style. 
This style conditioning can be mixed with textual description. 

Check out our [sample page][musicgen_style_samples] or test the available demo!

We use 16K hours of licensed music to train MusicGen-Style. Specifically, we rely on an internal dataset
of 10K high-quality music tracks, and on the ShutterStock and Pond5 music data.


## Model Card

See [the model card](../model_cards/MUSICGEN_STYLE_MODEL_CARD.md).


## Installation

Please follow the AudioCraft installation instructions from the [README](../README.md).

MusicGen-Stem requires a GPU with at least 16 GB of memory for running inference with the medium-sized models (~1.5B parameters).

## Usage

1. You can play with MusicGen-Style by running the jupyter notebook at [`demos/musicgen_style_demo.ipynb`](../demos/musicgen_style_demo.ipynb) locally (if you have a GPU).
2. You can use the gradio demo locally by running python -m demos.musicgen_style_app --share.
3. You can play with MusicGen by running the jupyter notebook at demos/musicgen_style_demo.ipynb locally (if you have a GPU).

## API

We provide a simple API 1 pre-trained model with MERT used as a feature extractor for the style conditioner:
- `facebook/musicgen-style`: medium (1.5B) MusicGen model, text and style to music, generates 30-second samples - [🤗 Hub](https://huggingface.co/facebook/musicgen-style)

In order to use MusicGen-Style locally **you must have a GPU**. We recommend 16GB of memory. 

See after a quick example for using the API.

To perform text-to-music:
```python
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-style')


model.set_generation_params(
    duration=8, # generate 8 seconds, can go up to 30
    use_sampling=True, 
    top_k=250,
    cfg_coef=3., # Classifier Free Guidance coefficient 
    cfg_coef_beta=None, # double CFG is only useful for text-and-style conditioning
)  

descriptions = ['disco beat', 'energetic EDM', 'funky groove']
wav = model.generate(descriptions)  # generates 3 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
```

To perform style-to-music:
```python
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-style')


model.set_generation_params(
    duration=8, # generate 8 seconds, can go up to 30
    use_sampling=True, 
    top_k=250,
    cfg_coef=3., # Classifier Free Guidance coefficient 
    cfg_coef_beta=None, # double CFG is only useful for text-and-style conditioning
)

model.set_style_conditioner_params(
    eval_q=1, # integer between 1 and 6
              # eval_q is the level of quantization that passes
              # through the conditioner. When low, the models adheres less to the 
              # audio conditioning
    excerpt_length=3., # the length in seconds that is taken by the model in the provided excerpt
    )

melody, sr = torchaudio.load('./assets/electronic.mp3')


wav = model.generate_with_chroma(descriptions=[None, None, None], 
                melody[None].expand(3, -1, -1), sr)  # generates 3 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
```

To perform style-and-text-to-music:
```python
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-style')


model.set_generation_params(
    duration=8, # generate 8 seconds, can go up to 30
    use_sampling=True, 
    top_k=250,
    cfg_coef=3., # Classifier Free Guidance coefficient 
    cfg_coef_beta=5., # double CFG is necessary for text-and-style conditioning
                   # Beta in the double CFG formula. between 1 and 9. When set to 1 it is equivalent to normal CFG. 
                   # When we increase this parameter, the text condition is pushed. See the bottom of https://musicgenstyle.github.io/ 
                   # to better understand the effects of the double CFG coefficients. 
)

model.set_style_conditioner_params(
    eval_q=1, # integer between 1 and 6
              # eval_q is the level of quantization that passes
              # through the conditioner. When low, the models adheres less to the 
              # audio conditioning
    excerpt_length=3., # the length in seconds that is taken by the model in the provided excerpt, can be                 
                       # between 1.5 and 4.5 seconds but it has to be shortest to the length of the provided conditioning
    )

melody, sr = torchaudio.load('./assets/electronic.mp3')

descriptions = ["8-bit old video game music", "Chill lofi remix", "80s New wave with synthesizer"]
wav = model.generate_with_chroma(descriptions=["8-bit old video game music"], 
                melody[None].expand(3, -1, -1), sr)  # generates 3 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
```


## Training
To train MusicGen-Style, we use the [MusicGenSolver](../audiocraft/solvers/musicgen.py).

Note that **we do NOT provide any of the datasets** used for training MusicGen-Style.
We provide a dummy dataset containing just a few examples for illustrative purposes.

Please read first the [TRAINING documentation](./TRAINING.md), in particular the Environment Setup section.


### Example configurations and grids

We provide the configuration to reproduce the training of MusicGen-Style in [config/solver/musicgen/musicgen_style_32khz.yaml](../config/solver/musicgen/musicgen_style_32khz.yaml),

In particular, the conditioner configuration is provided in [/config/conditioner/style2music.yaml](../config/conditioner/style2music.yaml).

The grid to train the model is 
[audiocraft/grids/musicgen/musicgen_style_32khz.py](../audiocraft/grids/musicgen/musicgen_style_32khz.py).

```shell
# text-and-style-to-music
dora grid musicgen.musicgen_style_32khz --dry_run --init

# Remove the `--dry_run --init` flags to actually schedule the jobs once everything is setup.
```

### dataset and metadata
Learn more in the [datasets section](./DATASETS.md).

### Audio tokenizers

See [MusicGen](./MUSICGEN.md)

### Fine tuning existing models

You can initialize your model to one of the pretrained models by using the `continue_from` argument, in particular

```bash
# Using pretrained MusicGen-Style model.
dora run solver=musicgen/musicgen_style_32khz model/lm/model_scale=medium continue_from=//pretrained/facebook/musicgen-style conditioner=style2music

# Using another model you already trained with a Dora signature SIG.
dora run solver=musicgen/musicgen_style_32khz model/lm/model_scale=medium continue_from=//sig/SIG conditioner=style2music

# Or providing manually a path
dora run solver=musicgen/musicgen_style_32khz model/lm/model_scale=medium continue_from=/checkpoints/my_other_xp/checkpoint.th
```

**Warning:** You are responsible for selecting the other parameters accordingly, in a way that make it compatible
    with the model you are fine tuning. Configuration is NOT automatically inherited from the model you continue from. In particular make sure to select the proper `conditioner` and `model/lm/model_scale`.

**Warning:** We currently do not support fine tuning a model with slightly different layers. If you decide
 to change some parts, like the conditioning or some other parts of the model, you are responsible for manually crafting a checkpoint file from which we can safely run `load_state_dict`.
 If you decide to do so, make sure your checkpoint is saved with `torch.save` and contains a dict
    `{'best_state': {'model': model_state_dict_here}}`. Directly give the path to `continue_from` without a `//pretrained/` prefix.


[arxiv]: https://arxiv.org/abs/2407.12563
[musicgen_samples]: https://musicgenstyle.github.io/
