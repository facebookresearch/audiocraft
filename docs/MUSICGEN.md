# MusicGen: Simple and Controllable Music Generation

AudioCraft provides the code and models for MusicGen, [a simple and controllable model for music generation][arxiv].
MusicGen is a single stage auto-regressive Transformer model trained over a 32kHz
<a href="https://github.com/facebookresearch/encodec">EnCodec tokenizer</a> with 4 codebooks sampled at 50 Hz.
Unlike existing methods like [MusicLM](https://arxiv.org/abs/2301.11325), MusicGen doesn't require
a self-supervised semantic representation, and it generates all 4 codebooks in one pass. By introducing
a small delay between the codebooks, we show we can predict them in parallel, thus having only 50 auto-regressive
steps per second of audio.
Check out our [sample page][musicgen_samples] or test the available demo!

<a target="_blank" href="https://colab.research.google.com/drive/1JlTOjB-G0A2Hz3h8PK63vLZk4xdCI5QB?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<a target="_blank" href="https://huggingface.co/spaces/facebook/MusicGen">
  <img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg" alt="Open in HugginFace"/>
</a>
<br>

We use 20K hours of licensed music to train MusicGen. Specifically, we rely on an internal dataset
of 10K high-quality music tracks, and on the ShutterStock and Pond5 music data.


## Model Card

See [the model card](../model_cards/MUSICGEN_MODEL_CARD.md).


## Installation

Please follow the AudioCraft installation instructions from the [README](../README.md).

AudioCraft requires a GPU with at least 16 GB of memory for running inference with the medium-sized models (~1.5B parameters).

## Usage

We offer a number of way to interact with MusicGen:
1. A demo is also available on the [`facebook/MusicGen` Hugging Face Space](https://huggingface.co/spaces/facebook/MusicGen)
(huge thanks to all the HF team for their support).
2. You can run the extended demo on a Colab:
[colab notebook](https://colab.research.google.com/drive/1JlTOjB-G0A2Hz3h8PK63vLZk4xdCI5QB?usp=sharing)
3. You can use the gradio demo locally by running [`python -m demos.musicgen_app --share`](../demos/musicgen_app.py).
4. You can play with MusicGen by running the jupyter notebook at [`demos/musicgen_demo.ipynb`](../demos/musicgen_demo.ipynb) locally (if you have a GPU).
5. Finally, checkout [@camenduru Colab page](https://github.com/camenduru/MusicGen-colab)
which is regularly updated with contributions from @camenduru and the community.


## API

We provide a simple API and 4 pre-trained models. The pre trained models are:
- `facebook/musicgen-small`: 300M model, text to music only - [🤗 Hub](https://huggingface.co/facebook/musicgen-small)
- `facebook/musicgen-medium`: 1.5B model, text to music only - [🤗 Hub](https://huggingface.co/facebook/musicgen-medium)
- `facebook/musicgen-melody`: 1.5B model, text to music and text+melody to music - [🤗 Hub](https://huggingface.co/facebook/musicgen-melody)
- `facebook/musicgen-large`: 3.3B model, text to music only - [🤗 Hub](https://huggingface.co/facebook/musicgen-large)

We observe the best trade-off between quality and compute with the `facebook/musicgen-medium` or `facebook/musicgen-melody` model.
In order to use MusicGen locally **you must have a GPU**. We recommend 16GB of memory, but smaller
GPUs will be able to generate short sequences, or longer sequences with the `facebook/musicgen-small` model.

See after a quick example for using the API.

```python
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=8)  # generate 8 seconds.
wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
wav = model.generate(descriptions)  # generates 3 samples.

melody, sr = torchaudio.load('./assets/bach.mp3')
# generates using the melody from the given audio and the provided descriptions.
wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
```

## 🤗 Transformers Usage

MusicGen is available in the 🤗 Transformers library from version 4.31.0 onwards, requiring minimal dependencies
and additional packages. Steps to get started:

1. First install the 🤗 [Transformers library](https://github.com/huggingface/transformers) from main:

```shell
pip install git+https://github.com/huggingface/transformers.git
```

2. Run the following Python code to generate text-conditional audio samples:

```py
from transformers import AutoProcessor, MusicgenForConditionalGeneration


processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

inputs = processor(
    text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
    padding=True,
    return_tensors="pt",
)

audio_values = model.generate(**inputs, max_new_tokens=256)
```

3. Listen to the audio samples either in an ipynb notebook:

```py
from IPython.display import Audio

sampling_rate = model.config.audio_encoder.sampling_rate
Audio(audio_values[0].numpy(), rate=sampling_rate)
```

Or save them as a `.wav` file using a third-party library, e.g. `scipy`:

```py
import scipy

sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())
```

For more details on using the MusicGen model for inference using the 🤗 Transformers library, refer to the
[MusicGen docs](https://huggingface.co/docs/transformers/main/en/model_doc/musicgen) or the hands-on
[Google Colab](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/MusicGen.ipynb).


## Training

The [MusicGenSolver](../audiocraft/solvers/musicgen.py) implements MusicGen's training pipeline.
It defines an autoregressive language modeling task over multiple streams of discrete tokens
extracted from a pre-trained EnCodec model (see [EnCodec documentation](./ENCODEC.md)
for more details on how to train such model).

Note that **we do NOT provide any of the datasets** used for training MusicGen.
We provide a dummy dataset containing just a few examples for illustrative purposes.

Please read first the [TRAINING documentation](./TRAINING.md), in particular the Environment Setup section.


**Warning:** As of version 1.1.0, a few breaking changes were introduced. Check the [CHANGELOG.md](../CHANGELOG.md)
file for more information. You might need to retrain some of your models.

### Example configurations and grids

We provide configurations to reproduce the released models and our research.
MusicGen solvers configuration are available in [config/solver/musicgen](../config/solver/musicgen),
in particular:
* MusicGen base model for text-to-music:
[`solver=musicgen/musicgen_base_32khz`](../config/solver/musicgen/musicgen_base_32khz.yaml)
* MusicGen model with chromagram-conditioning support:
[`solver=musicgen/musicgen_melody_32khz`](../config/solver/musicgen/musicgen_melody_32khz.yaml)

We provide 3 different scales, e.g. `model/lm/model_scale=small` (300M), or `medium` (1.5B), and `large` (3.3B).

Please find some example grids to train MusicGen at
[audiocraft/grids/musicgen](../audiocraft/grids/musicgen/).

```shell
# text-to-music
dora grid musicgen.musicgen_base_32khz --dry_run --init
# melody-guided music generation
dora grid musicgen.musicgen_melody_base_32khz --dry_run --init
# Remove the `--dry_run --init` flags to actually schedule the jobs once everything is setup.
```

### Music dataset and metadata

MusicGen's underlying dataset is an AudioDataset augmented with music-specific metadata.
The MusicGen dataset implementation expects the metadata to be available as `.json` files
at the same location as the audio files. Learn more in the [datasets section](./DATASETS.md).


### Audio tokenizers

We support a number of audio tokenizers: either pretrained EnCodec models, [DAC](https://github.com/descriptinc/descript-audio-codec), or your own models.
The tokenizer is controlled with the setting `compression_model_checkpoint`.
For instance,

```bash
# Using the 32kHz EnCodec trained on music
dora run solver=musicgen/debug \
    compression_model_checkpoint=//pretrained/facebook/encodec_32khz \
    transformer_lm.n_q=4 transformer_lm.card=2048

# Using DAC
dora run solver=musicgen/debug \
    compression_model_checkpoint=//pretrained/dac_44khz \
    transformer_lm.n_q=9 transformer_lm.card=1024 \
    'codebooks_pattern.delay.delays=[0,1,2,3,4,5,6,7,8]'

# Using your own model after export (see ENCODEC.md)
dora run solver=musicgen/debug \
    compression_model_checkpoint=//pretrained//checkpoints/my_audio_lm/compression_state_dict.bin \
    transformer_lm.n_q=... transformer_lm.card=...

# Using your own model from its training checkpoint.
dora run solver=musicgen/debug \
    compression_model_checkpoint=//sig/SIG \ # where SIG is the Dora signature of the EnCodec XP.
    transformer_lm.n_q=... transformer_lm.card=...
```

**Warning:** you are responsible for setting the proper value for `transformer_lm.n_q` and `transformer_lm.card` (cardinality of the codebooks). You also have to update the codebook_pattern to match `n_q` as shown in the example for using DAC. .


### Fine tuning existing models

You can initialize your model to one of the pretrained models by using the `continue_from` argument, in particular

```bash
# Using pretrained MusicGen model.
dora run solver=musicgen/musicgen_base_32khz model/lm/model_scale=medium continue_from=//pretrained/facebook/musicgen-medium conditioner=text2music

# Using another model you already trained with a Dora signature SIG.
dora run solver=musicgen/musicgen_base_32khz model/lm/model_scale=medium continue_from=//sig/SIG conditioner=text2music

# Or providing manually a path
dora run solver=musicgen/musicgen_base_32khz model/lm/model_scale=medium continue_from=/checkpoints/my_other_xp/checkpoint.th
```

**Warning:** You are responsible for selecting the other parameters accordingly, in a way that make it compatible
    with the model you are fine tuning. Configuration is NOT automatically inherited from the model you continue from. In particular make sure to select the proper `conditioner` and `model/lm/model_scale`.

**Warning:** We currently do not support fine tuning a model with slightly different layers. If you decide
 to change some parts, like the conditioning or some other parts of the model, you are responsible for manually crafting a checkpoint file from which we can safely run `load_state_dict`.
 If you decide to do so, make sure your checkpoint is saved with `torch.save` and contains a dict
    `{'best_state': {'model': model_state_dict_here}}`. Directly give the path to `continue_from` without a `//pretrained/` prefix.

### Caching of EnCodec tokens

It is possible to precompute the EnCodec tokens and other metadata.
An example of generating and using this cache provided in the [musicgen.musicgen_base_cached_32khz grid](../audiocraft/grids/musicgen/musicgen_base_cached_32khz.py).

### Evaluation stage

By default, evaluation stage is also computing the cross-entropy and the perplexity over the
evaluation dataset. Indeed the objective metrics used for evaluation can be costly to run
or require some extra dependencies. Please refer to the [metrics documentation](./METRICS.md)
for more details on the requirements for each metric.

We provide an off-the-shelf configuration to enable running the objective metrics
for audio generation in
[config/solver/musicgen/evaluation/objective_eval](../config/solver/musicgen/evaluation/objective_eval.yaml).

One can then activate evaluation the following way:
```shell
# using the configuration
dora run solver=musicgen/debug solver/musicgen/evaluation=objective_eval
# specifying each of the fields, e.g. to activate KL computation
dora run solver=musicgen/debug evaluate.metrics.kld=true
```

See [an example evaluation grid](../audiocraft/grids/musicgen/musicgen_pretrained_32khz_eval.py).

### Generation stage

The generation stage allows to generate samples conditionally and/or unconditionally and to perform
audio continuation (from a prompt). We currently support greedy sampling (argmax), sampling
from softmax with a given temperature, top-K and top-P (nucleus) sampling. The number of samples
generated and the batch size used are controlled by the `dataset.generate` configuration
while the other generation parameters are defined in `generate.lm`.

```shell
# control sampling parameters
dora run solver=musicgen/debug generate.lm.gen_duration=10 generate.lm.use_sampling=true generate.lm.top_k=15
```

#### Listening to samples

Note that generation happens automatically every 25 epochs. You can easily access and
compare samples between models (as long as they are trained) on the same dataset using the
MOS tool. For that first `pip install Flask gunicorn`. Then
```
gunicorn -w 4 -b 127.0.0.1:8895 -t 120 'scripts.mos:app'  --access-logfile -
```
And access the tool at [https://127.0.0.1:8895](https://127.0.0.1:8895).

### Playing with the model

Once you have launched some experiments, you can easily get access
to the Solver with the latest trained model using the following snippet.

```python
from audiocraft.solvers.musicgen import MusicGen

solver = MusicGen.get_eval_solver_from_sig('SIG', device='cpu', batch_size=8)
solver.model
solver.dataloaders
```

### Importing / Exporting models

We do not support currently loading a model from the Hugging Face implementation or exporting to it.
If you want to export your model in a way that is compatible with `audiocraft.models.MusicGen`
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
musicgen = audiocraft.models.MusicGen.get_pretrained('/checkpoints/my_audio_lm/')
```


### Learn more

Learn more about AudioCraft training pipelines in the [dedicated section](./TRAINING.md).

## FAQ

#### I need help on Windows

@FurkanGozukara made a complete tutorial for [AudioCraft/MusicGen on Windows](https://youtu.be/v-YpvPkhdO4)

#### I need help for running the demo on Colab

Check [@camenduru tutorial on YouTube](https://www.youtube.com/watch?v=EGfxuTy9Eeo).

#### What are top-k, top-p, temperature and classifier-free guidance?

Check out [@FurkanGozukara tutorial](https://github.com/FurkanGozukara/Stable-Diffusion/blob/main/Tutorials/AI-Music-Generation-Audiocraft-Tutorial.md#more-info-about-top-k-top-p-temperature-and-classifier-free-guidance-from-chatgpt).

#### Should I use FSDP or autocast ?

The two are mutually exclusive (because FSDP does autocast on its own).
You can use autocast up to 1.5B (medium), if you have enough RAM on your GPU.
FSDP makes everything more complex but will free up some memory for the actual
activations by sharding the optimizer state.

## Citation
```
@article{copet2023simple,
    title={Simple and Controllable Music Generation},
    author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
    year={2023},
    journal={arXiv preprint arXiv:2306.05284},
}
```


## License

See license information in the [model card](../model_cards/MUSICGEN_MODEL_CARD.md).


[arxiv]: https://arxiv.org/abs/2306.05284
[musicgen_samples]: https://ai.honu.io/papers/musicgen/
