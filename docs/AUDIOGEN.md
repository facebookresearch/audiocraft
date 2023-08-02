# AudioGen: Textually-guided audio generation

AudioCraft provides the code and a model re-implementing AudioGen, a [textually-guided audio generation][audiogen_arxiv]
model that performs text-to-sound generation.

The provided AudioGen reimplementation follows the LM model architecture introduced in [MusicGen][musicgen_arxiv]
and is a single stage auto-regressive Transformer model trained over a 16kHz
<a href="https://github.com/facebookresearch/encodec">EnCodec tokenizer</a> with 4 codebooks sampled at 50 Hz.
This model variant reaches similar audio quality than the original implementation introduced in the AudioGen publication
while providing faster generation speed given the smaller frame rate.

**Important note:** The provided models are NOT the original models used to report numbers in the
[AudioGen publication][audiogen_arxiv]. Refer to the model card to learn more about architectural changes.

Listen to samples from the **original AudioGen implementation** in our [sample page][audiogen_samples].


## Model Card

See [the model card](../model_cards/AUDIOGEN_MODEL_CARD.md).


## Installation

Please follow the AudioCraft installation instructions from the [README](../README.md).

AudioCraft requires a GPU with at least 16 GB of memory for running inference with the medium-sized models (~1.5B parameters).

## API and usage

We provide a simple API and 1 pre-trained models for AudioGen:

`facebook/audiogen-medium`: 1.5B model, text to sound - [🤗 Hub](https://huggingface.co/facebook/audiogen-medium)

You can play with AudioGen by running the jupyter notebook at [`demos/audiogen_demo.ipynb`](../demos/audiogen_demo.ipynb) locally (if you have a GPU).

See after a quick example for using the API.

```python
import torchaudio
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write

model = AudioGen.get_pretrained('facebook/audiogen-medium')
model.set_generation_params(duration=5)  # generate 5 seconds.
descriptions = ['dog barking', 'sirene of an emergency vehicle', 'footsteps in a corridor']
wav = model.generate(descriptions)  # generates 3 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
```

## Training

The [AudioGenSolver](../audiocraft/solvers/audiogen.py) implements the AudioGen's training pipeline
used to develop the released model. Note that this may not fully reproduce the results presented in the paper.
Similarly to MusicGen, it defines an autoregressive language modeling task over multiple streams of
discrete tokens extracted from a pre-trained EnCodec model (see [EnCodec documentation](./ENCODEC.md)
for more details on how to train such model) with dataset-specific changes for environmental sound
processing.

Note that **we do NOT provide any of the datasets** used for training AudioGen.

### Example configurations and grids

We provide configurations to reproduce the released models and our research.
AudioGen solvers configuration are available in [config/solver/audiogen](../config/solver/audiogen).
The base training configuration used for the released models is the following:
[`solver=audiogen/audiogen_base_16khz`](../config/solver/audiogen/audiogen_base_16khz.yaml)

Please find some example grids to train AudioGen at
[audiocraft/grids/audiogen](../audiocraft/grids/audiogen/).

```shell
# text-to-sound
dora grid audiogen.audiogen_base_16khz
```

### Sound dataset and metadata

AudioGen's underlying dataset is an AudioDataset augmented with description metadata.
The AudioGen dataset implementation expects the metadata to be available as `.json` files
at the same location as the audio files or through specified external folder.
Learn more in the [datasets section](./DATASETS.md).

### Evaluation stage

By default, evaluation stage is also computing the cross-entropy and the perplexity over the
evaluation dataset. Indeed the objective metrics used for evaluation can be costly to run
or require some extra dependencies. Please refer to the [metrics documentation](./METRICS.md)
for more details on the requirements for each metric.

We provide an off-the-shelf configuration to enable running the objective metrics
for audio generation in
[config/solver/audiogen/evaluation/objective_eval](../config/solver/audiogen/evaluation/objective_eval.yaml).

One can then activate evaluation the following way:
```shell
# using the configuration
dora run solver=audiogen/debug solver/audiogen/evaluation=objective_eval
# specifying each of the fields, e.g. to activate KL computation
dora run solver=audiogen/debug evaluate.metrics.kld=true
```

See [an example evaluation grid](../audiocraft/grids/audiogen/audiogen_pretrained_16khz_eval.py).

### Generation stage

The generation stage allows to generate samples conditionally and/or unconditionally and to perform
audio continuation (from a prompt). We currently support greedy sampling (argmax), sampling
from softmax with a given temperature, top-K and top-P (nucleus) sampling. The number of samples
generated and the batch size used are controlled by the `dataset.generate` configuration
while the other generation parameters are defined in `generate.lm`.

```shell
# control sampling parameters
dora run solver=audiogen/debug generate.lm.gen_duration=5 generate.lm.use_sampling=true generate.lm.top_k=15
```

## More information

Refer to [MusicGen's instructions](./MUSICGEN.md).

### Learn more

Learn more about AudioCraft training pipelines in the [dedicated section](./TRAINING.md).


## Citation

AudioGen
```
@article{kreuk2022audiogen,
    title={Audiogen: Textually guided audio generation},
    author={Kreuk, Felix and Synnaeve, Gabriel and Polyak, Adam and Singer, Uriel and D{\'e}fossez, Alexandre and Copet, Jade and Parikh, Devi and Taigman, Yaniv and Adi, Yossi},
    journal={arXiv preprint arXiv:2209.15352},
    year={2022}
}
```

MusicGen
```
@article{copet2023simple,
    title={Simple and Controllable Music Generation},
    author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
    year={2023},
    journal={arXiv preprint arXiv:2306.05284},
}
```

## License

See license information in the [model card](../model_cards/AUDIOGEN_MODEL_CARD.md).

[audiogen_arxiv]: https://arxiv.org/abs/2209.15352
[musicgen_arxiv]: https://arxiv.org/abs/2306.05284
[audiogen_samples]: https://felixkreuk.github.io/audiogen/
