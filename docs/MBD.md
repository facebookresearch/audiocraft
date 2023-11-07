# MultiBand Diffusion

AudioCraft provides the code and models for MultiBand Diffusion, [From Discrete Tokens to High Fidelity Audio using MultiBand Diffusion][arxiv].
MultiBand diffusion is a collection of 4 models that can decode tokens from
<a href="https://github.com/facebookresearch/encodec">EnCodec tokenizer</a> into waveform audio. You can listen to some examples on the <a href="https://ai.honu.io/papers/mbd/">sample page</a>.

<a target="_blank" href="https://colab.research.google.com/drive/1JlTOjB-G0A2Hz3h8PK63vLZk4xdCI5QB?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<br>


## Installation

Please follow the AudioCraft installation instructions from the [README](../README.md).


## Usage

We offer a number of way to use MultiBand Diffusion:
1. The MusicGen demo includes a toggle to try diffusion decoder. You can use the demo locally by running [`python -m demos.musicgen_app --share`](../demos/musicgen_app.py), or through the [MusicGen Colab](https://colab.research.google.com/drive/1JlTOjB-G0A2Hz3h8PK63vLZk4xdCI5QB?usp=sharing).
2. You can play with MusicGen by running the jupyter notebook at [`demos/musicgen_demo.ipynb`](../demos/musicgen_demo.ipynb) locally (if you have a GPU).

## API

We provide a simple API and pre-trained models for MusicGen and for EnCodec at 24 khz for 3 bitrates (1.5 kbps, 3 kbps and 6 kbps).

See after a quick example for using MultiBandDiffusion with the MusicGen API:

```python
import torchaudio
from audiocraft.models import MusicGen, MultiBandDiffusion
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-melody')
mbd = MultiBandDiffusion.get_mbd_musicgen()
model.set_generation_params(duration=8)  # generate 8 seconds.
wav, tokens = model.generate_unconditional(4, return_tokens=True)    # generates 4 unconditional audio samples and keep the tokens for MBD generation
descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
wav_diffusion = mbd.tokens_to_wav(tokens)
wav, tokens = model.generate(descriptions, return_tokens=True)  # generates 3 samples and keep the tokens.
wav_diffusion = mbd.tokens_to_wav(tokens)
melody, sr = torchaudio.load('./assets/bach.mp3')
# Generates using the melody from the given audio and the provided descriptions, returns audio and audio tokens.
wav, tokens = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr, return_tokens=True)
wav_diffusion = mbd.tokens_to_wav(tokens)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav and {idx}_diffusion.wav, with loudness normalization at -14 db LUFS for comparing the methods.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
    audio_write(f'{idx}_diffusion', wav_diffusion[idx].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
```

For the compression task (and to compare with [EnCodec](https://github.com/facebookresearch/encodec)):

```python
import torch
from audiocraft.models import MultiBandDiffusion
from encodec import EncodecModel
from audiocraft.data.audio import audio_read, audio_write

bandwidth = 3.0  # 1.5, 3.0, 6.0
mbd = MultiBandDiffusion.get_mbd_24khz(bw=bandwidth)
encodec = EncodecModel.encodec_model_24khz()

somepath = ''
wav, sr = audio_read(somepath)
with torch.no_grad():
    compressed_encodec = encodec(wav)
    compressed_diffusion = mbd.regenerate(wav, sample_rate=sr)

audio_write('sample_encodec', compressed_encodec.squeeze(0).cpu(), mbd.sample_rate, strategy="loudness", loudness_compressor=True)
audio_write('sample_diffusion', compressed_diffusion.squeeze(0).cpu(), mbd.sample_rate, strategy="loudness", loudness_compressor=True)
```


## Training

The [DiffusionSolver](../audiocraft/solvers/diffusion.py) implements our diffusion training pipeline.
It generates waveform audio conditioned on the embeddings extracted from a pre-trained EnCodec model
(see [EnCodec documentation](./ENCODEC.md) for more details on how to train such model).

Note that **we do NOT provide any of the datasets** used for training our diffusion models.
We provide a dummy dataset containing just a few examples for illustrative purposes.

### Example configurations and grids

One can train diffusion models as described in the paper by using this [dora grid](../audiocraft/grids/diffusion/4_bands_base_32khz.py).
```shell
# 4 bands MBD trainning
dora grid diffusion.4_bands_base_32khz
```

### Learn more

Learn more about AudioCraft training pipelines in the [dedicated section](./TRAINING.md).


## Citation

```
@article{sanroman2023fromdi,
  title={From Discrete Tokens to High-Fidelity Audio Using Multi-Band Diffusion},
  author={San Roman, Robin and Adi, Yossi and Deleforge, Antoine and Serizel, Romain and Synnaeve, Gabriel and Défossez, Alexandre},
  journal={arXiv preprint arXiv:},
  year={2023}
}
```


## License

See license information in the [README](../README.md).


[arxiv]: https://arxiv.org/abs/2308.02560
[mbd_samples]: https://ai.honu.io/papers/mbd/
