# MAGNeT Model Card

## Model details

**Organization developing the model:** The FAIR team of Meta AI.

**Model date:** MAGNeT was trained between November 2023 and January 2024.

**Model version:** This is the version 1 of the model.

**Model type:** MAGNeT consists of an EnCodec model for audio tokenization, and a non-autoregressive model based on the transformer architecture for music modeling. The model comes in different sizes: 300M and 1.5B; and two variants: a model trained for text-to-music generation, and a model trained for text-to-sound generation.
 
**Paper or resources for more information:** More information can be found in the paper [Masked Audio Generation using a Single Non-Autoregressive Transformer][arxiv].

**Citation details:** See [our paper][arxiv]

**License:** Code is released under MIT, model weights are released under CC-BY-NC 4.0.

**Where to send questions or comments about the model:** Questions and comments about MAGNeT can be sent via the [GitHub repository](https://github.com/facebookresearch/audiocraft) of the project, or by opening an issue.

## Intended use
**Primary intended use:** The primary use of MAGNeT is research on AI-based music generation, including:

- Research efforts, such as probing and better understanding the limitations of generative models to further improve the state of science
- Generation of music guided by text to understand current abilities of generative AI models by machine learning amateurs

**Primary intended users:** The primary intended users of the model are researchers in audio, machine learning and artificial intelligence, as well as amateur seeking to better understand those models.

**Out-of-scope use cases:** The model should not be used on downstream applications without further risk evaluation and mitigation. The model should not be used to intentionally create or disseminate music pieces that create hostile or alienating environments for people. This includes generating music that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.

## Metrics

**Models performance measures:** We used the following objective measure to evaluate the model on a standard music benchmark:

- Frechet Audio Distance computed on features extracted from a pre-trained audio classifier (VGGish)
- Kullback-Leibler Divergence on label distributions extracted from a pre-trained audio classifier (PaSST)
- CLAP Score between audio embedding and text embedding extracted from a pre-trained CLAP model

Additionally, we run qualitative studies with human participants, evaluating the performance of the model with the following axes:

- Overall quality of the music samples;
- Text relevance to the provided text input;

More details on performance measures and human studies can be found in the paper.

**Decision thresholds:** Not applicable.

## Evaluation datasets

The model was evaluated on the [MusicCaps benchmark](https://www.kaggle.com/datasets/googleai/musiccaps) and on an in-domain held-out evaluation set, with no artist overlap with the training set.

## Training datasets

The model was trained on licensed data using the following sources: the [Meta Music Initiative Sound Collection](https://www.fb.com/sound),  [Shutterstock music collection](https://www.shutterstock.com/music) and the [Pond5 music collection](https://www.pond5.com/). See the paper for more details about the training set and corresponding preprocessing.

## Evaluation results

Below are the objective metrics obtained on MusicCaps with the released model. Note that for the publicly released models, we used the state-of-the-art music source separation method, namely the open source [Hybrid Transformer for Music Source Separation](https://github.com/facebookresearch/demucs) (HT-Demucs), in order to keep only instrumental tracks. This explains the difference in objective metrics with the models used in the paper.

| Model | Frechet Audio Distance | KLD | Text Consistency |
|---|---|---|---|
| **facebook/magnet-small-10secs**  | 4.22 | 1.11 | 0.28 |
| facebook/magnet-medium-10secs | 4.61 | 1.14 | 0.28 |
| facebook/magnet-small-30secs  | 4.35 | 1.17 | 0.28 |
| facebook/magnet-medium-30secs | 4.63 | 1.20 | 0.28 |

More information can be found in the paper  [Masked Audio Generation using a Single Non-Autoregressive Transformer][arxiv], in the Results section.

## Limitations and biases

**Data:** The data sources used to train the model are created by music professionals and covered by legal agreements with the right holders. The model is trained on 16K hours of data, we believe that scaling the model on larger datasets can further improve the performance of the model.

**Mitigations:** Tracks that include vocals have been removed from the data source using corresponding tags, and using a state-of-the-art music source separation method, namely using the open source [Hybrid Transformer for Music Source Separation](https://github.com/facebookresearch/demucs) (HT-Demucs).

**Limitations:**

- The model is not able to generate realistic vocals.
- The model has been trained with English descriptions and will not perform as well in other languages.
- The model does not perform equally well for all music styles and cultures.
- The model sometimes generates end of songs, collapsing to silence.
- It is sometimes difficult to assess what types of text descriptions provide the best generations. Prompt engineering may be required to obtain satisfying results.

**Biases:** The source of data is potentially lacking diversity and all music cultures are not equally represented in the dataset. The model may not perform equally well on the wide variety of music genres that exists. The generated samples from the model will reflect the biases from the training data. Further work on this model should include methods for balanced and just representations of cultures, for example, by scaling the training data to be both diverse and inclusive.

**Risks and harms:** Biases and limitations of the model may lead to generation of samples that may be considered as biased, inappropriate or offensive. We believe that providing the code to reproduce the research and train new models will allow to broaden the application to new and more representative data.

**Use cases:** Users must be aware of the biases, limitations and risks of the model. MAGNeT is a model developed for artificial intelligence research on controllable music generation. As such, it should not be used for downstream applications without further investigation and mitigation of risks.

[arxiv]: https://arxiv.org/abs/2401.04577

## Audio-MAGNeT - Sound-effect generation models

### Training datasets

The audio-MAGNeT models were trained on the following data sources: a subset of AudioSet (Gemmeke et al., 2017), [BBC sound effects](https://sound-effects.bbcrewind.co.uk/), AudioCaps (Kim et al., 2019), Clotho v2 (Drossos et al., 2020), VGG-Sound (Chen et al., 2020), FSD50K (Fonseca et al., 2021), [Free To Use Sounds](https://www.freetousesounds.com/all-in-one-bundle/), [Sonniss Game Effects](https://sonniss.com/gameaudiogdc), [WeSoundEffects](https://wesoundeffects.com/we-sound-effects-bundle-2020/), [Paramount Motion - Odeon Cinematic Sound Effects](https://www.paramountmotion.com/odeon-sound-effects).


### Evaluation datasets

The audio-magnet models (sound effect generation) were evaluated on the [AudioCaps benchmark](https://audiocaps.github.io/).

### Evaluation results

Below are the objective metrics obtained with the released audio-magnet models on AudioCaps (consisting of 10-second long samples). 

| Model | Frechet Audio Distance | KLD |
|---|---|---|
| facebook/audio-magnet-small | 3.21 | 1.42 |
| facebook/audio-magnet-medium | 2.32 | 1.64 |
