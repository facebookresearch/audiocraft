# AudioGen Model Card

## Model details
**Organization developing the model:** The FAIR team of Meta AI.

**Model date:** This version of AudioGen was trained between July 2023 and August 2023.

**Model version:** This is version 2 of the model, not to be confused with the original AudioGen model published in ["AudioGen: Textually Guided Audio Generation"][audiogen].
In this version (v2), AudioGen was trained on the same data, but with some other differences:
1. This model was trained on 10 seconds (vs. 5 seconds in v1).
2. The discrete representation used under the hood is extracted using a retrained EnCodec model on the environmental sound data, following the EnCodec setup detailed in the ["Simple and Controllable Music Generation" paper][musicgen].
3. No audio mixing augmentations.

**Model type:** AudioGen consists of an EnCodec model for audio tokenization, and an auto-regressive language model based on the transformer architecture for audio modeling. The released model has 1.5B parameters.

**Paper or resource for more information:** More information can be found in the paper [AudioGen: Textually Guided Audio Generation](https://arxiv.org/abs/2209.15352).

**Citation details:** See [AudioGen paper][audiogen]

**License:** Code is released under MIT, model weights are released under CC-BY-NC 4.0.

**Where to send questions or comments about the model:** Questions and comments about AudioGen can be sent via the [GitHub repository](https://github.com/facebookresearch/audiocraft) of the project, or by opening an issue.

## Intended use
**Primary intended use:** The primary use of AudioGen is research on AI-based audio generation, including:
- Research efforts, such as probing and better understanding the limitations of generative models to further improve the state of science
- Generation of sound guided by text to understand current abilities of generative AI models by machine learning amateurs

**Primary intended users:** The primary intended users of the model are researchers in audio, machine learning and artificial intelligence, as well as amateur seeking to better understand those models.

**Out-of-scope use cases** The model should not be used on downstream applications without further risk evaluation and mitigation. The model should not be used to intentionally create or disseminate audio pieces that create hostile or alienating environments for people. This includes generating audio that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.

## Metrics

**Models performance measures:** We used the following objective measure to evaluate the model on a standard audio benchmark:
- Frechet Audio Distance computed on features extracted from a pre-trained audio classifier (VGGish)
- Kullback-Leibler Divergence on label distributions extracted from a pre-trained audio classifier (PaSST)

Additionally, we run qualitative studies with human participants, evaluating the performance of the model with the following axes:
- Overall quality of the audio samples;
- Text relevance to the provided text input;

More details on performance measures and human studies can be found in the paper.

**Decision thresholds:** Not applicable.

## Evaluation datasets

The model was evaluated on the [AudioCaps benchmark](https://audiocaps.github.io/).

## Training datasets

The model was trained on the following data sources: a subset of AudioSet (Gemmeke et al., 2017), [BBC sound effects](https://sound-effects.bbcrewind.co.uk/), AudioCaps (Kim et al., 2019), Clotho v2 (Drossos et al., 2020), VGG-Sound (Chen et al., 2020), FSD50K (Fonseca et al., 2021), [Free To Use Sounds](https://www.freetousesounds.com/all-in-one-bundle/), [Sonniss Game Effects](https://sonniss.com/gameaudiogdc), [WeSoundEffects](https://wesoundeffects.com/we-sound-effects-bundle-2020/), [Paramount Motion - Odeon Cinematic Sound Effects](https://www.paramountmotion.com/odeon-sound-effects).

## Evaluation results

Below are the objective metrics obtained with the released model on AudioCaps (consisting of 10-second long samples). Note that the model differs from the original AudioGen model introduced in the paper, hence the difference in the metrics.

| Model | Frechet Audio Distance | KLD | Text consistency |
|---|---|---|---|
| facebook/audiogen-medium | 1.77 | 1.58 | 0.30 |

More information can be found in the paper [AudioGen: Textually Guided Audio Generation][audiogen], in the Experiments section.

## Limitations and biases

**Limitations:**
- The model is not able to generate realistic vocals.
- The model has been trained with English descriptions and will not perform as well in other languages.
- It is sometimes difficult to assess what types of text descriptions provide the best generations. Prompt engineering may be required to obtain satisfying results.

**Biases:** The datasets used for training may be lacking of diversity and are not representative of all possible sound events. The generated samples from the model will reflect the biases from the training data.

**Risks and harms:** Biases and limitations of the model may lead to generation of samples that may be considered as biased, inappropriate or offensive. We believe that providing the code to reproduce the research and train new models will allow to broaden the application to new and more representative data.

**Use cases:** Users must be aware of the biases, limitations and risks of the model. AudioGen is a model developed for artificial intelligence research on audio generation. As such, it should not be used for downstream applications without further investigation and mitigation of risks.

[musicgen]: https://arxiv.org/abs/2306.05284
[audiogen]: https://arxiv.org/abs/2209.15352
