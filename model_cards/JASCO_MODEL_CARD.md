## Model details

**Organization developing the model:** The FAIR team of Meta AI.

**Model date:** JASCO was trained in November 2024.

**Model version:** This is the version 1 of the model.

**Model type:** JASCO consists of an EnCodec model for audio tokenization, and a flow-matching model based on the transformer architecture for music modeling.
The model comes in different sizes: 400M and 1B; and currently have a two variant: text-to-music + {chords, drums} controls and text-to-music + {chords, drums, melody} controls.
JASCO is trained with condition dropout and could be used for inference with dropped conditions.
 
**Paper or resources for more information:** More information can be found in the paper [Joint Audio And Symbolic Conditioning for Temporally Controlled Text-To-Music Generation][arxiv].

**Citation details:** 

Code was implemented by Or Tal and Alon Ziv.

```
@misc{tal2024joint,
    title={Joint Audio and Symbolic Conditioning for Temporally Controlled Text-to-Music Generation}, 
    author={Or Tal and Alon Ziv and Itai Gat and Felix Kreuk and Yossi Adi},
    year={2024},
    eprint={2406.10970},
    archivePrefix={arXiv},
    primaryClass={cs.SD}
}
```

**License:** Code is released under MIT, model weights are released under CC-BY-NC 4.0.

**Where to send questions or comments about the model:** Questions and comments about JASCO can be sent via the [GitHub repository](https://github.com/facebookresearch/audiocraft) of the project, or by opening an issue.

## Intended use
**Primary intended use:** The primary use of JASCO is research on AI-based music generation, including:

- Research efforts, such as probing and better understanding the limitations of generative models to further improve the state of science
- Generation of music guided by text and (opt) local controls, to understand current abilities of generative AI models by machine learning amateurs

**Primary intended users:** The primary intended users of the model are researchers in audio, machine learning and artificial intelligence, as well as amateur seeking to better understand those models.

**Out-of-scope use cases:** The model should not be used on downstream applications without further risk evaluation and mitigation. The model should not be used to intentionally create or disseminate music pieces that create hostile or alienating environments for people. This includes generating music that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.

## Metrics

**Models performance measures:** We used the following objective measure to evaluate the model on a standard music benchmark:

- Frechet Audio Distance computed on features extracted from a pre-trained audio classifier (VGGish).
- CLAP Score between audio embedding and text embedding extracted from a pre-trained CLAP model.
- Melody cosine similarity - pairwise comparison of chromagram extracted from refrence and generated waveforms.
- Onset F1 - pairwise comparison of onsets extracted from refrence and generated waveforms.
- Chords Intersection over union (IOU) - pairwise comparison of symbolic chords extracted from refrence and generated waveforms.

Additionally, we run qualitative studies with human participants, evaluating the performance of the model with the following axes:

- Overall quality of the music samples;
- Text relevance to the provided text input;
- Melody match w.r.t reference signal;
- Drum beat match w.r.t reference signal;

More details on performance measures and human studies can be found in the [paper][arxiv].

**Decision thresholds:** Not applicable.

## Evaluation datasets

The model was evaluated on the [MusicCaps benchmark](https://www.kaggle.com/datasets/googleai/musiccaps) and on an in-domain held-out evaluation set, with no artist overlap with the training set.

## Training datasets

The model was trained on licensed data using the following sources: the [Meta Music Initiative Sound Collection](https://www.fb.com/sound),  [Shutterstock music collection](https://www.shutterstock.com/music) and the [Pond5 music collection](https://www.pond5.com/). See the paper for more details about the training set and corresponding preprocessing.

## Evaluation results

Below are the objective metrics obtained on MusicCaps with the released model. 

Text-to-music with temporal controls

| Model                                   | Frechet Audio Distance | Text Consistency | Chord IOU | Onset F1 | Melody Cosine Similarity |
|---|---|---|---|---|---|
| facebook/jasco-chords-drums-400M        | 5.866                  | 0.284            | 0.588     | 0.328    | 0.096                    |
| facebook/jasco-chords-drums-1B          | 5.587                  | 0.291            | 0.589     | 0.331    | 0.097                    |
| facebook/jasco-chords-drums-melody-400M | 4.730                  | 0.317            | 0.689     | 0.379    | 0.423                    |
| facebook/jasco-chords-drums-melody-1B   | 5.098                  | 0.313            | 0.690     | 0.378    | 0.427                    |

Note: reccommanded CFG coefficient ratio stands at 1:2 - 'all':'text', results for chords-drums-melody were sampled with all: 1.75, text: 3.5 

Text-to-music w.o temporal controls (dropped)


| Model                                   | Frechet Audio Distance | Text Consistency | Chord IOU | Onset F1 | Melody Cosine Similarity |
|---|---|---|---|---|---|
| facebook/jasco-chords-drums-400M        | 5.648                  | 0.272            | 0.070     | 0.204    | 0.093                    |
| facebook/jasco-chords-drums-1B          | 5.602                  | 0.281            | 0.071     | 0.214    | 0.093                    |
| facebook/jasco-chords-drums-melody-400M | 5.816                  | 0.293            | 0.091     | 0.203    | 0.098                    |
| facebook/jasco-chords-drums-melody-1B   | 5.470                  | 0.297            | 0.097     | 0.208    | 0.097                    |

## Limitations and biases

**Data:** The data sources used to train the model are created by music professionals and covered by legal agreements with the right holders. The model is trained on ~16k hours of data, we believe that scaling the model on larger datasets can further improve the performance of the model.

**Mitigations:** 
Pre-trained models were used to obtain pseudo symbolic supervision. Refer to **Data Preprocessing** section in [Jasco's docs](../docs/JASCO.md)

**Limitations:**

- The model is not able to generate realistic vocals.
- The model has been trained with English descriptions and will not perform as well in other languages.
- The model does not perform equally well for all music styles and cultures.
- It is sometimes difficult to assess what types of text descriptions provide the best generations. Prompt engineering and experimentation with classifier free guidance coefficients may be required to obtain satisfying results.
- Model could be sensitive to CFG coefficients as melody introduces a strong bias that would require higher text coefficient during generation, some hyper-parameter search could be necessary to obtain desired results.

**Biases:** The source of data is potentially lacking diversity and all music cultures are not equally represented in the dataset. The model may not perform equally well on the wide variety of music genres that exists. The generated samples from the model will reflect the biases from the training data. Further work on this model should include methods for balanced and just representations of cultures, for example, by scaling the training data to be both diverse and inclusive.

**Risks and harms:** Biases and limitations of the model may lead to generation of samples that may be considered as biased, inappropriate or offensive. We believe that providing the code to reproduce the research and train new models will allow to broaden the application to new and more representative data.

**Use cases:** Users must be aware of the biases, limitations and risks of the model. JASCO is a model developed for artificial intelligence research on controllable music generation. As such, it should not be used for downstream applications without further investigation and mitigation of risks.

## API

We provide a simple API and pre-trained models:
- `facebook/jasco-chords-drums-400M`: 400M model, text to music with chords and drums support, generates 10-second samples - [🤗 Hub](https://huggingface.co/facebook/jasco-chords-drums-400M)
- `facebook/jasco-chords-drums-1B`: 1B model, text to music with chords and drums support, generates 10-second samples - [🤗 Hub](https://huggingface.co/facebook/jasco-chords-drums-1B)
- `facebook/jasco-chords-drums-melody-400M`: 400M model, text to music with chords, drums and melody support, generates 10-second samples - [🤗 Hub](https://huggingface.co/facebook/jasco-chords-drums-melody-400M)
- `facebook/jasco-chords-drums-melody-1B`: 1B model, text to music with chords, drums and melody support, generates 10-second samples - [🤗 Hub](https://huggingface.co/facebook/jasco-chords-drums-melody-1B)


See after a quick example for using the API.

```python
from audiocraft.models import JASCO

model = JASCO.get_pretrained('facebook/jasco-chords-drums-400M', chords_mapping_path='../assets/chord_to_index_mapping.pkl')

model.set_generation_params(
    cfg_coef_all=1.5,
    cfg_coef_txt=0.5
)

# set textual prompt
text = "Strings, woodwind, orchestral, symphony."

# define chord progression
chords = [('C', 0.0), ('D', 2.0), ('F', 4.0), ('Ab', 6.0), ('Bb', 7.0), ('C', 8.0)]

# run inference
output = model.generate_music(descriptions=[text], chords=chords, progress=True)

audio_write('output', output.cpu().squeeze(0), model.sample_rate, strategy="loudness", loudness_compressor=True)
```

[arxiv]: https://arxiv.org/pdf/2406.10970