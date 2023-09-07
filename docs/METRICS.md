# AudioCraft objective metrics

In addition to training losses, AudioCraft provides a set of objective metrics
for audio synthesis and audio generation. As these metrics may require
extra dependencies and can be costly to train, they are often disabled by default.
This section provides guidance for setting up and using these metrics in
the AudioCraft training pipelines.

## Available metrics

### Audio synthesis quality metrics

#### SI-SNR

We provide an implementation of the Scale-Invariant Signal-to-Noise Ratio in PyTorch.
No specific requirement is needed for this metric. Please activate the metric at the
evaluation stage with the appropriate flag:

**Warning:** We report the opposite of the SI-SNR, e.g. multiplied by -1. This is due to internal 
    details where the SI-SNR score can also be used as a training loss function, where lower
    values should indicate better reconstruction. Negative values are such expected and a good sign! Those should be again multiplied by `-1` before publication :)

```shell
dora run <...> evaluate.metrics.sisnr=true
```

#### ViSQOL

We provide a Python wrapper around the ViSQOL [official implementation](https://github.com/google/visqol)
to conveniently run ViSQOL within the training pipelines.

One must specify the path to the ViSQOL installation through the configuration in order
to enable ViSQOL computations in AudioCraft:

```shell
# the first parameter is used to activate visqol computation while the second specify
# the path to visqol's library to be used by our python wrapper
dora run <...> evaluate.metrics.visqol=true metrics.visqol.bin=<path_to_visqol>
```

See an example grid: [Compression with ViSQOL](../audiocraft/grids/compression/encodec_musicgen_32khz.py)

To learn more about ViSQOL and how to build ViSQOL binary using bazel, please refer to the
instructions available in the [open source repository](https://github.com/google/visqol).

### Audio generation metrics

#### Frechet Audio Distance

Similarly to ViSQOL, we use a Python wrapper around the Frechet Audio Distance
[official implementation](https://github.com/google-research/google-research/tree/master/frechet_audio_distance)
in TensorFlow.

Note that we had to make several changes to the actual code in order to make it work.
Please refer to the [FrechetAudioDistanceMetric](../audiocraft/metrics/fad.py) class documentation
for more details. We do not plan to provide further support in obtaining a working setup for the
Frechet Audio Distance at this stage.

```shell
# the first parameter is used to activate FAD metric computation while the second specify
# the path to FAD library to be used by our python wrapper
dora run <...> evaluate.metrics.fad=true metrics.fad.bin=<path_to_google_research_repository>
```

See an example grid: [Evaluation with FAD](../audiocraft/grids/musicgen/musicgen_pretrained_32khz_eval.py)

#### Kullback-Leibler Divergence

We provide a PyTorch implementation of the Kullback-Leibler Divergence computed over the probabilities
of the labels obtained by a state-of-the-art audio classifier. We provide our implementation of the KLD
using the [PaSST classifier](https://github.com/kkoutini/PaSST).

In order to use the KLD metric over PaSST, you must install the PaSST library as an extra dependency:
```shell
pip install 'git+https://github.com/kkoutini/passt_hear21@0.0.19#egg=hear21passt'
```

Then similarly, you can use the metric activating the corresponding flag:

```shell
# one could extend the kld metric with additional audio classifier models that can then be picked through the configuration
dora run <...> evaluate.metrics.kld=true metrics.kld.model=passt
```

#### Text consistency

We provide a text-consistency metric, similarly to the MuLan Cycle Consistency from
[MusicLM](https://arxiv.org/pdf/2301.11325.pdf) or the CLAP score used in
[Make-An-Audio](https://arxiv.org/pdf/2301.12661v1.pdf).
More specifically, we provide a PyTorch implementation of a Text consistency metric
relying on a pre-trained [Contrastive Language-Audio Pretraining (CLAP)](https://github.com/LAION-AI/CLAP).

Please install the CLAP library as an extra dependency prior to using the metric:
```shell
pip install laion_clap
```

Then similarly, you can use the metric activating the corresponding flag:

```shell
# one could extend the text consistency metric with additional audio classifier models that can then be picked through the configuration
dora run ... evaluate.metrics.text_consistency=true metrics.text_consistency.model=clap
```

Note that the text consistency metric based on CLAP will require the CLAP checkpoint to be
provided in the configuration.

#### Chroma cosine similarity

Finally, as introduced in MusicGen, we provide a Chroma Cosine Similarity metric in PyTorch.
No specific requirement is needed for this metric. Please activate the metric at the
evaluation stage with the appropriate flag:

```shell
dora run ... evaluate.metrics.chroma_cosine=true
```

#### Comparing against reconstructed audio

For all the above audio generation metrics, we offer the option to compute the metric on the reconstructed audio
fed in EnCodec instead of the generated sample using the flag `<metric>.use_gt=true`.

## Example usage

You will find example of configuration for the different metrics introduced above in:
* The [musicgen's default solver](../config/solver/musicgen/default.yaml) for all audio generation metrics
* The [compression's default solver](../config/solver/compression/default.yaml) for all audio synthesis metrics

Similarly, we provide different examples in our grids:
* [Evaluation with ViSQOL](../audiocraft/grids/compression/encodec_musicgen_32khz.py)
* [Evaluation with FAD and others](../audiocraft/grids/musicgen/musicgen_pretrained_32khz_eval.py)
