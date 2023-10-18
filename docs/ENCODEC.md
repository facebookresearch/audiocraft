# EnCodec: High Fidelity Neural Audio Compression

AudioCraft provides the training code for EnCodec, a state-of-the-art deep learning
based audio codec supporting both mono and stereo audio, presented in the
[High Fidelity Neural Audio Compression][arxiv] paper.
Check out our [sample page][encodec_samples].

## Original EnCodec models

The EnCodec models presented in High Fidelity Neural Audio Compression can be accessed
and used with the [EnCodec repository](https://github.com/facebookresearch/encodec).

**Note**: We do not guarantee compatibility between the AudioCraft and EnCodec codebases
and released checkpoints at this stage.


## Installation

Please follow the AudioCraft installation instructions from the [README](../README.md).


## Training

The [CompressionSolver](../audiocraft/solvers/compression.py) implements the audio reconstruction
task to train an EnCodec model. Specifically, it trains an encoder-decoder with a quantization
bottleneck - a SEANet encoder-decoder with Residual Vector Quantization bottleneck for EnCodec -
using a combination of objective and perceptual losses in the forms of discriminators.

The default configuration matches a causal EnCodec training at a single bandwidth.

### Example configuration and grids

We provide sample configuration and grids for training EnCodec models.

The compression configuration are defined in
[config/solver/compression](../config/solver/compression).

The example grids are available at
[audiocraft/grids/compression](../audiocraft/grids/compression).

```shell
# base causal encodec on monophonic audio sampled at 24 khz
dora grid compression.encodec_base_24khz
# encodec model used for MusicGen on monophonic audio sampled at 32 khz
dora grid compression.encodec_musicgen_32khz
```

### Training and validation stages

The model is trained using a combination of objective and perceptual losses.
More specifically, EnCodec is trained with the MS-STFT discriminator along with
objective losses through the use of a loss balancer to effectively weight
the different losses, in an intuitive manner.

### Evaluation stage

Evaluation metrics for audio generation:
* SI-SNR: Scale-Invariant Signal-to-Noise Ratio.
* ViSQOL: Virtual Speech Quality Objective Listener.

Note: Path to the ViSQOL binary (compiled with bazel) needs to be provided in
order to run the ViSQOL metric on the reference and degraded signals.
The metric is disabled by default.
Please refer to the [metrics documentation](../METRICS.md) to learn more.

### Generation stage

The generation stage consists in generating the reconstructed audio from samples
with the current model. The number of samples generated and the batch size used are
controlled by the `dataset.generate` configuration. The output path and audio formats
are defined in the generate stage configuration.

```shell
# generate samples every 5 epoch
dora run solver=compression/encodec_base_24khz generate.every=5
# run with a different dset
dora run solver=compression/encodec_base_24khz generate.path=<PATH_IN_DORA_XP_FOLDER>
# limit the number of samples or use a different batch size
dora grid solver=compression/encodec_base_24khz dataset.generate.num_samples=10 dataset.generate.batch_size=4
```

### Playing with the model

Once you have a model trained, it is possible to get the entire solver, or just
the trained model with the following functions:

```python
from audiocraft.solvers import CompressionSolver

# If you trained a custom model with signature SIG.
model = CompressionSolver.model_from_checkpoint('//sig/SIG')
# If you want to get one of the pretrained models with the `//pretrained/` prefix.
model = CompressionSolver.model_from_checkpoint('//pretrained/facebook/encodec_32khz')
# Or load from a custom checkpoint path
model = CompressionSolver.model_from_checkpoint('/my_checkpoints/foo/bar/checkpoint.th')


# If you only want to use a pretrained model, you can also directly get it
# from the CompressionModel base model class.
from audiocraft.models import CompressionModel

# Here do not put the `//pretrained/` prefix!
model = CompressionModel.get_pretrained('facebook/encodec_32khz')
model = CompressionModel.get_pretrained('dac_44khz')

# Finally, you can also retrieve the full Solver object, with its dataloader etc.
from audiocraft import train
from pathlib import Path
import logging
import os
import sys

# Uncomment the following line if you want some detailed logs when loading a Solver.
# logging.basicConfig(stream=sys.stderr, level=logging.INFO)

# You must always run the following function from the root directory.
os.chdir(Path(train.__file__).parent.parent)


# You can also get the full solver (only for your own experiments).
# You can provide some overrides to the parameters to make things more convenient.
solver = train.get_solver_from_sig('SIG', {'device': 'cpu', 'dataset': {'batch_size': 8}})
solver.model
solver.dataloaders
```

### Importing / Exporting models

At the moment we do not have a definitive workflow for exporting EnCodec models, for
instance to Hugging Face (HF). We are working on supporting automatic conversion between
AudioCraft and Hugging Face implementations.

We still have some support for fine-tuning an EnCodec model coming from HF in AudioCraft,
using for instance `continue_from=//pretrained/facebook/encodec_32k`.

An AudioCraft checkpoint can be exported in a more compact format (excluding the optimizer etc.)
using `audiocraft.utils.export.export_encodec`. For instance, you could run

```python
from audiocraft.utils import export
from audiocraft import train
xp = train.main.get_xp_from_sig('SIG')
export.export_encodec(
    xp.folder / 'checkpoint.th',
    '/checkpoints/my_audio_lm/compression_state_dict.bin')


from audiocraft.models import CompressionModel
model = CompressionModel.get_pretrained('/checkpoints/my_audio_lm/compression_state_dict.bin')

from audiocraft.solvers import CompressionSolver
# The two are strictly equivalent, but this function supports also loading from non-already exported models.
model = CompressionSolver.model_from_checkpoint('//pretrained//checkpoints/my_audio_lm/compression_state_dict.bin')
```

We will see then how to use this model as a tokenizer for MusicGen/AudioGen in the
[MusicGen documentation](./MUSICGEN.md).

### Learn more

Learn more about AudioCraft training pipelines in the [dedicated section](./TRAINING.md).


## Citation
```
@article{defossez2022highfi,
  title={High Fidelity Neural Audio Compression},
  author={Défossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  journal={arXiv preprint arXiv:2210.13438},
  year={2022}
}
```


## License

See license information in the [README](../README.md).

[arxiv]: https://arxiv.org/abs/2210.13438
[encodec_samples]: https://ai.honu.io/papers/encodec/samples.html
