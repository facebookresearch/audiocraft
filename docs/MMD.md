# An Independence-promoting Loss for Music Generation with Language Models

AudioCraft provides the training code for EnCodec-MMD, a variant of EnCodec,
where a criterion based on the Maximum-Mean Discrepancy principle
in Gaussian Reproduce Kernel Hilbert Spaces is introduced.
This auxiliary training loss is optimized to promote independence 
between the codebooks of the RVQ quantizer located in the bottleneck of EnCodec.
We use this modified codec for music generation: having more independent 
codebooks enables the modeling of the factorized distribution of EnCodec codes,
implied by the decoding strategy of MusicGen, to be closer in theory to the 
true model of the joint distribution.
The training criterion, its correlation with inter-codebook dependence and 
music generation results are presented in our 
[An Independence-promoting Loss for Music Generation with Language Models][ICML_arxiv] paper.
You can check out our [sample page][musicgen-mmd_samples].


## Installation

Please follow the AudioCraft installation instructions from the [README](../README.md).


## Training

The [CompressionSolver](../audiocraft/solvers/compression.py) implements the audio reconstruction
task to train an EnCodec model, just as with the traditional EnCodec solver.
Specifically, it trains an encoder-decoder with a quantization
bottleneck - a SEANet encoder-decoder with Residual Vector Quantization bottleneck for EnCodec -
using a combination of objective and perceptual losses in the forms of discriminators.
This solver is modified to include our MMD-based loss.

### Example configuration and grids

We provide sample configuration and grids for training the 32kHz mono EnCodec-MMD model
presented in the paper.

The compression configuration is defined in
[config/solver/compression/encodec_musicgen_mmd_32khz.yaml]
(../config/solver/compression/encodec_musicgen_mmd_32khz.yaml).

The corresponding example grid is available at
[audiocraft/grids/compression/encodec_musicgen_mmd_32khz.py]
(../audiocraft/grids/encodec_musicgen_mmd_32khz.py).

```shell
# encodec-mmd model used for MusicGen-MMD on monophonic audio sampled at 32 khz
dora grid compression.encodec_musicgen_mmd_32khz
```

### Playing with the model

<!-- We provide a pretrained EnCodec-MMD model [here][drive_link]. -->
We will provide a pretrained EnCodec-MMD 32kHz model very soon.
Please bear with us.
Once you have a model trained, it is possible to get the entire solver, or just
the trained model with the following functions:

```python
from audiocraft.solvers import CompressionSolver

# If you trained a custom model with signature SIG.
model = CompressionSolver.model_from_checkpoint('//sig/SIG')
# If you want to get one of the pretrained models with the `//pretrained/` prefix, using
# the model available online
model = CompressionSolver.model_from_checkpoint('//pretrained//foo/bar/checkpoint.th')

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

We are currently working on providing the pretrained
EnCodec-MMD model on Hugging Face.
<!-- 
We still have some support for fine-tuning an EnCodec model coming from HF in AudioCraft,
using for instance `continue_from=//pretrained/facebook/encodec_32k`. -->

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
@article{lemercier2024independence,
  title={An Independence-promoting Loss for Music Generation with Language Models},
  author={Lemercier, Jean-Marie and Rouard, Simon and Copet, Jade and Adi, Yossi and Défossez, Alexandre},
  journal={ICML},
  year={2024}
}
```


## License

See license information in the [README](../README.md).

[ICML_arxiv]: http://arxiv.org/abs/2406.02315
[musicgen-mmd_samples]: https://jmlemercier.github.io/encodec-mmd.github.io/
[drive_link]: https://drive.google.com/drive/u/2/folders/1KYQ_kQgFZDkOdFRFLEREi7tFsNdi7RLS
