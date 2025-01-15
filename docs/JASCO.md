# JASCO: Joint Audio And Symbolic Conditioning for Temporally Controlled Text-To-Music Generation

AudioCraft provides the code and models for JASCO, [Joint Audio And Symbolic Conditioning for Temporally Controlled Text-To-Music Generation][arxiv].

We present JASCO, a temporally controlled text-to-music generation model utilizing both symbolic and audio-based conditions.
JASCO can generate high-quality music samples conditioned on global text descriptions along with fine-grained local controls.
JASCO is based on the Flow Matching modeling paradigm together with a novel conditioning method, allowing for music generation controlled both locally (e.g., chords) and globally (text description).

Check out our [sample page][sample_page] or test the available demo!

We use ~16K hours of licensed music to train JASCO. 


## Model Card

See [the model card](../model_cards/JASCO_MODEL_CARD.md).


## Installation

First, Please follow the AudioCraft installation instructions from the [README](../README.md).

Then, download and install chord_extractor from [source](http://www.isophonics.net/nnls-chroma)

See further required installation under **Data Preprocessing** section

## Usage

We currently offer two ways to interact with JASCO:
1. You can use the gradio demo locally by running [`python -m demos.jasco_app`](../demos/jasco_app.py), you can add `--share` to deploy a sharable space mounted on your device.
2. You can play with JASCO by running the jupyter notebook at [`demos/jasco_demo.ipynb`](../demos/jasco_demo.ipynb) locally.

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
    cfg_coef_all=5.0,
    cfg_coef_txt=0.0
)

# set textual prompt
text = "Strings, woodwind, orchestral, symphony."

# define chord progression
chords = [('C', 0.0), ('D', 2.0), ('F', 4.0), ('Ab', 6.0), ('Bb', 7.0), ('C', 8.0)]

# run inference
output = model.generate_music(descriptions=[text], chords=chords, progress=True)

audio_write('output', output.cpu().squeeze(0), model.sample_rate, strategy="loudness", loudness_compressor=True)
```

For more examples check out `demos/jasco_demo.ipynb`

## 🤗 Transformers Usage

Coming soon...

## Data Preprocessing
In order to to use the JascoDataset with chords / melody conditioning, please follow the instructions below:


### Chords conditioning
To extract chords from your desired data follow the following steps:

1. Prepare a `*.jsonl` containing list of absolute file paths in your dataset, should simply be absolute paths seperated by newlines.
2. Download and install chord_extractor from [source](http://www.isophonics.net/nnls-chroma)
3. For training purposes run: `python scripts/chords/extract_chords.py --src_jsonl_file=<abs path to .jsonl file containing list of absolute file paths seperated by new line> --target_output_dir=<target directory to save parsed chord files to, individual files will be saved inside>`
<br>
and then run: `python scripts/chords/build_chord_map.py --chords_folder=<abs path to directory containing parsed chords files> --output_directory=<path to output directory to generate code maps to, if not given - chords_folder would be used>`

4. For evaluation of our released models run: `python scripts/chords/extract_chords.py --src_jsonl_file=<abs path to .jsonl file containing list of absolute file paths seperated by new line> --target_output_dir=<target directory to save parsed chord files to, individual files will be saved inside> --path_to_pre_defined_map=<abs path to pre-defined mapping file>`
<br>
and then run: `python scripts/chords/build_chord_map.py --chords_folder=<abs path to directory containing parsed chords files> --output_directory=<path to output directory to generate code maps to, if not given - chords_folder would be used> --path_to_pre_defined_map=<for evaluation purpose, use pre-defined chord-to-index map absolute path>`


NOTE: current scripts assume that all audio files are of `.wav` format, some changes may be required if your data consists of other formats.

NOTE: predefined chord mapping file is available in `assets` directory.

### Melody conditioning

This section relies on [Deepsalience repo](https://github.com/rabitt/ismir2017-deepsalience) with slight custom scripts written.

#### Clone repo and create virtual environment
1. `git clone git@github.com:lonzi/ismir2017-deepsalience.git forked_deepsalience_repo`
2. `cd forked_deepsalience_repo`
3. `conda create --name deep_salience python=3.7`
4. `conda activate deep_salience`
5. `pip install -r requirements.txt`


#### Salience map dumps (of entire directory, using slurm job)

##### From src dir

1. create job array: `python predict/create_predict_saliency_cmds.py --src_dir=<path to dir containing files> --out_dir=<path to desired dir to dump files to> --n_shards=<desired number of shards> --multithread`
2. run job array: `sbatch predict_saliency.sh`

##### From track list

1. create job array: `python predict/create_predict_saliency_cmds.py --tracks_list=tracks_train.txt --out_dir=<path to desired dir to dump files to> --n_shards=2 --multithread --sbatch_script_name=predict_saliency_train.sh --saliency_threshold=<threshold, ours is 0.5>`
2. run job array: `sbatch predict_saliency_train.sh`

tracks_train.txt: a list of track paths to process seperated by new lines


## Training

The [JascoSolver](../audiocraft/solvers/jasco.py) implements JASCO's training pipeline.
conditional flow matching objective over the continuous extracted latents from a pre-trained EnCodec model (see [EnCodec documentation](./ENCODEC.md)
for more details on how to train such model).

Note that **we do NOT provide any of the datasets** used for training JASCO.
We provide a dummy dataset containing just a few examples for illustrative purposes.

Please read first the [TRAINING documentation](./TRAINING.md), in particular the Environment Setup section.


### Fine tuning existing models

You can initialize your model to one of the pretrained models by using the `continue_from` argument, in particular

```bash
# Using pretrained JASCO model.
dora run solver=jasco/chords_drums model/lm/model_scale=small continue_from=//pretrained/facebook/jasco-chords-drums-400M conditioner=jasco_chords_drums

# Using another model you already trained with a Dora signature SIG.
dora run solver=jasco/chords_drums model/lm/model_scale=small continue_from=//sig/SIG conditioner=jasco_chords_drums

# Or providing manually a path
dora run solver=jasco/chords_drums model/lm/model_scale=small conditioner=jasco_chords_drums continue_from=/checkpoints/my_other_xp/checkpoint.th
```

**Warning:** You are responsible for selecting the other parameters accordingly, in a way that make it compatible
    with the model you are fine tuning. Configuration is NOT automatically inherited from the model you continue from. In particular make sure to select the proper `conditioner` and `model/lm/model_scale`.

**Warning:** We currently do not support fine tuning a model with slightly different layers. If you decide
 to change some parts, like the conditioning or some other parts of the model, you are responsible for manually crafting a checkpoint file from which we can safely run `load_state_dict`.
 If you decide to do so, make sure your checkpoint is saved with `torch.save` and contains a dict
    `{'best_state': {'model': model_state_dict_here}}`. Directly give the path to `continue_from` without a `//pretrained/` prefix.


### Evaluation & Generation stage

See [MusicGen](./MUSICGEN.md)

### Playing with the model

Once you have launched some experiments, you can easily get access
to the Solver with the latest trained model using the following snippet.

```python
from audiocraft.solvers.jasco import JascoSolver

solver = JascoSolver.get_eval_solver_from_sig('SIG', device='cpu', batch_size=8)
solver.model
solver.dataloaders
```

### Importing / Exporting models

We do not support currently loading a model from the Hugging Face implementation or exporting to it.
If you want to export your model in a way that is compatible with `audiocraft.models.JASCO`
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
jasco = audiocraft.models.JASCO.get_pretrained('/checkpoints/my_audio_lm/')
```


### Learn more

Learn more about AudioCraft training pipelines in the [dedicated section](./TRAINING.md).


## Citation
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

## License

See license information in the [model card](../model_cards/JASCO_MODEL_CARD.md).

[arxiv]: https://arxiv.org/pdf/2406.10970
[sample_page]: https://pages.cs.huji.ac.il/adiyoss-lab/JASCO/
