# AudioCraft datasets

Our dataset manifest files consist in 1-json-per-line files, potentially gzipped,
as `data.jsons` or `data.jsons.gz` files. This JSON contains the path to the audio
file and associated metadata. The manifest files are then provided in the configuration,
as `datasource` sub-configuration. A datasource contains the pointers to the paths of
the manifest files for each AudioCraft stage (or split) along with additional information
(eg. maximum sample rate to use against this dataset). All the datasources are under the
`dset` group config, with a dedicated configuration file for each dataset.

## Getting started

### Example

See the provided example in the directory that provides a manifest to use the example dataset
provided under the [dataset folder](../dataset/example).

The manifest files are stored in the [egs folder](../egs/example).

```shell
egs/
  example/data.json.gz
```

A datasource is defined in the configuration folder, in the dset group config for this dataset
at [config/dset/audio/example](../config/dset/audio/example.yaml):

```shell
# @package __global__

datasource:
  max_sample_rate: 44100
  max_channels: 2

  train: egs/example
  valid: egs/example
  evaluate: egs/example
  generate: egs/example
```

For proper dataset, one should create manifest for each of the splits and specify the correct path
to the given manifest in the datasource for each split.

Then, using a dataset through the configuration can be done pointing to the
corresponding dataset configuration:
```shell
dset=<dataset_name> # <dataset_name> should match the yaml file name

# for example
dset=audio/example
```

### Creating manifest files

Assuming you want to create manifest files to load with AudioCraft's AudioDataset, you can use
the following command to create new manifest files from a given folder containing audio files:

```shell
python -m audiocraft.data.audio_dataset <path_to_dataset_folder> egs/my_dataset/my_dataset_split/data.jsonl.gz

# For example to generate the manifest for dset=audio/example
# note: we don't use any split and we don't compress the jsonl file for this dummy example
python -m audiocraft.data.audio_dataset dataset/example egs/example/data.jsonl

# More info with: python -m audiocraft.data.audio_dataset --help
```

## Additional information

### MusicDataset and metadata

The MusicDataset is an AudioDataset with additional metadata. The MusicDataset expects
the additional metadata to be stored in a JSON file that has the same path as the corresponding
audio file, but with a `.json` extension.

### SoundDataset and metadata

The SoundDataset is an AudioDataset with descriptions metadata. Similarly to the MusicDataset,
the SoundDataset expects the additional metadata to be stored in a JSON file that has the same
path as the corresponding audio file, but with a `.json` extension. Additionally, the SoundDataset
supports an additional parameter pointing to an extra folder `external_metadata_source` containing
all the JSON metadata files given they have the same filename as the audio file.
