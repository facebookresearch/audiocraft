# AudioCraft conditioning modules

AudioCraft provides a
[modular implementation of conditioning modules](../audiocraft/modules/conditioners.py)
that can be used with the language model to condition the generation.
The codebase was developed in order to easily extend the set of modules
currently supported to easily develop new ways of controlling the generation.


## Conditioning methods

For now, we support 3 main types of conditioning within AudioCraft:
* Text-based conditioning methods
* Waveform-based conditioning methods
* Joint embedding conditioning methods for text and audio projected in a shared latent space.

The Language Model relies on 2 core components that handle processing information:
* The `ConditionProvider` class, that maps metadata to processed conditions, leveraging
all the defined conditioners for the given task.
* The `ConditionFuser` class, that takes preprocessed conditions and properly fuse the
conditioning embedding to the language model inputs following a given fusing strategy.

Different conditioners (for text, waveform, joint embeddings...) are provided as torch
modules in AudioCraft and are used internally in the language model to process the
conditioning signals and feed them to the language model.


## Core concepts

### Conditioners

The `BaseConditioner` torch module is the base implementation for all conditioners in AudioCraft.

Each conditioner is expected to implement 2 methods:
* The `tokenize` method that is used as a preprocessing method that contains all processing
that can lead to synchronization points (e.g. BPE tokenization with transfer to the GPU).
The output of the tokenize method will then be used to feed the forward method.
* The `forward` method that takes the output of the tokenize method and contains the core computation
to obtain the conditioning embedding along with a mask indicating valid indices (e.g. padding tokens).

### ConditionProvider

The ConditionProvider prepares and provides conditions given a dictionary of conditioners.

Conditioners are specified as a dictionary of attributes and the corresponding conditioner
providing the processing logic for the given attribute.

Similarly to the conditioners, the condition provider works in two steps to avoid synchronization points:
* A `tokenize` method that takes a list of conditioning attributes for the batch,
and runs all tokenize steps for the set of conditioners.
* A `forward` method that takes the output of the tokenize step and runs all the forward steps
for the set of conditioners.

The list of conditioning attributes is passed as a list of `ConditioningAttributes`
that is presented just below.

### ConditionFuser

Once all conditioning signals have been extracted and processed by the `ConditionProvider`
as dense embeddings, they remain to be passed to the language model along with the original
language model inputs.

The `ConditionFuser` handles specifically the logic to combine the different conditions
to the actual model input, supporting different strategies to combine them.

One can therefore define different strategies to combine or fuse the condition to the input, in particular:
* Prepending the conditioning signal to the input with the `prepend` strategy,
* Summing the conditioning signal to the input with the `sum` strategy,
* Combining the conditioning relying on a cross-attention mechanism with the `cross` strategy,
* Using input interpolation with the `input_interpolate` strategy.

### SegmentWithAttributes and ConditioningAttributes: From metadata to conditions

The `ConditioningAttributes` dataclass is the base class for metadata
containing all attributes used for conditioning the language model.

It currently supports the following types of attributes:
* Text conditioning attributes: Dictionary of textual attributes used for text-conditioning.
* Wav conditioning attributes: Dictionary of waveform attributes used for waveform-based
conditioning such as the chroma conditioning.
* JointEmbed conditioning attributes: Dictionary of text and waveform attributes
that are expected to be represented in a shared latent space.

These different types of attributes are the attributes that are processed
by the different conditioners.

`ConditioningAttributes` are extracted from metadata loaded along the audio in the datasets,
provided that the metadata used by the dataset implements the `SegmentWithAttributes` abstraction.

All metadata-enabled datasets to use for conditioning in AudioCraft inherits
the [`audiocraft.data.info_dataset.InfoAudioDataset`](../audiocraft/data/info_audio_dataset.py) class
and the corresponding metadata inherits and implements the `SegmentWithAttributes` abstraction.
Refer to the [`audiocraft.data.music_dataset.MusicAudioDataset`](../audiocraft/data/music_dataset.py)
class as an example.


## Available conditioners

### Text conditioners

All text conditioners are expected to inherit from the `TextConditioner` class.

AudioCraft currently provides two text conditioners:
* The `LUTConditioner` that relies on look-up-table of embeddings learned at train time,
and relying on either no tokenizer or a spacy tokenizer. This conditioner is particularly
useful for simple experiments and categorical labels.
* The `T5Conditioner` that relies on a
[pre-trained T5 model](https://huggingface.co/docs/transformers/model_doc/t5)
frozen or fine-tuned at train time to extract the text embeddings.

### Waveform conditioners

All waveform conditioners are expected to inherit from the `WaveformConditioner` class and
consist of a conditioning method that takes a waveform as input. The waveform conditioner
must implement the logic to extract the embedding from the waveform and define the downsampling
factor from the waveform to the resulting embedding.

The `ChromaStemConditioner` conditioner is a waveform conditioner for the chroma features
conditioning used by MusicGen. It takes a given waveform, extracts relevant stems for melody
(namely all non drums and bass stems) using a
[pre-trained Demucs model](https://github.com/facebookresearch/demucs)
and then extracts the chromagram bins from the remaining mix of stems.

### Joint embeddings conditioners

We finally provide support for conditioning based on joint text and audio embeddings through
the `JointEmbeddingConditioner` class and the `CLAPEmbeddingConditioner` that implements such
a conditioning method relying on a [pretrained CLAP model](https://github.com/LAION-AI/CLAP).

## Classifier Free Guidance

We provide a Classifier Free Guidance implementation in AudioCraft. With the classifier free
guidance dropout, all attributes are dropped with the same probability.

## Attribute Dropout

We further provide an attribute dropout strategy. Unlike the classifier free guidance dropout,
the attribute dropout drops given attributes with a defined probability, allowing the model
not to expect all conditioning signals to be provided at once.

## Faster computation of conditions

Conditioners that require some heavy computation on the waveform can be cached, in particular
the `ChromaStemConditioner` or `CLAPEmbeddingConditioner`. You just need to provide the
`cache_path` parameter to them. We recommend running dummy jobs for filling up the cache quickly.
An example is provided in the [musicgen.musicgen_melody_32khz grid](../audiocraft/grids/musicgen/musicgen_melody_32khz.py).