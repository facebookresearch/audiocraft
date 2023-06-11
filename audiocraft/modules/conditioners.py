# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
import logging
import math
import random
import re
import typing as tp
import warnings

from einops import rearrange
from num2words import num2words
import spacy
from transformers import T5EncoderModel, T5Tokenizer  # type: ignore
import torchaudio
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .streaming import StreamingModule
from .transformer import create_sin_embedding
from ..data.audio_dataset import SegmentInfo
from ..utils.autocast import TorchAutocast
from ..utils.utils import hash_trick, length_to_mask, collate


logger = logging.getLogger(__name__)
TextCondition = tp.Optional[str]  # a text condition can be a string or None (if doesn't exist)
ConditionType = tp.Tuple[Tensor, Tensor]  # condition, mask


class WavCondition(tp.NamedTuple):
    wav: Tensor
    length: Tensor
    path: tp.List[tp.Optional[str]] = []


def nullify_condition(condition: ConditionType, dim: int = 1):
    """This function transforms an input condition to a null condition.
    The way it is done by converting it to a single zero vector similarly
    to how it is done inside WhiteSpaceTokenizer and NoopTokenizer.

    Args:
        condition (ConditionType): a tuple of condition and mask (tp.Tuple[Tensor, Tensor])
        dim (int): the dimension that will be truncated (should be the time dimension)
        WARNING!: dim should not be the batch dimension!
    Returns:
        ConditionType: a tuple of null condition and mask
    """
    assert dim != 0, "dim cannot be the batch dimension!"
    assert type(condition) == tuple and \
        type(condition[0]) == Tensor and \
        type(condition[1]) == Tensor, "'nullify_condition' got an unexpected input type!"
    cond, mask = condition
    B = cond.shape[0]
    last_dim = cond.dim() - 1
    out = cond.transpose(dim, last_dim)
    out = 0. * out[..., :1]
    out = out.transpose(dim, last_dim)
    mask = torch.zeros((B, 1), device=out.device).int()
    assert cond.dim() == out.dim()
    return out, mask


def nullify_wav(wav: Tensor) -> WavCondition:
    """Create a nullified WavCondition from a wav tensor with appropriate shape.

    Args:
        wav (Tensor): tensor of shape [B, T]
    Returns:
        WavCondition: wav condition with nullified wav.
    """
    null_wav, _ = nullify_condition((wav, torch.zeros_like(wav)), dim=wav.dim() - 1)
    return WavCondition(
        wav=null_wav,
        length=torch.tensor([0] * wav.shape[0], device=wav.device),
        path=['null_wav'] * wav.shape[0]
    )


@dataclass
class ConditioningAttributes:
    text: tp.Dict[str, tp.Optional[str]] = field(default_factory=dict)
    wav: tp.Dict[str, WavCondition] = field(default_factory=dict)

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def text_attributes(self):
        return self.text.keys()

    @property
    def wav_attributes(self):
        return self.wav.keys()

    @property
    def attributes(self):
        return {"text": self.text_attributes, "wav": self.wav_attributes}

    def to_flat_dict(self):
        return {
            **{f"text.{k}": v for k, v in self.text.items()},
            **{f"wav.{k}": v for k, v in self.wav.items()},
        }

    @classmethod
    def from_flat_dict(cls, x):
        out = cls()
        for k, v in x.items():
            kind, att = k.split(".")
            out[kind][att] = v
        return out


class SegmentWithAttributes(SegmentInfo):
    """Base class for all dataclasses that are used for conditioning.
    All child classes should implement `to_condition_attributes` that converts
    the existing attributes to a dataclass of type ConditioningAttributes.
    """
    def to_condition_attributes(self) -> ConditioningAttributes:
        raise NotImplementedError()


class Tokenizer:
    """Base class for all tokenizers
    (in case we want to introduce more advances tokenizers in the future).
    """
    def __call__(self, texts: tp.List[tp.Optional[str]]) -> tp.Tuple[Tensor, Tensor]:
        raise NotImplementedError()


class WhiteSpaceTokenizer(Tokenizer):
    """This tokenizer should be used for natural language descriptions.
    For example:
    ["he didn't, know he's going home.", 'shorter sentence'] =>
    [[78, 62, 31,  4, 78, 25, 19, 34],
    [59, 77,  0,  0,  0,  0,  0,  0]]
    """
    PUNCTUATIONS = "?:!.,;"

    def __init__(self, n_bins: int, pad_idx: int = 0, language: str = "en_core_web_sm",
                 lemma: bool = True, stopwords: bool = True) -> None:
        self.n_bins = n_bins
        self.pad_idx = pad_idx
        self.lemma = lemma
        self.stopwords = stopwords
        try:
            self.nlp = spacy.load(language)
        except IOError:
            spacy.cli.download(language)  # type: ignore
            self.nlp = spacy.load(language)

    @tp.no_type_check
    def __call__(
        self,
        texts: tp.List[tp.Optional[str]],
        return_text: bool = False
    ) -> tp.Tuple[Tensor, Tensor]:
        """Take a list of strings and convert them to a tensor of indices.

        Args:
            texts (tp.List[str]): List of strings.
            return_text (bool, optional): Whether to return text as additional tuple item. Defaults to False.
        Returns:
            tp.Tuple[Tensor, Tensor]:
                - Indices of words in the LUT.
                - And a mask indicating where the padding tokens are
        """
        output, lengths = [], []
        texts = deepcopy(texts)
        for i, text in enumerate(texts):
            # if current sample doesn't have a certain attribute, replace with pad token
            if text is None:
                output.append(Tensor([self.pad_idx]))
                lengths.append(0)
                continue

            # convert numbers to words
            text = re.sub(r"(\d+)", lambda x: num2words(int(x.group(0))), text)  # type: ignore
            # normalize text
            text = self.nlp(text)  # type: ignore
            # remove stopwords
            if self.stopwords:
                text = [w for w in text if not w.is_stop]  # type: ignore
            # remove punctuations
            text = [w for w in text if w.text not in self.PUNCTUATIONS]  # type: ignore
            # lemmatize if needed
            text = [getattr(t, "lemma_" if self.lemma else "text") for t in text]  # type: ignore

            texts[i] = " ".join(text)
            lengths.append(len(text))
            # convert to tensor
            tokens = Tensor([hash_trick(w, self.n_bins) for w in text])
            output.append(tokens)

        mask = length_to_mask(torch.IntTensor(lengths)).int()
        padded_output = pad_sequence(output, padding_value=self.pad_idx).int().t()
        if return_text:
            return padded_output, mask, texts  # type: ignore
        return padded_output, mask


class NoopTokenizer(Tokenizer):
    """This tokenizer should be used for global conditioners such as: artist, genre, key, etc.
    The difference between this and WhiteSpaceTokenizer is that NoopTokenizer does not split
    strings, so "Jeff Buckley" will get it's own index. Whereas WhiteSpaceTokenizer will
    split it to ["Jeff", "Buckley"] and return an index per word.

    For example:
    ["Queen", "ABBA", "Jeff Buckley"] => [43, 55, 101]
    ["Metal", "Rock", "Classical"] => [0, 223, 51]
    """
    def __init__(self, n_bins: int, pad_idx: int = 0):
        self.n_bins = n_bins
        self.pad_idx = pad_idx

    def __call__(self, texts: tp.List[tp.Optional[str]]) -> tp.Tuple[Tensor, Tensor]:
        output, lengths = [], []
        for text in texts:
            # if current sample doesn't have a certain attribute, replace with pad token
            if text is None:
                output.append(self.pad_idx)
                lengths.append(0)
            else:
                output.append(hash_trick(text, self.n_bins))
                lengths.append(1)

        tokens = torch.LongTensor(output).unsqueeze(1)
        mask = length_to_mask(torch.IntTensor(lengths)).int()
        return tokens, mask


class BaseConditioner(nn.Module):
    """Base model for all conditioner modules. We allow the output dim to be different
    than the hidden dim for two reasons: 1) keep our LUTs small when the vocab is large;
    2) make all condition dims consistent.

    Args:
        dim (int): Hidden dim of the model (text-encoder/LUT).
        output_dim (int): Output dim of the conditioner.
    """
    def __init__(self, dim, output_dim):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.output_proj = nn.Linear(dim, output_dim)

    def tokenize(self, *args, **kwargs) -> tp.Any:
        """Should be any part of the processing that will lead to a synchronization
        point, e.g. BPE tokenization with transfer to the GPU.

        The returned value will be saved and return later when calling forward().
        """
        raise NotImplementedError()

    def forward(self, inputs: tp.Any) -> ConditionType:
        """Gets input that should be used as conditioning (e.g, genre, description or a waveform).
        Outputs a ConditionType, after the input data was embedded as a dense vector.

        Returns:
            ConditionType:
                - A tensor of size [B, T, D] where B is the batch size, T is the length of the
                  output embedding and D is the dimension of the embedding.
                - And a mask indicating where the padding tokens.
        """
        raise NotImplementedError()


class TextConditioner(BaseConditioner):
    ...


class LUTConditioner(TextConditioner):
    """Lookup table TextConditioner.

    Args:
        n_bins (int): Number of bins.
        dim (int): Hidden dim of the model (text-encoder/LUT).
        output_dim (int): Output dim of the conditioner.
        tokenizer (str): Name of the tokenizer.
        pad_idx (int, optional): Index for padding token. Defaults to 0.
    """
    def __init__(self, n_bins: int, dim: int, output_dim: int, tokenizer: str, pad_idx: int = 0):
        super().__init__(dim, output_dim)
        self.embed = nn.Embedding(n_bins, dim)
        self.tokenizer: Tokenizer
        if tokenizer == "whitespace":
            self.tokenizer = WhiteSpaceTokenizer(n_bins, pad_idx=pad_idx)
        elif tokenizer == "noop":
            self.tokenizer = NoopTokenizer(n_bins, pad_idx=pad_idx)
        else:
            raise ValueError(f"unrecognized tokenizer `{tokenizer}`.")

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        device = self.embed.weight.device
        tokens, mask = self.tokenizer(x)
        tokens, mask = tokens.to(device), mask.to(device)
        return tokens, mask

    def forward(self, inputs: tp.Tuple[torch.Tensor, torch.Tensor]) -> ConditionType:
        tokens, mask = inputs
        embeds = self.embed(tokens)
        embeds = self.output_proj(embeds)
        embeds = (embeds * mask.unsqueeze(-1))
        return embeds, mask


class T5Conditioner(TextConditioner):
    """T5-based TextConditioner.

    Args:
        name (str): Name of the T5 model.
        output_dim (int): Output dim of the conditioner.
        finetune (bool): Whether to fine-tune T5 at train time.
        device (str): Device for T5 Conditioner.
        autocast_dtype (tp.Optional[str], optional): Autocast dtype.
        word_dropout (float, optional): Word dropout probability.
        normalize_text (bool, optional): Whether to apply text normalization.
    """
    MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl"]
    MODELS_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }

    def __init__(self, name: str, output_dim: int, finetune: bool, device: str,
                 autocast_dtype: tp.Optional[str] = 'float32', word_dropout: float = 0.,
                 normalize_text: bool = False):
        assert name in self.MODELS, f"unrecognized t5 model name (should in {self.MODELS})"
        super().__init__(self.MODELS_DIMS[name], output_dim)
        self.device = device
        self.name = name
        self.finetune = finetune
        self.word_dropout = word_dropout

        if autocast_dtype is None or self.device == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
            if self.device != 'cpu':
                logger.warning("T5 has no autocast, this might lead to NaN")
        else:
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            logger.info(f"T5 will be evaluated with autocast as {autocast_dtype}")
            self.autocast = TorchAutocast(enabled=True, device_type=self.device, dtype=dtype)
        # Let's disable logging temporarily because T5 will vomit some errors otherwise.
        # thanks https://gist.github.com/simon-weber/7853144
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.t5_tokenizer = T5Tokenizer.from_pretrained(name)
                t5 = T5EncoderModel.from_pretrained(name).train(mode=finetune)
            finally:
                logging.disable(previous_level)
        if finetune:
            self.t5 = t5
        else:
            # this makes sure that the t5 models is not part
            # of the saved checkpoint
            self.__dict__["t5"] = t5.to(device)

        self.normalize_text = normalize_text
        if normalize_text:
            self.text_normalizer = WhiteSpaceTokenizer(1, lemma=True, stopwords=True)

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, torch.Tensor]:
        # if current sample doesn't have a certain attribute, replace with empty string
        entries: tp.List[str] = [xi if xi is not None else "" for xi in x]
        if self.normalize_text:
            _, _, entries = self.text_normalizer(entries, return_text=True)
        if self.word_dropout > 0. and self.training:
            new_entries = []
            for entry in entries:
                words = [word for word in entry.split(" ") if random.random() >= self.word_dropout]
                new_entries.append(" ".join(words))
            entries = new_entries

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

        inputs = self.t5_tokenizer(entries, return_tensors="pt", padding=True).to(self.device)
        mask = inputs["attention_mask"]
        mask[empty_idx, :] = 0  # zero-out index where the input is non-existant
        return inputs

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        mask = inputs["attention_mask"]
        with torch.set_grad_enabled(self.finetune), self.autocast:
            embeds = self.t5(**inputs).last_hidden_state
        embeds = self.output_proj(embeds.to(self.output_proj.weight))
        embeds = (embeds * mask.unsqueeze(-1))
        return embeds, mask


class WaveformConditioner(BaseConditioner):
    """Base class for all conditioners that take a waveform as input.
    Classes that inherit must implement `_get_wav_embedding` that outputs
    a continuous tensor, and `_downsampling_factor` that returns the down-sampling
    factor of the embedding model.

    Args:
        dim (int): The internal representation dimension.
        output_dim (int): Output dimension.
        device (tp.Union[torch.device, str]): Device.
    """
    def __init__(self, dim: int, output_dim: int, device: tp.Union[torch.device, str]):
        super().__init__(dim, output_dim)
        self.device = device

    def tokenize(self, wav_length: WavCondition) -> WavCondition:
        wav, length, path = wav_length
        assert length is not None
        return WavCondition(wav.to(self.device), length.to(self.device), path)

    def _get_wav_embedding(self, wav: Tensor) -> Tensor:
        """Gets as input a wav and returns a dense vector of conditions."""
        raise NotImplementedError()

    def _downsampling_factor(self):
        """Returns the downsampling factor of the embedding model."""
        raise NotImplementedError()

    def forward(self, inputs: WavCondition) -> ConditionType:
        """
        Args:
            input (WavCondition): Tuple of (waveform, lengths).
        Returns:
            ConditionType: Dense vector representing the conditioning along with its' mask.
        """
        wav, lengths, path = inputs
        with torch.no_grad():
            embeds = self._get_wav_embedding(wav)
        embeds = embeds.to(self.output_proj.weight)
        embeds = self.output_proj(embeds)

        if lengths is not None:
            lengths = lengths / self._downsampling_factor()
            mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()  # type: ignore
        else:
            mask = torch.ones_like(embeds)
        embeds = (embeds * mask.unsqueeze(2).to(self.device))

        return embeds, mask


class ChromaStemConditioner(WaveformConditioner):
    """Chroma conditioner that uses DEMUCS to first filter out drums and bass. The is followed by
    the insight the drums and bass often dominate the chroma, leading to the chroma not containing the
    information about melody.

    Args:
        output_dim (int): Output dimension for the conditioner.
        sample_rate (int): Sample rate for the chroma extractor.
        n_chroma (int): Number of chroma for the chroma extractor.
        radix2_exp (int): Radix2 exponent for the chroma extractor.
        duration (float): Duration used during training. This is later used for correct padding
            in case we are using chroma as prefix.
        match_len_on_eval (bool, optional): If True then all chromas are padded to the training
            duration. Defaults to False.
        eval_wavs (str, optional): Path to a json egg with waveform, this waveforms are used as
            conditions during eval (for cases where we don't want to leak test conditions like MusicCaps).
            Defaults to None.
        n_eval_wavs (int, optional): Limits the number of waveforms used for conditioning. Defaults to 0.
        device (tp.Union[torch.device, str], optional): Device for the conditioner.
        **kwargs: Additional parameters for the chroma extractor.
    """
    def __init__(self, output_dim: int, sample_rate: int, n_chroma: int, radix2_exp: int,
                 duration: float, match_len_on_eval: bool = True, eval_wavs: tp.Optional[str] = None,
                 n_eval_wavs: int = 0, device: tp.Union[torch.device, str] = "cpu", **kwargs):
        from demucs import pretrained
        super().__init__(dim=n_chroma, output_dim=output_dim, device=device)
        self.autocast = TorchAutocast(enabled=device != "cpu", device_type=self.device, dtype=torch.float32)
        self.sample_rate = sample_rate
        self.match_len_on_eval = match_len_on_eval
        self.duration = duration
        self.__dict__["demucs"] = pretrained.get_model('htdemucs').to(device)
        self.stem2idx = {'drums': 0, 'bass': 1, 'other': 2, 'vocal': 3}
        self.stem_idx = torch.LongTensor([self.stem2idx['vocal'], self.stem2idx['other']]).to(device)
        self.chroma = ChromaExtractor(sample_rate=sample_rate, n_chroma=n_chroma, radix2_exp=radix2_exp,
                                      device=device, **kwargs)
        self.chroma_len = self._get_chroma_len()

    def _downsampling_factor(self):
        return self.chroma.winhop

    def _get_chroma_len(self):
        """Get length of chroma during training"""
        dummy_wav = torch.zeros((1, self.sample_rate * self.duration), device=self.device)
        dummy_chr = self.chroma(dummy_wav)
        return dummy_chr.shape[1]

    @torch.no_grad()
    def _get_filtered_wav(self, wav):
        from demucs.apply import apply_model
        from demucs.audio import convert_audio
        with self.autocast:
            wav = convert_audio(wav, self.sample_rate, self.demucs.samplerate, self.demucs.audio_channels)
            stems = apply_model(self.demucs, wav, device=self.device)
            stems = stems[:, self.stem_idx]  # extract stem
            stems = stems.sum(1)  # merge extracted stems
            stems = stems.mean(1, keepdim=True)  # mono
            stems = convert_audio(stems, self.demucs.samplerate, self.sample_rate, 1)
            return stems

    @torch.no_grad()
    def _get_wav_embedding(self, wav):
        # avoid 0-size tensors when we are working with null conds
        if wav.shape[-1] == 1:
            return self.chroma(wav)
        stems = self._get_filtered_wav(wav)
        chroma = self.chroma(stems)

        if self.match_len_on_eval:
            b, t, c = chroma.shape
            if t > self.chroma_len:
                chroma = chroma[:, :self.chroma_len]
                logger.debug(f'chroma was truncated! ({t} -> {chroma.shape[1]})')
            elif t < self.chroma_len:
                # chroma = F.pad(chroma, (0, 0, 0, self.chroma_len - t))
                n_repeat = int(math.ceil(self.chroma_len / t))
                chroma = chroma.repeat(1, n_repeat, 1)
                chroma = chroma[:, :self.chroma_len]
                logger.debug(f'chroma was zero-padded! ({t} -> {chroma.shape[1]})')
        return chroma


class ChromaExtractor(nn.Module):
    """Chroma extraction class, handles chroma extraction and quantization.

    Args:
        sample_rate (int): Sample rate.
        n_chroma (int): Number of chroma to consider.
        radix2_exp (int): Radix2 exponent.
        nfft (tp.Optional[int], optional): Number of FFT.
        winlen (tp.Optional[int], optional): Window length.
        winhop (tp.Optional[int], optional): Window hop size.
        argmax (bool, optional): Whether to use argmax. Defaults to False.
        norm (float, optional): Norm for chroma normalization. Defaults to inf.
        device (tp.Union[torch.device, str], optional): Device to use. Defaults to cpu.
    """
    def __init__(self, sample_rate: int, n_chroma: int = 12, radix2_exp: int = 12,
                 nfft: tp.Optional[int] = None, winlen: tp.Optional[int] = None, winhop: tp.Optional[int] = None,
                 argmax: bool = False, norm: float = torch.inf, device: tp.Union[torch.device, str] = "cpu"):
        super().__init__()
        from librosa import filters
        self.device = device
        self.autocast = TorchAutocast(enabled=device != "cpu", device_type=self.device, dtype=torch.float32)
        self.winlen = winlen or 2 ** radix2_exp
        self.nfft = nfft or self.winlen
        self.winhop = winhop or (self.winlen // 4)
        self.sr = sample_rate
        self.n_chroma = n_chroma
        self.norm = norm
        self.argmax = argmax
        self.window = torch.hann_window(self.winlen).to(device)
        self.fbanks = torch.from_numpy(filters.chroma(sr=sample_rate, n_fft=self.nfft, tuning=0,
                                                      n_chroma=self.n_chroma)).to(device)
        self.spec = torchaudio.transforms.Spectrogram(n_fft=self.nfft, win_length=self.winlen,
                                                      hop_length=self.winhop, power=2, center=True,
                                                      pad=0, normalized=True).to(device)

    def forward(self, wav):
        with self.autocast:
            T = wav.shape[-1]
            # in case we are getting a wav that was dropped out (nullified)
            # make sure wav length is no less that nfft
            if T < self.nfft:
                pad = self.nfft - T
                r = 0 if pad % 2 == 0 else 1
                wav = F.pad(wav, (pad // 2, pad // 2 + r), 'constant', 0)
                assert wav.shape[-1] == self.nfft, f'expected len {self.nfft} but got {wav.shape[-1]}'
            spec = self.spec(wav).squeeze(1)
            raw_chroma = torch.einsum("cf,...ft->...ct", self.fbanks, spec)
            norm_chroma = torch.nn.functional.normalize(raw_chroma, p=self.norm, dim=-2, eps=1e-6)
            norm_chroma = rearrange(norm_chroma, "b d t -> b t d")

            if self.argmax:
                idx = norm_chroma.argmax(-1, keepdims=True)
                norm_chroma[:] = 0
                norm_chroma.scatter_(dim=-1, index=idx, value=1)

            return norm_chroma


def dropout_condition(sample: ConditioningAttributes, condition_type: str, condition: str):
    """Utility function for nullifying an attribute inside an ConditioningAttributes object.
    If the condition is of type "wav", then nullify it using "nullify_condition".
    If the condition is of any other type, set its' value to None.
    Works in-place.
    """
    if condition_type not in ["text", "wav"]:
        raise ValueError(
            "dropout_condition got an unexpected condition type!"
            f" expected 'wav' or 'text' but got '{condition_type}'"
        )

    if condition not in getattr(sample, condition_type):
        raise ValueError(
            "dropout_condition received an unexpected condition!"
            f" expected wav={sample.wav.keys()} and text={sample.text.keys()}"
            f"but got '{condition}' of type '{condition_type}'!"
        )

    if condition_type == "wav":
        wav, length, path = sample.wav[condition]
        sample.wav[condition] = nullify_wav(wav)
    else:
        sample.text[condition] = None

    return sample


class DropoutModule(nn.Module):
    """Base class for all dropout modules."""
    def __init__(self, seed: int = 1234):
        super().__init__()
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)


class AttributeDropout(DropoutModule):
    """Applies dropout with a given probability per attribute. This is different from the behavior of
    ClassifierFreeGuidanceDropout as this allows for attributes to be dropped out separately. For example,
    "artist" can be dropped while "genre" remains. This is in contrast to ClassifierFreeGuidanceDropout
    where if "artist" is dropped "genre" must also be dropped.

    Args:
        p (tp.Dict[str, float]): A dict mapping between attributes and dropout probability. For example:
            ...
            "genre": 0.1,
            "artist": 0.5,
            "wav": 0.25,
            ...
        active_on_eval (bool, optional): Whether the dropout is active at eval. Default to False.
        seed (int, optional): Random seed.
    """
    def __init__(self, p: tp.Dict[str, tp.Dict[str, float]], active_on_eval: bool = False, seed: int = 1234):
        super().__init__(seed=seed)
        self.active_on_eval = active_on_eval
        # construct dict that return the values from p otherwise 0
        self.p = {}
        for condition_type, probs in p.items():
            self.p[condition_type] = defaultdict(lambda: 0, probs)

    def forward(self, samples: tp.List[ConditioningAttributes]) -> tp.List[ConditioningAttributes]:
        """
        Args:
            samples (tp.List[ConditioningAttributes]): List of conditions.
        Returns:
            tp.List[ConditioningAttributes]: List of conditions after certain attributes were set to None.
        """
        if not self.training and not self.active_on_eval:
            return samples

        samples = deepcopy(samples)

        for condition_type, ps in self.p.items():  # for condition types [text, wav]
            for condition, p in ps.items():  # for attributes of each type (e.g., [artist, genre])
                if torch.rand(1, generator=self.rng).item() < p:
                    for sample in samples:
                        dropout_condition(sample, condition_type, condition)

        return samples

    def __repr__(self):
        return f"AttributeDropout({dict(self.p)})"


class ClassifierFreeGuidanceDropout(DropoutModule):
    """Applies Classifier Free Guidance dropout, meaning all attributes
    are dropped with the same probability.

    Args:
        p (float): Probability to apply condition dropout during training.
        seed (int): Random seed.
    """
    def __init__(self, p: float, seed: int = 1234):
        super().__init__(seed=seed)
        self.p = p

    def forward(self, samples: tp.List[ConditioningAttributes]) -> tp.List[ConditioningAttributes]:
        """
        Args:
            samples (tp.List[ConditioningAttributes]): List of conditions.
        Returns:
            tp.List[ConditioningAttributes]: List of conditions after all attributes were set to None.
        """
        if not self.training:
            return samples

        # decide on which attributes to drop in a batched fashion
        drop = torch.rand(1, generator=self.rng).item() < self.p
        if not drop:
            return samples

        # nullify conditions of all attributes
        samples = deepcopy(samples)

        for condition_type in ["wav", "text"]:
            for sample in samples:
                for condition in sample.attributes[condition_type]:
                    dropout_condition(sample, condition_type, condition)

        return samples

    def __repr__(self):
        return f"ClassifierFreeGuidanceDropout(p={self.p})"


class ConditioningProvider(nn.Module):
    """Main class to provide conditions given all the supported conditioners.

    Args:
        conditioners (dict): Dictionary of conditioners.
        merge_text_conditions_p (float, optional): Probability to merge all text sources
            into a single text condition. Defaults to 0.
        drop_desc_p (float, optional): Probability to drop the original description
            when merging all text sources into a single text condition. Defaults to 0.
        device (tp.Union[torch.device, str], optional): Device for conditioners and output condition types.
    """
    def __init__(
        self,
        conditioners: tp.Dict[str, BaseConditioner],
        merge_text_conditions_p: float = 0,
        drop_desc_p: float = 0,
        device: tp.Union[torch.device, str] = "cpu",
    ):
        super().__init__()
        self.device = device
        self.merge_text_conditions_p = merge_text_conditions_p
        self.drop_desc_p = drop_desc_p
        self.conditioners = nn.ModuleDict(conditioners)

    @property
    def text_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, TextConditioner)]

    @property
    def wav_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, WaveformConditioner)]

    @property
    def has_wav_condition(self):
        return len(self.wav_conditions) > 0

    def tokenize(self, inputs: tp.List[ConditioningAttributes]) -> tp.Dict[str, tp.Any]:
        """Match attributes/wavs with existing conditioners in self, and compute tokenize them accordingly.
        This should be called before starting any real GPU work to avoid synchronization points.
        This will return a dict matching conditioner names to their arbitrary tokenized representations.

        Args:
            inputs (list[ConditioningAttribres]): List of ConditioningAttributes objects containing
                text and wav conditions.
        """
        assert all([type(x) == ConditioningAttributes for x in inputs]), \
            "got unexpected types input for conditioner! should be tp.List[ConditioningAttributes]" \
            f" but types were {set([type(x) for x in inputs])}"

        output = {}
        text = self._collate_text(inputs)
        wavs = self._collate_wavs(inputs)

        assert set(text.keys() | wavs.keys()).issubset(set(self.conditioners.keys())), \
            f"got an unexpected attribute! Expected {self.conditioners.keys()}, got {text.keys(), wavs.keys()}"

        for attribute, batch in chain(text.items(), wavs.items()):
            output[attribute] = self.conditioners[attribute].tokenize(batch)
        return output

    def forward(self, tokenized: tp.Dict[str, tp.Any]) -> tp.Dict[str, ConditionType]:
        """Compute pairs of `(embedding, mask)` using the configured conditioners
        and the tokenized representations. The output is for example:

            {
                "genre": (torch.Tensor([B, 1, D_genre]), torch.Tensor([B, 1])),
                "description": (torch.Tensor([B, T_desc, D_desc]), torch.Tensor([B, T_desc])),
                ...
            }

        Args:
            tokenized (dict): Dict of tokenized representations as returned by `tokenize()`.
        """
        output = {}
        for attribute, inputs in tokenized.items():
            condition, mask = self.conditioners[attribute](inputs)
            output[attribute] = (condition, mask)
        return output

    def _collate_text(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, tp.List[tp.Optional[str]]]:
        """Given a list of ConditioningAttributes objects, compile a dictionary where the keys
        are the attributes and the values are the aggregated input per attribute.
        For example:
        Input:
        [
            ConditioningAttributes(text={"genre": "Rock", "description": "A rock song with a guitar solo"}, wav=...),
            ConditioningAttributes(text={"genre": "Hip-hop", "description": "A hip-hop verse"}, wav=...),
        ]
        Output:
        {
            "genre": ["Rock", "Hip-hop"],
            "description": ["A rock song with a guitar solo", "A hip-hop verse"]
        }
        """
        batch_per_attribute: tp.Dict[str, tp.List[tp.Optional[str]]] = defaultdict(list)

        def _merge_conds(cond, merge_text_conditions_p=0, drop_desc_p=0):
            def is_valid(k, v):
                k_valid = k in ['key', 'bpm', 'genre', 'moods', 'instrument']
                v_valid = v is not None and isinstance(v, (int, float, str, list))
                return k_valid and v_valid

            def process_value(v):
                if isinstance(v, (int, float, str)):
                    return v
                if isinstance(v, list):
                    return ", ".join(v)
                else:
                    RuntimeError(f"unknown type for text value! ({type(v), v})")

            desc = cond.text['description']
            meta_data = ""
            if random.uniform(0, 1) < merge_text_conditions_p:
                meta_pairs = [f'{k}: {process_value(v)}' for k, v in cond.text.items() if is_valid(k, v)]
                random.shuffle(meta_pairs)
                meta_data = ". ".join(meta_pairs)
                desc = desc if not random.uniform(0, 1) < drop_desc_p else None

            if desc is None:
                desc = meta_data if len(meta_data) > 1 else None
            else:
                desc = desc.rstrip('.') + ". " + meta_data
            cond.text['description'] = desc.strip() if desc else None

        if self.training and self.merge_text_conditions_p:
            for sample in samples:
                _merge_conds(sample, self.merge_text_conditions_p, self.drop_desc_p)

        texts = [x.text for x in samples]
        for text in texts:
            for condition in self.text_conditions:
                batch_per_attribute[condition].append(text[condition])

        return batch_per_attribute

    def _collate_wavs(self, samples: tp.List[ConditioningAttributes]):
        """Generate a dict where the keys are attributes by which we fetch similar wavs,
        and the values are Tensors of wavs according to said attribtues.

        *Note*: by the time the samples reach this function, each sample should have some waveform
        inside the "wav" attribute. It should be either:
        1. A real waveform
        2. A null waveform due to the sample having no similar waveforms (nullified by the dataset)
        3. A null waveform due to it being dropped in a dropout module (nullified by dropout)

        Args:
            samples (tp.List[ConditioningAttributes]): List of ConditioningAttributes samples.
        Returns:
            dict: A dicionary mapping an attribute name to wavs.
        """
        wavs = defaultdict(list)
        lens = defaultdict(list)
        paths = defaultdict(list)
        out = {}

        for sample in samples:
            for attribute in self.wav_conditions:
                wav, length, path = sample.wav[attribute]
                wavs[attribute].append(wav.flatten())
                lens[attribute].append(length)
                paths[attribute].append(path)

        # stack all wavs to a single tensor
        for attribute in self.wav_conditions:
            stacked_wav, _ = collate(wavs[attribute], dim=0)
            out[attribute] = WavCondition(stacked_wav.unsqueeze(1),
                                          torch.cat(lens['self_wav']), paths[attribute])  # type: ignore

        return out


class ConditionFuser(StreamingModule):
    """Condition fuser handles the logic to combine the different conditions
    to the actual model input.

    Args:
        fuse2cond (tp.Dict[str, str]): A dictionary that says how to fuse
            each condition. For example:
            {
                "prepend": ["description"],
                "sum": ["genre", "bpm"],
                "cross": ["description"],
            }
        cross_attention_pos_emb (bool, optional): Use positional embeddings in cross attention.
        cross_attention_pos_emb_scale (int): Scale for positional embeddings in cross attention if used.
    """
    FUSING_METHODS = ["sum", "prepend", "cross", "input_interpolate"]

    def __init__(self, fuse2cond: tp.Dict[str, tp.List[str]], cross_attention_pos_emb: bool = False,
                 cross_attention_pos_emb_scale: float = 1.0):
        super().__init__()
        assert all(
            [k in self.FUSING_METHODS for k in fuse2cond.keys()]
        ), f"got invalid fuse method, allowed methods: {self.FUSING_MEHTODS}"
        self.cross_attention_pos_emb = cross_attention_pos_emb
        self.cross_attention_pos_emb_scale = cross_attention_pos_emb_scale
        self.fuse2cond: tp.Dict[str, tp.List[str]] = fuse2cond
        self.cond2fuse: tp.Dict[str, str] = {}
        for fuse_method, conditions in fuse2cond.items():
            for condition in conditions:
                self.cond2fuse[condition] = fuse_method

    def forward(
        self,
        input: Tensor,
        conditions: tp.Dict[str, ConditionType]
    ) -> tp.Tuple[Tensor, tp.Optional[Tensor]]:
        """Fuse the conditions to the provided model input.

        Args:
            input (Tensor): Transformer input.
            conditions (tp.Dict[str, ConditionType]): Dict of conditions.
        Returns:
            tp.Tuple[Tensor, Tensor]: The first tensor is the transformer input
                after the conditions have been fused. The second output tensor is the tensor
                used for cross-attention or None if no cross attention inputs exist.
        """
        B, T, _ = input.shape

        if 'offsets' in self._streaming_state:
            first_step = False
            offsets = self._streaming_state['offsets']
        else:
            first_step = True
            offsets = torch.zeros(input.shape[0], dtype=torch.long, device=input.device)

        assert set(conditions.keys()).issubset(set(self.cond2fuse.keys())), \
            f"given conditions contain unknown attributes for fuser, " \
            f"expected {self.cond2fuse.keys()}, got {conditions.keys()}"
        cross_attention_output = None
        for cond_type, (cond, cond_mask) in conditions.items():
            op = self.cond2fuse[cond_type]
            if op == "sum":
                input += cond
            elif op == "input_interpolate":
                cond = rearrange(cond, "b t d -> b d t")
                cond = F.interpolate(cond, size=input.shape[1])
                input += rearrange(cond, "b d t -> b t d")
            elif op == "prepend":
                if first_step:
                    input = torch.cat([cond, input], dim=1)
            elif op == "cross":
                if cross_attention_output is not None:
                    cross_attention_output = torch.cat([cross_attention_output, cond], dim=1)
                else:
                    cross_attention_output = cond
            else:
                raise ValueError(f"unknown op ({op})")

        if self.cross_attention_pos_emb and cross_attention_output is not None:
            positions = torch.arange(
                cross_attention_output.shape[1],
                device=cross_attention_output.device
            ).view(1, -1, 1)
            pos_emb = create_sin_embedding(positions, cross_attention_output.shape[-1])
            cross_attention_output = cross_attention_output + self.cross_attention_pos_emb_scale * pos_emb

        if self._is_streaming:
            self._streaming_state['offsets'] = offsets + T

        return input, cross_attention_output
