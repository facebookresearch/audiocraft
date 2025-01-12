# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
All the functions to build the relevant models and modules
from the Hydra config.
"""

import typing as tp

import omegaconf
import torch

import audiocraft

from .. import quantization as qt
from ..modules.codebooks_patterns import (CoarseFirstPattern,
                                          CodebooksPatternProvider,
                                          DelayedPatternProvider,
                                          MusicLMPattern,
                                          ParallelPatternProvider,
                                          UnrolledPatternProvider)
from ..modules.conditioners import (BaseConditioner, ChromaStemConditioner,
                                    CLAPEmbeddingConditioner,
                                    ConditionFuser, JascoCondConst,
                                    ConditioningProvider, LUTConditioner,
                                    T5Conditioner, StyleConditioner)
from ..modules.jasco_conditioners import (JascoConditioningProvider, ChordsEmbConditioner,
                                          DrumsConditioner, MelodyConditioner)
from ..modules.diffusion_schedule import MultiBandProcessor, SampleProcessor
from ..utils.utils import dict_from_config
from .encodec import (CompressionModel, EncodecModel,
                      InterleaveStereoCompressionModel)
from .lm import LMModel
from .lm_magnet import MagnetLMModel
from .flow_matching import FlowMatchingModel
from .unet import DiffusionUnet
from .watermark import WMModel


def get_quantizer(
    quantizer: str, cfg: omegaconf.DictConfig, dimension: int
) -> qt.BaseQuantizer:
    klass = {"no_quant": qt.DummyQuantizer, "rvq": qt.ResidualVectorQuantizer}[
        quantizer
    ]
    kwargs = dict_from_config(getattr(cfg, quantizer))
    if quantizer != "no_quant":
        kwargs["dimension"] = dimension
    return klass(**kwargs)


def get_encodec_autoencoder(encoder_name: str, cfg: omegaconf.DictConfig):
    if encoder_name == "seanet":
        kwargs = dict_from_config(getattr(cfg, "seanet"))
        encoder_override_kwargs = kwargs.pop("encoder")
        decoder_override_kwargs = kwargs.pop("decoder")
        encoder_kwargs = {**kwargs, **encoder_override_kwargs}
        decoder_kwargs = {**kwargs, **decoder_override_kwargs}
        encoder = audiocraft.modules.SEANetEncoder(**encoder_kwargs)
        decoder = audiocraft.modules.SEANetDecoder(**decoder_kwargs)
        return encoder, decoder
    else:
        raise KeyError(f"Unexpected compression model {cfg.compression_model}")


def get_compression_model(cfg: omegaconf.DictConfig) -> CompressionModel:
    """Instantiate a compression model."""
    if cfg.compression_model == "encodec":
        kwargs = dict_from_config(getattr(cfg, "encodec"))
        encoder_name = kwargs.pop("autoencoder")
        quantizer_name = kwargs.pop("quantizer")
        encoder, decoder = get_encodec_autoencoder(encoder_name, cfg)
        quantizer = get_quantizer(quantizer_name, cfg, encoder.dimension)
        frame_rate = kwargs["sample_rate"] // encoder.hop_length
        renormalize = kwargs.pop("renormalize", False)
        # deprecated params
        kwargs.pop("renorm", None)
        return EncodecModel(
            encoder,
            decoder,
            quantizer,
            frame_rate=frame_rate,
            renormalize=renormalize,
            **kwargs,
        ).to(cfg.device)
    else:
        raise KeyError(f"Unexpected compression model {cfg.compression_model}")


def get_jasco_model(cfg: omegaconf.DictConfig,
                    compression_model: tp.Optional[CompressionModel] = None) -> FlowMatchingModel:
    kwargs = dict_from_config(getattr(cfg, "transformer_lm"))
    attribute_dropout = dict_from_config(getattr(cfg, "attribute_dropout"))
    cls_free_guidance = dict_from_config(getattr(cfg, "classifier_free_guidance"))
    cfg_prob = cls_free_guidance["training_dropout"]
    cfg_coef = cls_free_guidance["inference_coef"]
    fuser = get_condition_fuser(cfg)
    condition_provider = get_conditioner_provider(kwargs["dim"], cfg).to(cfg.device)
    if JascoCondConst.DRM.value in condition_provider.conditioners:  # use self_wav for drums
        assert compression_model is not None

        # use compression model for drums conditioning
        condition_provider.conditioners.self_wav.compression_model = compression_model
        condition_provider.conditioners.self_wav.compression_model.requires_grad_(False)

    # downcast to jasco conditioning provider
    seq_len = cfg.compression_model_framerate * cfg.dataset.segment_duration
    chords_card = cfg.conditioners.chords.chords_emb.card if JascoCondConst.CRD.value in cfg.conditioners else -1
    condition_provider = JascoConditioningProvider(device=condition_provider.device,
                                                   conditioners=condition_provider.conditioners,
                                                   chords_card=chords_card,
                                                   sequence_length=seq_len)

    if len(fuser.fuse2cond["cross"]) > 0:  # enforce cross-att programmatically
        kwargs["cross_attention"] = True

    kwargs.pop("n_q", None)
    kwargs.pop("card", None)

    return FlowMatchingModel(
        condition_provider=condition_provider,
        fuser=fuser,
        cfg_dropout=cfg_prob,
        cfg_coef=cfg_coef,
        attribute_dropout=attribute_dropout,
        dtype=getattr(torch, cfg.dtype),
        device=cfg.device,
        **kwargs,
    ).to(cfg.device)


def get_lm_model(cfg: omegaconf.DictConfig) -> LMModel:
    """Instantiate a transformer LM."""
    if cfg.lm_model in ["transformer_lm", "transformer_lm_magnet"]:
        kwargs = dict_from_config(getattr(cfg, "transformer_lm"))
        n_q = kwargs["n_q"]
        q_modeling = kwargs.pop("q_modeling", None)
        codebooks_pattern_cfg = getattr(cfg, "codebooks_pattern")
        attribute_dropout = dict_from_config(getattr(cfg, "attribute_dropout"))
        cls_free_guidance = dict_from_config(getattr(cfg, "classifier_free_guidance"))
        cfg_prob, cfg_coef = (
            cls_free_guidance["training_dropout"],
            cls_free_guidance["inference_coef"],
        )
        fuser = get_condition_fuser(cfg)
        condition_provider = get_conditioner_provider(kwargs["dim"], cfg).to(cfg.device)
        if len(fuser.fuse2cond["cross"]) > 0:  # enforce cross-att programmatically
            kwargs["cross_attention"] = True
        if codebooks_pattern_cfg.modeling is None:
            assert (
                q_modeling is not None
            ), "LM model should either have a codebook pattern defined or transformer_lm.q_modeling"
            codebooks_pattern_cfg = omegaconf.OmegaConf.create(
                {"modeling": q_modeling, "delay": {"delays": list(range(n_q))}}
            )

        pattern_provider = get_codebooks_pattern_provider(n_q, codebooks_pattern_cfg)
        lm_class = MagnetLMModel if cfg.lm_model == "transformer_lm_magnet" else LMModel
        return lm_class(
            pattern_provider=pattern_provider,
            condition_provider=condition_provider,
            fuser=fuser,
            cfg_dropout=cfg_prob,
            cfg_coef=cfg_coef,
            attribute_dropout=attribute_dropout,
            dtype=getattr(torch, cfg.dtype),
            device=cfg.device,
            **kwargs,
        ).to(cfg.device)
    else:
        raise KeyError(f"Unexpected LM model {cfg.lm_model}")


def get_conditioner_provider(
    output_dim: int, cfg: omegaconf.DictConfig
) -> ConditioningProvider:
    """Instantiate a conditioning model."""
    device = cfg.device
    duration = cfg.dataset.segment_duration
    cfg = getattr(cfg, "conditioners")
    dict_cfg = {} if cfg is None else dict_from_config(cfg)
    conditioners: tp.Dict[str, BaseConditioner] = {}
    condition_provider_args = dict_cfg.pop("args", {})
    condition_provider_args.pop("merge_text_conditions_p", None)
    condition_provider_args.pop("drop_desc_p", None)

    for cond, cond_cfg in dict_cfg.items():
        model_type = cond_cfg["model"]
        model_args = cond_cfg[model_type]
        if model_type == "t5":
            conditioners[str(cond)] = T5Conditioner(
                output_dim=output_dim, device=device, **model_args
            )
        elif model_type == "lut":
            conditioners[str(cond)] = LUTConditioner(
                output_dim=output_dim, **model_args
            )
        elif model_type == "chroma_stem":
            conditioners[str(cond)] = ChromaStemConditioner(
                output_dim=output_dim, duration=duration, device=device, **model_args
            )
        elif model_type in {"chords_emb", "drum_latents", "melody"}:
            conditioners_classes = {"chords_emb": ChordsEmbConditioner,
                                    "drum_latents": DrumsConditioner,
                                    "melody": MelodyConditioner}
            conditioner_class = conditioners_classes[model_type]
            conditioners[str(cond)] = conditioner_class(device=device, **model_args)
        elif model_type == "clap":
            conditioners[str(cond)] = CLAPEmbeddingConditioner(
                output_dim=output_dim, device=device, **model_args
            )
        elif model_type == 'style':
            conditioners[str(cond)] = StyleConditioner(
                output_dim=output_dim,
                device=device,
                **model_args
            )
        else:
            raise ValueError(f"Unrecognized conditioning model: {model_type}")
    conditioner = ConditioningProvider(
        conditioners, device=device, **condition_provider_args
    )
    return conditioner


def get_condition_fuser(cfg: omegaconf.DictConfig) -> ConditionFuser:
    """Instantiate a condition fuser object."""
    fuser_cfg = getattr(cfg, "fuser")
    fuser_methods = ["sum", "cross", "prepend", "ignore", "input_interpolate"]
    fuse2cond = {k: fuser_cfg[k] for k in fuser_methods if k in fuser_cfg}
    kwargs = {k: v for k, v in fuser_cfg.items() if k not in fuser_methods}
    fuser = ConditionFuser(fuse2cond=fuse2cond, **kwargs)
    return fuser


def get_codebooks_pattern_provider(
    n_q: int, cfg: omegaconf.DictConfig
) -> CodebooksPatternProvider:
    """Instantiate a codebooks pattern provider object."""
    pattern_providers = {
        "parallel": ParallelPatternProvider,
        "delay": DelayedPatternProvider,
        "unroll": UnrolledPatternProvider,
        "coarse_first": CoarseFirstPattern,
        "musiclm": MusicLMPattern,
    }
    name = cfg.modeling
    kwargs = dict_from_config(cfg.get(name)) if hasattr(cfg, name) else {}
    klass = pattern_providers[name]
    return klass(n_q, **kwargs)


def get_debug_compression_model(device="cpu", sample_rate: int = 32000):
    """Instantiate a debug compression model to be used for unit tests."""
    assert sample_rate in [
        16000,
        32000,
    ], "unsupported sample rate for debug compression model"
    model_ratios = {
        16000: [10, 8, 8],  # 25 Hz at 16kHz
        32000: [10, 8, 16],  # 25 Hz at 32kHz
    }
    ratios: tp.List[int] = model_ratios[sample_rate]
    frame_rate = 25
    seanet_kwargs: dict = {
        "n_filters": 4,
        "n_residual_layers": 1,
        "dimension": 32,
        "ratios": ratios,
    }
    encoder = audiocraft.modules.SEANetEncoder(**seanet_kwargs)
    decoder = audiocraft.modules.SEANetDecoder(**seanet_kwargs)
    quantizer = qt.ResidualVectorQuantizer(dimension=32, bins=400, n_q=4)
    init_x = torch.randn(8, 32, 128)
    quantizer(init_x, 1)  # initialize kmeans etc.
    compression_model = EncodecModel(
        encoder,
        decoder,
        quantizer,
        frame_rate=frame_rate,
        sample_rate=sample_rate,
        channels=1,
    ).to(device)
    return compression_model.eval()


def get_diffusion_model(cfg: omegaconf.DictConfig):
    # TODO Find a way to infer the channels from dset
    channels = cfg.channels
    num_steps = cfg.schedule.num_steps
    return DiffusionUnet(chin=channels, num_steps=num_steps, **cfg.diffusion_unet)


def get_processor(cfg, sample_rate: int = 24000):
    sample_processor = SampleProcessor()
    if cfg.use:
        kw = dict(cfg)
        kw.pop("use")
        kw.pop("name")
        if cfg.name == "multi_band_processor":
            sample_processor = MultiBandProcessor(sample_rate=sample_rate, **kw)
    return sample_processor


def get_debug_lm_model(device="cpu"):
    """Instantiate a debug LM to be used for unit tests."""
    pattern = DelayedPatternProvider(n_q=4)
    dim = 16
    providers = {
        "description": LUTConditioner(
            n_bins=128, dim=dim, output_dim=dim, tokenizer="whitespace"
        ),
    }
    condition_provider = ConditioningProvider(providers)
    fuser = ConditionFuser(
        {"cross": ["description"], "prepend": [], "sum": [], "input_interpolate": []}
    )
    lm = LMModel(
        pattern,
        condition_provider,
        fuser,
        n_q=4,
        card=400,
        dim=dim,
        num_heads=4,
        custom=True,
        num_layers=2,
        cross_attention=True,
        causal=True,
    )
    return lm.to(device).eval()


def get_wrapped_compression_model(
    compression_model: CompressionModel, cfg: omegaconf.DictConfig
) -> CompressionModel:
    if hasattr(cfg, "interleave_stereo_codebooks"):
        if cfg.interleave_stereo_codebooks.use:
            kwargs = dict_from_config(cfg.interleave_stereo_codebooks)
            kwargs.pop("use")
            compression_model = InterleaveStereoCompressionModel(
                compression_model, **kwargs
            )
    if hasattr(cfg, "compression_model_n_q"):
        if cfg.compression_model_n_q is not None:
            compression_model.set_num_codebooks(cfg.compression_model_n_q)
    return compression_model


def get_watermark_model(cfg: omegaconf.DictConfig) -> WMModel:
    """Build a WMModel based by audioseal. This requires audioseal to be installed"""
    import audioseal

    from .watermark import AudioSeal

    # Builder encoder and decoder directly using audiocraft API to avoid cyclic import
    assert hasattr(
        cfg, "seanet"
    ), "Missing required `seanet` parameters in AudioSeal config"
    encoder, decoder = get_encodec_autoencoder("seanet", cfg)

    # Build message processor
    kwargs = (
        dict_from_config(getattr(cfg, "audioseal")) if hasattr(cfg, "audioseal") else {}
    )
    nbits = kwargs.get("nbits", 0)
    hidden_size = getattr(cfg.seanet, "dimension", 128)
    msg_processor = audioseal.MsgProcessor(nbits, hidden_size=hidden_size)

    # Build detector using audioseal API
    def _get_audioseal_detector():
        # We don't need encoder and decoder params from seanet, remove them
        seanet_cfg = dict_from_config(cfg.seanet)
        seanet_cfg.pop("encoder")
        seanet_cfg.pop("decoder")
        detector_cfg = dict_from_config(cfg.detector)

        typed_seanet_cfg = audioseal.builder.SEANetConfig(**seanet_cfg)
        typed_detector_cfg = audioseal.builder.DetectorConfig(**detector_cfg)
        _cfg = audioseal.builder.AudioSealDetectorConfig(
            nbits=nbits, seanet=typed_seanet_cfg, detector=typed_detector_cfg
        )
        return audioseal.builder.create_detector(_cfg)

    detector = _get_audioseal_detector()
    generator = audioseal.AudioSealWM(
        encoder=encoder, decoder=decoder, msg_processor=msg_processor
    )
    model = AudioSeal(generator=generator, detector=detector, nbits=nbits)

    device = torch.device(getattr(cfg, "device", "cpu"))
    dtype = getattr(torch, getattr(cfg, "dtype", "float32"))
    return model.to(device=device, dtype=dtype)
