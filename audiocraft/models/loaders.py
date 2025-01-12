# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions to load from the checkpoints.
Each checkpoint is a torch.saved dict with the following keys:
- 'xp.cfg': the hydra config as dumped during training. This should be used
    to rebuild the object using the audiocraft.models.builders functions,
- 'model_best_state': a readily loadable best state for the model, including
    the conditioner. The model obtained from `xp.cfg` should be compatible
    with this state dict. In the case of a LM, the encodec model would not be
    bundled along but instead provided separately.

Those functions also support loading from a remote location with the Torch Hub API.
They also support overriding some parameters, in particular the device and dtype
of the returned model.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download
import typing as tp
import os

from omegaconf import OmegaConf, DictConfig
import torch

import audiocraft

from . import builders
from .encodec import CompressionModel


def get_audiocraft_cache_dir() -> tp.Optional[str]:
    return os.environ.get('AUDIOCRAFT_CACHE_DIR', None)


def _get_state_dict(
    file_or_url_or_id: tp.Union[Path, str],
    filename: tp.Optional[str] = None,
    device='cpu',
    cache_dir: tp.Optional[str] = None,
):
    if cache_dir is None:
        cache_dir = get_audiocraft_cache_dir()
    # Return the state dict either from a file or url
    file_or_url_or_id = str(file_or_url_or_id)
    assert isinstance(file_or_url_or_id, str)

    if os.path.isfile(file_or_url_or_id):
        return torch.load(file_or_url_or_id, map_location=device)

    if os.path.isdir(file_or_url_or_id):
        file = f"{file_or_url_or_id}/{filename}"
        return torch.load(file, map_location=device)

    elif file_or_url_or_id.startswith('https://'):
        return torch.hub.load_state_dict_from_url(file_or_url_or_id, map_location=device, check_hash=True)

    else:
        assert filename is not None, "filename needs to be defined if using HF checkpoints"
        file = hf_hub_download(
            repo_id=file_or_url_or_id,
            filename=filename,
            cache_dir=cache_dir,
            library_name="audiocraft",
            library_version=audiocraft.__version__,
        )
        return torch.load(file, map_location=device)


def load_compression_model_ckpt(file_or_url_or_id: tp.Union[Path, str], cache_dir: tp.Optional[str] = None):
    return _get_state_dict(file_or_url_or_id, filename="compression_state_dict.bin", cache_dir=cache_dir)


def load_compression_model(
    file_or_url_or_id: tp.Union[Path, str],
    device="cpu",
    cache_dir: tp.Optional[str] = None,
):
    pkg = load_compression_model_ckpt(file_or_url_or_id, cache_dir=cache_dir)
    if 'pretrained' in pkg:
        return CompressionModel.get_pretrained(pkg['pretrained'], device=device)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    model = builders.get_compression_model(cfg)
    model.load_state_dict(pkg["best_state"])
    model.eval()
    return model


def load_lm_model_ckpt(file_or_url_or_id: tp.Union[Path, str], cache_dir: tp.Optional[str] = None):
    return _get_state_dict(file_or_url_or_id, filename="state_dict.bin", cache_dir=cache_dir)


def _delete_param(cfg: DictConfig, full_name: str):
    parts = full_name.split('.')
    for part in parts[:-1]:
        if part in cfg:
            cfg = cfg[part]
        else:
            return
    OmegaConf.set_struct(cfg, False)
    if parts[-1] in cfg:
        del cfg[parts[-1]]
    OmegaConf.set_struct(cfg, True)


def load_lm_model(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = load_lm_model_ckpt(file_or_url_or_id, cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    _delete_param(cfg, 'conditioners.self_wav.chroma_stem.cache_path')
    _delete_param(cfg, 'conditioners.args.merge_text_conditions_p')
    _delete_param(cfg, 'conditioners.args.drop_desc_p')
    model = builders.get_lm_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    model.cfg = cfg
    return model


def load_lm_model_magnet(file_or_url_or_id: tp.Union[Path, str], compression_model_frame_rate: int,
                         device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = load_lm_model_ckpt(file_or_url_or_id, cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    _delete_param(cfg, 'conditioners.args.merge_text_conditions_p')
    _delete_param(cfg, 'conditioners.args.drop_desc_p')

    cfg.transformer_lm.compression_model_framerate = compression_model_frame_rate
    cfg.transformer_lm.segment_duration = cfg.dataset.segment_duration
    cfg.transformer_lm.span_len = cfg.masking.span_len

    # MAGNeT models v1 support only xformers backend.
    from audiocraft.modules.transformer import set_efficient_attention_backend

    if cfg.transformer_lm.memory_efficient:
        set_efficient_attention_backend("xformers")

    model = builders.get_lm_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    model.cfg = cfg
    return model


def load_jasco_model(file_or_url_or_id: tp.Union[Path, str],
                     compression_model: CompressionModel,
                     device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = load_lm_model_ckpt(file_or_url_or_id, cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    model = builders.get_jasco_model(cfg, compression_model)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    model.cfg = cfg
    return model


def load_mbd_ckpt(file_or_url_or_id: tp.Union[Path, str],
                  filename: tp.Optional[str] = None,
                  cache_dir: tp.Optional[str] = None):
    return _get_state_dict(file_or_url_or_id, filename=filename, cache_dir=cache_dir)


def load_diffusion_models(file_or_url_or_id: tp.Union[Path, str],
                          device='cpu',
                          filename: tp.Optional[str] = None,
                          cache_dir: tp.Optional[str] = None):
    pkg = load_mbd_ckpt(file_or_url_or_id, filename=filename, cache_dir=cache_dir)
    models = []
    processors = []
    cfgs = []
    sample_rate = pkg['sample_rate']
    for i in range(pkg['n_bands']):
        cfg = pkg[i]['cfg']
        model = builders.get_diffusion_model(cfg)
        model_dict = pkg[i]['model_state']
        model.load_state_dict(model_dict)
        model.to(device)
        processor = builders.get_processor(cfg=cfg.processor, sample_rate=sample_rate)
        processor_dict = pkg[i]['processor_state']
        processor.load_state_dict(processor_dict)
        processor.to(device)
        models.append(model)
        processors.append(processor)
        cfgs.append(cfg)
    return models, processors, cfgs


def load_audioseal_models(
    file_or_url_or_id: tp.Union[Path, str],
    device="cpu",
    filename: tp.Optional[str] = None,
    cache_dir: tp.Optional[str] = None,
):

    detector_ckpt = _get_state_dict(
        file_or_url_or_id,
        filename=f"detector_{filename}.pth",
        device=device,
        cache_dir=cache_dir,
    )
    assert (
        "model" in detector_ckpt
    ), f"No model state dict found in {file_or_url_or_id}/detector_{filename}.pth"
    detector_state = detector_ckpt["model"]

    generator_ckpt = _get_state_dict(
        file_or_url_or_id,
        filename=f"generator_{filename}.pth",
        device=device,
        cache_dir=cache_dir,
    )
    assert (
        "model" in generator_ckpt
    ), f"No model state dict found in {file_or_url_or_id}/generator_{filename}.pth"
    generator_state = generator_ckpt["model"]

    def load_model_config():
        if Path(file_or_url_or_id).joinpath(f"{filename}.yaml").is_file():
            return OmegaConf.load(Path(file_or_url_or_id).joinpath(f"{filename}.yaml"))
        elif file_or_url_or_id.startswith("https://"):
            import requests  # type: ignore

            resp = requests.get(f"{file_or_url_or_id}/{filename}.yaml")
            return OmegaConf.create(resp.text)
        else:
            file = hf_hub_download(
                repo_id=file_or_url_or_id,
                filename=f"{filename}.yaml",
                cache_dir=cache_dir,
                library_name="audiocraft",
                library_version=audiocraft.__version__,
            )
            return OmegaConf.load(file)

    try:
        cfg = load_model_config()
    except Exception as exc:  # noqa
        cfg_fp = (
            Path(__file__)
            .parents[2]
            .joinpath("config", "model", "watermark", "default.yaml")
        )
        cfg = OmegaConf.load(cfg_fp)

    OmegaConf.resolve(cfg)
    model = builders.get_watermark_model(cfg)

    model.generator.load_state_dict(generator_state)
    model.detector.load_state_dict(detector_state)
    return model.to(device)
