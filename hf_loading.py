"""Utility for loading the models from HF."""
from pathlib import Path
import typing as tp

from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
import torch

from audiocraft.models import builders, MusicGen

MODEL_CHECKPOINTS_MAP = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
    "melody": "facebook/musicgen-melody",
}


def _get_state_dict(file_or_url: tp.Union[Path, str],
                    filename="state_dict.bin", device='cpu'):
    # Return the state dict either from a file or url
    print("loading", file_or_url, filename)
    file_or_url = str(file_or_url)
    assert isinstance(file_or_url, str)
    return torch.load(
        hf_hub_download(repo_id=file_or_url, filename=filename), map_location=device)


def load_compression_model(file_or_url: tp.Union[Path, str], device='cpu'):
    pkg = _get_state_dict(file_or_url, filename="compression_state_dict.bin")
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    model = builders.get_compression_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    model.cfg = cfg
    return model


def load_lm_model(file_or_url: tp.Union[Path, str], device='cpu'):
    pkg = _get_state_dict(file_or_url)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.transformer_lm.memory_efficient = False
        cfg.transformer_lm.custom = True
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    model = builders.get_lm_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    model.cfg = cfg
    return model


def get_pretrained(name: str = 'small', device='cuda'):
    model_id = MODEL_CHECKPOINTS_MAP[name]
    compression_model = load_compression_model(model_id, device=device)
    lm = load_lm_model(model_id, device=device)
    return MusicGen(name, compression_model, lm)
