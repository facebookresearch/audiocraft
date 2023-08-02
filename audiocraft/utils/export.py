# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility to export a training checkpoint to a lightweight release checkpoint.
"""

from pathlib import Path
import typing as tp

from omegaconf import OmegaConf
import torch

from audiocraft import __version__


def export_encodec(checkpoint_path: tp.Union[Path, str], out_file: tp.Union[Path, str]):
    """Export only the best state from the given EnCodec checkpoint. This
    should be used if you trained your own EnCodec model.
    """
    pkg = torch.load(checkpoint_path, 'cpu')
    new_pkg = {
        'best_state': pkg['best_state']['model'],
        'xp.cfg': OmegaConf.to_yaml(pkg['xp.cfg']),
        'version': __version__,
        'exported': True,
    }
    Path(out_file).parent.mkdir(exist_ok=True, parents=True)
    torch.save(new_pkg, out_file)
    return out_file


def export_pretrained_compression_model(pretrained_encodec: str, out_file: tp.Union[Path, str]):
    """Export a compression model (potentially EnCodec) from a pretrained model.
    This is required for packaging the audio tokenizer along a MusicGen or AudioGen model.
    Do not include the //pretrained/ prefix. For instance if you trained a model
    with `facebook/encodec_32khz`, just put that as a name. Same for `dac_44khz`.

    In that case, this will not actually include a copy of the model, simply the reference
    to the model used.
    """
    if Path(pretrained_encodec).exists():
        pkg = torch.load(pretrained_encodec)
        assert 'best_state' in pkg
        assert 'xp.cfg' in pkg
        assert 'version' in pkg
        assert 'exported' in pkg
    else:
        pkg = {
            'pretrained': pretrained_encodec,
            'exported': True,
            'version': __version__,
        }
    Path(out_file).parent.mkdir(exist_ok=True, parents=True)
    torch.save(pkg, out_file)


def export_lm(checkpoint_path: tp.Union[Path, str], out_file: tp.Union[Path, str]):
    """Export only the best state from the given MusicGen or AudioGen checkpoint.
    """
    pkg = torch.load(checkpoint_path, 'cpu')
    if pkg['fsdp_best_state']:
        best_state = pkg['fsdp_best_state']['model']
    else:
        assert pkg['best_state']
        best_state = pkg['best_state']['model']
    new_pkg = {
        'best_state': best_state,
        'xp.cfg': OmegaConf.to_yaml(pkg['xp.cfg']),
        'version': __version__,
        'exported': True,
    }

    Path(out_file).parent.mkdir(exist_ok=True, parents=True)
    torch.save(new_pkg, out_file)
    return out_file
