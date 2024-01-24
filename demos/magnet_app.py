# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under thmage license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
import os
from pathlib import Path
import subprocess as sp
import sys
from tempfile import NamedTemporaryFile
import time
import typing as tp
import warnings

import gradio as gr

from audiocraft.data.audio import audio_write
from audiocraft.models import MAGNeT


MODEL = None  # Last used model
SPACE_ID = os.environ.get('SPACE_ID', '')
MAX_BATCH_SIZE = 12
N_REPEATS = 2
INTERRUPTING = False
MBD = None
# We have to wrap subprocess call to clean a bit the log when using gr.make_waveform
_old_call = sp.call

PROD_STRIDE_1 = "prod-stride1 (new!)"


def _call_nostderr(*args, **kwargs):
    # Avoid ffmpeg vomiting on the logs.
    kwargs['stderr'] = sp.DEVNULL
    kwargs['stdout'] = sp.DEVNULL
    _old_call(*args, **kwargs)


sp.call = _call_nostderr
# Preallocating the pool of processes.
pool = ProcessPoolExecutor(4)
pool.__enter__()


def interrupt():
    global INTERRUPTING
    INTERRUPTING = True


class FileCleaner:
    def __init__(self, file_lifetime: float = 3600):
        self.file_lifetime = file_lifetime
        self.files = []

    def add(self, path: tp.Union[str, Path]):
        self._cleanup()
        self.files.append((time.time(), Path(path)))

    def _cleanup(self):
        now = time.time()
        for time_added, path in list(self.files):
            if now - time_added > self.file_lifetime:
                if path.exists():
                    path.unlink()
                self.files.pop(0)
            else:
                break


file_cleaner = FileCleaner()


def make_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out


def load_model(version='facebook/magnet-small-10secs'):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        MODEL = None  # in case loading would crash
        MODEL = MAGNeT.get_pretrained(version)


def _do_predictions(texts, progress=False, gradio_progress=None, **gen_kwargs):
    MODEL.set_generation_params(**gen_kwargs)
    print("new batch", len(texts), texts)
    be = time.time()

    try:
        outputs = MODEL.generate(texts, progress=progress, return_tokens=False)
    except RuntimeError as e:
        raise gr.Error("Error while generating " + e.args[0])
    outputs = outputs.detach().cpu().float()
    pending_videos = []
    out_wavs = []
    for i, output in enumerate(outputs):
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, MODEL.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
            if i == 0:
                pending_videos.append(pool.submit(make_waveform, file.name))
            out_wavs.append(file.name)
            file_cleaner.add(file.name)
    out_videos = [pending_video.result() for pending_video in pending_videos]
    for video in out_videos:
        file_cleaner.add(video)
    print("batch finished", len(texts), time.time() - be)
    print("Tempfiles currently stored: ", len(file_cleaner.files))
    return out_videos, out_wavs


def predict_batched(texts, melodies):
    max_text_length = 512
    texts = [text[:max_text_length] for text in texts]
    load_model('facebook/magnet-small-10secs')
    res = _do_predictions(texts, melodies)
    return res


def predict_full(model, model_path, text, temperature, topp,
                 max_cfg_coef, min_cfg_coef, 
                 decoding_steps1, decoding_steps2, decoding_steps3, decoding_steps4, 
                 span_score,
                 progress=gr.Progress()):
    global INTERRUPTING
    INTERRUPTING = False
    progress(0, desc="Loading model...")
    model_path = model_path.strip()
    if model_path:
        if not Path(model_path).exists():
            raise gr.Error(f"Model path {model_path} doesn't exist.")
        if not Path(model_path).is_dir():
            raise gr.Error(f"Model path {model_path} must be a folder containing "
                           "state_dict.bin and compression_state_dict_.bin.")
        model = model_path
    if temperature < 0:
        raise gr.Error("Temperature must be >= 0.")

    load_model(model)

    max_generated = 0

    def _progress(generated, to_generate):
        nonlocal max_generated
        max_generated = max(generated, max_generated)
        progress((min(max_generated, to_generate), to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")
    MODEL.set_custom_progress_callback(_progress)
    
    videos, wavs = _do_predictions(
        [text] * N_REPEATS, progress=True,
        temperature=temperature, top_p=topp,
        max_cfg_coef=max_cfg_coef, min_cfg_coef=min_cfg_coef, 
        decoding_steps=[decoding_steps1, decoding_steps2, decoding_steps3, decoding_steps4],
        span_arrangement='stride1' if (span_score == PROD_STRIDE_1) else 'nonoverlap',
        gradio_progress=progress)

    outputs_ = [videos[0]] + [wav for wav in wavs]
    return tuple(outputs_)

def ui_full(launch_kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # MAGNeT
            This is your private demo for [MAGNeT](https://github.com/facebookresearch/audiocraft),
            A fast text-to-music model, consists of a single, non-autoregressive transformer.
            presented at: ["Masked Audio Generation using a Single Non-Autoregressive Transformer"] (https://huggingface.co/papers/2401.04577)
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Input Text", value="80s electronic track with melodic synthesizers, catchy beat and groovy bass", interactive=True)
                with gr.Row():
                    submit = gr.Button("Submit")
                    # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                    _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)
                with gr.Row():
                    model = gr.Radio(['facebook/magnet-small-10secs', 'facebook/magnet-medium-10secs',
                                      'facebook/magnet-small-30secs', 'facebook/magnet-medium-30secs',
                                      'facebook/audio-magnet-small', 'facebook/audio-magnet-medium'],
                                     label="Model", value='facebook/magnet-small-10secs', interactive=True)
                    model_path = gr.Text(label="Model Path (custom models)") 
                with gr.Row():
                    span_score = gr.Radio(["max-nonoverlap", PROD_STRIDE_1],
                                       label="Span Scoring", value=PROD_STRIDE_1, interactive=True)       
                with gr.Row():
                    decoding_steps1 = gr.Number(label="Decoding Steps (stage 1)", value=20, interactive=True)
                    decoding_steps2 = gr.Number(label="Decoding Steps (stage 2)", value=10, interactive=True)
                    decoding_steps3 = gr.Number(label="Decoding Steps (stage 3)", value=10, interactive=True)
                    decoding_steps4 = gr.Number(label="Decoding Steps (stage 4)", value=10, interactive=True)
                with gr.Row():
                    temperature = gr.Number(label="Temperature", value=3.0, step=0.25, minimum=0, interactive=True)
                    topp = gr.Number(label="Top-p", value=0.9, step=0.1, minimum=0, maximum=1, interactive=True)
                    max_cfg_coef = gr.Number(label="Max CFG coefficient", value=10.0, minimum=0, interactive=True)
                    min_cfg_coef = gr.Number(label="Min CFG coefficient", value=1.0, minimum=0, interactive=True)                
            with gr.Column():
                output = gr.Video(label="Generated Audio - variation 1")
                audio_outputs = [gr.Audio(label=f"Generated Audio - variation {i+1}", type='filepath') for i in range(N_REPEATS)]
        submit.click(fn=predict_full, 
                        inputs=[model, model_path, text, 
                                    temperature, topp,
                                    max_cfg_coef, min_cfg_coef,
                                    decoding_steps1, decoding_steps2, decoding_steps3, decoding_steps4,
                                    span_score],
                                    outputs=[output] + [o for o in audio_outputs])
        gr.Examples(
            fn=predict_full,
            examples=[
                [
                    "80s electronic track with melodic synthesizers, catchy beat and groovy bass",
                    'facebook/magnet-small-10secs',
                    20, 3.0, 0.9, 10.0,
                ],
                [
                    "80s electronic track with melodic synthesizers, catchy beat and groovy bass. 170 bpm",
                    'facebook/magnet-small-10secs',
                    20, 3.0, 0.9, 10.0,
                ],
                [
                    "Earthy tones, environmentally conscious, ukulele-infused, harmonic, breezy, easygoing, organic instrumentation, gentle grooves",
                    'facebook/magnet-medium-10secs',
                    20, 3.0, 0.9, 10.0,
                ],
                [   "Funky groove with electric piano playing blue chords rhythmically",
                    'facebook/magnet-medium-10secs',
                    20, 3.0, 0.9, 10.0,
                ],
                [
                    "Rock with saturated guitars, a heavy bass line and crazy drum break and fills.",
                    'facebook/magnet-small-30secs',
                    60, 3.0, 0.9, 10.0,
                ],
                [   "A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle",
                    'facebook/magnet-medium-30secs',
                    60, 3.0, 0.9, 10.0,
                ],
                [   "Seagulls squawking as ocean waves crash while wind blows heavily into a microphone.",
                    'facebook/audio-magnet-small', 
                    20, 3.5, 0.8, 20.0,
                ],
                [   "A toilet flushing as music is playing and a man is singing in the distance.",
                    'facebook/audio-magnet-medium', 
                    20, 3.5, 0.8, 20.0,
                ],
            ],

            inputs=[text, model, decoding_steps1, temperature, topp, max_cfg_coef],
            outputs=[output]
        )

        gr.Markdown(
            """
            ### More details
            
            #### Music Generation
            "magnet" models will generate a short music extract based on the textual description you provided.
            These models can generate either 10 seconds or 30 seconds of music.
            These models were trained with descriptions from a stock music catalog. Descriptions that will work best
            should include some level of details on the instruments present, along with some intended use case
            (e.g. adding "perfect for a commercial" can somehow help).

            We present 4 model variants:
            1. facebook/magnet-small-10secs - a 300M non-autoregressive transformer capable of generating 10-second music conditioned
                on text.
            2. facebook/magnet-medium-10secs - 1.5B parameters, 10 seconds audio.
            3. facebook/magnet-small-30secs - 300M parameters, 30 seconds audio.
            4. facebook/magnet-medium-30secs - 1.5B parameters, 30 seconds audio.
        
            #### Sound-Effect Generation
            "audio-magnet" models will generate a 10-second sound effect based on the description you provide. 

            These models were trained on the following data sources: a subset of AudioSet (Gemmeke et al., 2017), 
            [BBC sound effects](https://sound-effects.bbcrewind.co.uk/), AudioCaps (Kim et al., 2019), 
            Clotho v2 (Drossos et al., 2020), VGG-Sound (Chen et al., 2020), FSD50K (Fonseca et al., 2021), 
            [Free To Use Sounds](https://www.freetousesounds.com/all-in-one-bundle/), [Sonniss Game Effects](https://sonniss.com/gameaudiogdc), 
            [WeSoundEffects](https://wesoundeffects.com/we-sound-effects-bundle-2020/), 
            [Paramount Motion - Odeon Cinematic Sound Effects](https://www.paramountmotion.com/odeon-sound-effects).

            We present 2 model variants:
            1. facebook/audio-magnet-small - 10 second sound effect generation, 300M parameters.
            2. facebook/audio-magnet-medium - 10 second sound effect generation, 1.5B parameters.

            See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft/blob/main/docs/MAGNET.md)
            for more details.
            """
        )

        interface.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen

    if args.username and args.password:
        launch_kwargs['auth'] = (args.username, args.password)
    if args.server_port:
        launch_kwargs['server_port'] = args.server_port
    if args.inbrowser:
        launch_kwargs['inbrowser'] = args.inbrowser
    if args.share:
        launch_kwargs['share'] = args.share

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    # Show the interface
    ui_full(launch_kwargs)
