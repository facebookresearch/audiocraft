# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Updated to account for UI changes from https://github.com/rkfg/audiocraft/blob/long/app.py
# also released under the MIT license.

import random
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import subprocess as sp
from tempfile import NamedTemporaryFile
import time
import warnings

import torch
import gradio as gr
import numpy as np

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen
import subprocess, random, string

MODEL = None  # Last used model
MODELS = None
IS_SHARED_SPACE = "musicgen/MusicGen" in os.environ.get('SPACE_ID', '')
INTERRUPTED = False
UNLOAD_MODEL = False
MOVE_TO_CPU = False
IS_BATCHED = "facebook/MusicGen" in os.environ.get('SPACE_ID', '')
MAX_BATCH_SIZE = 12
BATCHED_DURATION = 15
INTERRUPTING = False
# We have to wrap subprocess call to clean a bit the log when using gr.make_waveform
_old_call = sp.call

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def resize_video(input_path, output_path, target_width, target_height):
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-i', input_path,
        '-vf', f'scale={target_width}:{target_height}',
        '-c:a', 'copy',
        output_path
    ]
    subprocess.run(ffmpeg_cmd)

def _call_nostderr(*args, **kwargs):
    # Avoid ffmpeg vomitting on the logs.
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


def make_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        waveform_video = gr.make_waveform(*args, **kwargs)
        out = f"{generate_random_string(12)}.mp4"
        resize_video(waveform_video, out, 900, 300)
        print("Make a video took", time.time() - be)
        return out


def load_model(version='melody'):
    global MODEL, MODELS
    print("Loading model", version)
    if MODELS is None:
        MODEL = MusicGen.get_pretrained(version)
        return
    else:
        t1 = time.monotonic()
        if MODEL is not None:
            MODEL.to('cpu') # move to cache
            print("Previous model moved to CPU in %.2fs" % (time.monotonic() - t1))
            t1 = time.monotonic()
        if MODELS.get(version) is None:
            print("Loading model %s from disk" % version)
            result = MusicGen.get_pretrained(version)
            MODELS[version] = result
            print("Model loaded in %.2fs" % (time.monotonic() - t1))
            MODEL = result
            return
        result = MODELS[version].to('cuda')
        print("Cached model loaded in %.2fs" % (time.monotonic() - t1))
        MODEL = result

def normalize_audio(audio_data):
    audio_data = audio_data.astype(np.float32)
    max_value = np.max(np.abs(audio_data))
    audio_data /= max_value
    return audio_data

def _do_predictions(texts, melodies, sample, duration, progress=False, **gen_kwargs):
    global MODEL
    MODEL.set_generation_params(duration=duration, **gen_kwargs)
    print("new batch", len(texts), texts, [None if m is None else (m[0], m[1].shape) for m in melodies], [None if sample is None else (sample[0], sample[1].shape)])
    be = time.time()
    processed_melodies = []
    target_sr = 32000
    target_ac = 1
    for melody in melodies:
        if melody is None:
            processed_melodies.append(None)
        else:
            sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t()
            if melody.dim() == 1:
                melody = melody[None]
            melody = melody[..., :int(sr * duration)]
            melody = convert_audio(melody, sr, target_sr, target_ac)
            processed_melodies.append(melody)
    
    if sample is not None:
        globalSR, sampleM = sample[0], sample[1]
        sampleM = normalize_audio(sampleM)
        sampleM = torch.from_numpy(sampleM).t()
    
        #if sampleM.dim() > 1:
        #    sampleM = convert_audio(sampleM, globalSR, target_sr, target_ac)
        
        #sampleM = sampleM.to(MODEL.device).float().unsqueeze(0)
        if sampleM.dim() == 1:
            sampleM = sampleM.unsqueeze(0)
        
        #if sampleM.dim() == 2:
        #    sampleM = sampleM[None]

        outputs = MODEL.generate_continuation(
            prompt=sampleM,
            prompt_sample_rate=globalSR,
            descriptions=texts,
            progress=progress,
        )
    elif any(m is not None for m in processed_melodies):
        outputs = MODEL.generate_with_chroma(
            descriptions=texts,
            melody_wavs=processed_melodies,
            melody_sample_rate=target_sr,
            progress=progress,
        )
    else:
        outputs = MODEL.generate(texts, progress=progress)

    outputs = outputs.detach().cpu().float()
    out_files = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, MODEL.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
            out_files.append(pool.submit(make_waveform, file.name, bg_color="#21b0fe" , bars_color=('#fe218b', '#fed700'), fg_alpha=1.0, bar_count=75))
    res = [out_file.result() for out_file in out_files]
    print("batch finished", len(texts), time.time() - be)
    if MOVE_TO_CPU:
        MODEL.to('cpu')
    if UNLOAD_MODEL:
        MODEL = None
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return res


def predict_batched(texts, melodies):
    max_text_length = 512
    texts = [text[:max_text_length] for text in texts]
    load_model('melody')
    res = _do_predictions(texts, melodies, BATCHED_DURATION)
    return [res]


def predict_full(model, text, melody, sample, duration, topk, topp, temperature, cfg_coef, seed, overlap, progress=gr.Progress()):
    global INTERRUPTING
    INTERRUPTING = False
    if temperature < 0:
        raise gr.Error("Temperature must be >= 0.")
    if topk < 0:
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        raise gr.Error("Topp must be non-negative.")

    topk = int(topk)
    if MODEL is None or MODEL.name != model:
        load_model(model)
    else:
        if MOVE_TO_CPU:
            MODEL.to('cuda')

    if seed < 0:
        seed = random.randint(0, 0xffff_ffff_ffff)
    torch.manual_seed(seed)
    predict_full.last_upd = time.monotonic()
    def _progress(generated, to_generate):
        if time.monotonic() - predict_full.last_upd > 1:
            progress((generated, to_generate))
            predict_full.last_upd = time.monotonic()
        if INTERRUPTING:
            raise gr.Error("Interrupted.")
    MODEL.set_custom_progress_callback(_progress)

    outs = _do_predictions(
        [text], [melody], sample, duration, progress=True,
        top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef, extend_stride=MODEL.max_duration-overlap)
    return outs[0], seed


def ui_full(launch_kwargs):
    with gr.Blocks(title='MusicGen+') as interface:
        gr.Markdown(
            """
            # MusicGen+ Experimental Version
            This is your private demo for [MusicGen](https://github.com/facebookresearch/audiocraft), a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284)
            
            This is an extended version of the original
            Thanks to: Camenduru, rkfg and GrandaddyShmax
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Input Text", interactive=True)
                    melody = gr.Audio(source="upload", type="numpy", label="Melody Condition (optional)", interactive=True)
                    sample = gr.Audio(source="upload", type="numpy", label="Music Sample (optional)", interactive=True)
                with gr.Row():
                    submit = gr.Button("Generate", variant="primary")
                    # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                    _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)
                with gr.Row():
                    model = gr.Radio(["melody", "medium", "small", "large"], label="Model", value="melody", interactive=True)
                with gr.Row():
                    duration = gr.Slider(minimum=1, maximum=300, value=10, step=1, label="Duration", interactive=True)
                with gr.Row():
                    overlap = gr.Slider(minimum=1, maximum=29, value=12, step=1, label="Overlap", interactive=True)
                with gr.Row():
                    topk = gr.Number(label="Top-k", value=250, interactive=True)
                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                    temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                    cfg_coef = gr.Number(label="Classifier Free Guidance", value=5.0, interactive=True)
                with gr.Row():
                    seed = gr.Number(label="Seed", value=-1, precision=0, interactive=True)
                    gr.Button('\U0001f3b2\ufe0f').style(full_width=False).click(fn=lambda: -1, outputs=[seed], queue=False)
                    reuse_seed = gr.Button('\u267b\ufe0f').style(full_width=False)
            with gr.Column() as c:
                output = gr.Video(label="Generated Music")
                seed_used = gr.Number(label='Seed used', value=-1, interactive=False)

        reuse_seed.click(fn=lambda x: x, inputs=[seed_used], outputs=[seed], queue=False)
        submit.click(predict_full, inputs=[model, text, melody, sample, duration, topk, topp, temperature, cfg_coef, seed, overlap], outputs=[output, seed_used])
        gr.Examples(
            fn=predict_full,
            examples=[
                [
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                    "melody"
                ],
                [
                    "A cheerful country song with acoustic guitars",
                    "./assets/bolero_ravel.mp3",
                    "melody"
                ],
                [
                    "90s rock song with electric guitar and heavy drums",
                    None,
                    "medium"
                ],
                [
                    "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions",
                    "./assets/bach.mp3",
                    "melody"
                ],
                [
                    "lofi slow bpm electro chill with organic samples",
                    None,
                    "medium",
                ],
            ],
            inputs=[text, melody, model],
            outputs=[output]
        )
        gr.Markdown(
            """
            ### More details

            The model will generate a short music extract based on the description you provided.
            The model can generate up to 30 seconds of audio in one pass. It is now possible
            to extend the generation by feeding back the end of the previous chunk of audio.
            This can take a long time, and the model might lose consistency. The model might also
            decide at arbitrary positions that the song ends.

            **WARNING:** Choosing long durations will take a long time to generate (2min might take ~10min). An overlap of 12 seconds
            is kept with the previously generated chunk, and 18 "new" seconds are generated each time.

            We present 4 model variations:
            1. Melody -- a music generation model capable of generating music condition on text and melody inputs. **Note**, you can also use text only.
            2. Small -- a 300M transformer decoder conditioned on text only.
            3. Medium -- a 1.5B transformer decoder conditioned on text only.
            4. Large -- a 3.3B transformer decoder conditioned on text only (might OOM for the longest sequences.)

            When using `melody`, ou can optionaly provide a reference audio from
            which a broad melody will be extracted. The model will then try to follow both the description and melody provided.

            You can also use your own GPU or a Google Colab by following the instructions on our repo.
            See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
            for more details.
            """
        )

        interface.queue().launch(**launch_kwargs)


def ui_batched(launch_kwargs):
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # MusicGen

            This is the demo for [MusicGen](https://github.com/facebookresearch/audiocraft), a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284).
            <br/>
            <a href="https://huggingface.co/spaces/facebook/MusicGen?duplicate=true" style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
            <img style="margin-bottom: 0em;display: inline;margin-top: -.25em;" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
            for longer sequences, more control and no queue.</p>
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Describe your music", lines=2, interactive=True)
                    melody = gr.Audio(source="upload", type="numpy", label="Condition on a melody (optional)", interactive=True)
                with gr.Row():
                    submit = gr.Button("Generate")
            with gr.Column():
                output = gr.Video(label="Generated Music")
        submit.click(predict_batched, inputs=[text, melody], outputs=[output], batch=True, max_batch_size=MAX_BATCH_SIZE)
        gr.Examples(
            fn=predict_batched,
            examples=[
                [
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                ],
                [
                    "A cheerful country song with acoustic guitars",
                    "./assets/bolero_ravel.mp3",
                ],
                [
                    "90s rock song with electric guitar and heavy drums",
                    None,
                ],
                [
                    "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130",
                    "./assets/bach.mp3",
                ],
                [
                    "lofi slow bpm electro chill with organic samples",
                    None,
                ],
            ],
            inputs=[text, melody],
            outputs=[output]
        )
        gr.Markdown("""
        ### More details

        The model will generate 12 seconds of audio based on the description you provided.
        You can optionaly provide a reference audio from which a broad melody will be extracted.
        The model will then try to follow both the description and melody provided.
        All samples are generated with the `melody` model.

        You can also use your own GPU or a Google Colab by following the instructions on our repo.

        See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
        for more details.
        """)

        demo.queue(max_size=8 * 4).launch(**launch_kwargs)


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
    parser.add_argument(
        '--unload_model', action='store_true', help='Unload the model after every generation to save GPU memory'
    )

    parser.add_argument(
        '--unload_to_cpu', action='store_true', help='Move the model to main RAM after every generation to save GPU memory but reload faster than after full unload (see above)'
    )

    parser.add_argument(
        '--cache', action='store_true', help='Cache models in RAM to quickly switch between them'
    )

    args = parser.parse_args()
    UNLOAD_MODEL = args.unload_model
    MOVE_TO_CPU = args.unload_to_cpu
    if args.cache:
        MODELS = {}

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

    # Show the interface
    if IS_BATCHED:
        ui_batched(launch_kwargs)
    else:
        ui_full(launch_kwargs)
