"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import random
from tempfile import NamedTemporaryFile
import argparse
import time
import torch
import gradio as gr
import os
import numpy as np
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio
import subprocess, random, string

MODEL = None
IS_SHARED_SPACE = "musicgen/MusicGen" in os.environ.get('SPACE_ID', '')
INTERRUPTED = False
UNLOAD_MODEL = False

def interrupt():
    global INTERRUPTED
    INTERRUPTED = True
    print('Interrupted!')

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

def load_model(version):
    print("Loading model", version)
    return MusicGen.get_pretrained(version)


def predict(model, text, melody, sample, duration, topk, topp, temperature, cfg_coef, seed, overlap=5, recondition=True, background="./assets/background.png", progress=gr.Progress()):
    global MODEL
    global INTERRUPTED
    INTERRUPTED = False
    topk = int(topk)
    if MODEL is None or MODEL.name != model:
        MODEL = load_model(model)

    if duration > MODEL.lm.cfg.dataset.segment_duration and melody is not None:
        raise gr.Error("Generating music longer than 30 seconds with melody conditioning is not yet supported!")
    
    output = None
    first_chunk = None
    total_samples = duration * 50 + 3
    segment_duration = duration
    if seed < 0:
        seed = random.randint(0, 0xffff_ffff_ffff)
    torch.manual_seed(seed)
    predict.last_progress_update = time.monotonic()
    while duration > 0:
        if INTERRUPTED:
            break

        if output is None: # first pass of long or short song
            if segment_duration > MODEL.lm.cfg.dataset.segment_duration: 
                segment_duration = MODEL.lm.cfg.dataset.segment_duration
            else:
                segment_duration = duration
        else: # next pass of long song
            if duration + overlap < MODEL.lm.cfg.dataset.segment_duration:
                segment_duration = duration + overlap
            else:
                segment_duration = MODEL.lm.cfg.dataset.segment_duration
        
        print(f'Segment duration: {segment_duration}, duration: {duration}, overlap: {overlap}')
        MODEL.set_generation_params(
            use_sampling=True,
            top_k=topk,
            top_p=topp,
            temperature=temperature,
            cfg_coef=cfg_coef,
            duration=segment_duration,
        )
        def updateProgress(step: int, total: int):
            now = time.monotonic()
            if now - predict.last_progress_update > 1:
                progress((total_samples - duration * 50 - 3 + step, total_samples))
                predict.last_progress_update = now

        if sample:
            def normalize_audio(audio_data):
                audio_data = audio_data.astype(np.float32)
                max_value = np.max(np.abs(audio_data))
                audio_data = audio_data / max_value
                return audio_data
            
            globalSR, sampleM = sample[0], sample[1]
            sampleM = normalize_audio(sampleM)
            sampleM = torch.from_numpy(sampleM).t()

            if sampleM.dim() > 1:
                sampleM = convert_audio(sampleM, globalSR, 32000, 1)

            sampleM = sampleM.to(MODEL.device).float().unsqueeze(0)
            
            if sampleM.dim() == 2:
                sampleM = sampleM[None]

            sample_length = sampleM.shape[sampleM.dim() - 1] / 32000
            if output is None:
                next_segment = sampleM
                duration -= sample_length
            else:
                if first_chunk is None and MODEL.name == "melody" and recondition:
                    first_chunk = output[:, :, 
                    :MODEL.lm.cfg.dataset.segment_duration*MODEL.sample_rate]
                last_chunk = output[:, :, -overlap*32000:]
                next_segment = MODEL.generate_continuation(last_chunk,
                    32000, descriptions=[text], progress=updateProgress,
                    melody_wavs=(first_chunk), resample=False)
                duration -= segment_duration - overlap
        elif melody:
            sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t().unsqueeze(0)
            print(melody.shape)
            if melody.dim() == 2:
                melody = melody[None]
            melody = melody[..., :int(sr * MODEL.lm.cfg.dataset.segment_duration)]
            next_segment = MODEL.generate_with_chroma(
                descriptions=[text],
                melody_wavs=melody,
                melody_sample_rate=sr,
                progress=updateProgress
            )
            duration -= segment_duration
        else:
            if output is None:
                next_segment = MODEL.generate(descriptions=[text], 
                                              progress=updateProgress)
                duration -= segment_duration
            else:
                if first_chunk is None and MODEL.name == "melody" and recondition:
                    first_chunk = output[:, :, 
                    :MODEL.lm.cfg.dataset.segment_duration*MODEL.sample_rate]
                last_chunk = output[:, :, -overlap*MODEL.sample_rate:]
                next_segment = MODEL.generate_continuation(last_chunk,
                    MODEL.sample_rate, descriptions=[text],
                progress=updateProgress, melody_wavs=(first_chunk), resample=False)
                duration -= segment_duration - overlap
        
        if output is None:
            output = next_segment
        else:
            output = torch.cat([output[:, :, :-overlap*MODEL.sample_rate], next_segment], 2)
        

    output = output.detach().cpu().float()[0]
    with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
        audio_write(
            file.name, output, MODEL.sample_rate, strategy="loudness",
            loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
        waveform_video = gr.make_waveform(file.name, bg_image=background, bg_color="#21b0fe" , bars_color=('#fe218b', '#fed700'), fg_alpha=1.0, bar_count=75)
        if background is None or len(background) == 0:
            random_string = generate_random_string(12)
            random_string = f"{random_string}.mp4"
            resize_video(waveform_video, random_string, 900, 300)
            waveform_video = random_string
    global UNLOAD_MODEL
    if UNLOAD_MODEL:
        MODEL = None
        torch.cuda.empty_cache()
    return waveform_video, seed


def ui(**kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # MusicGen
            This is your private demo for [MusicGen](https://github.com/facebookresearch/audiocraft), a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://arxiv.org/abs/2306.05284)
            """
        )
        if IS_SHARED_SPACE:
            gr.Markdown("""
                ⚠ This Space doesn't work in this shared UI ⚠

                <a href="https://huggingface.co/spaces/musicgen/MusicGen?duplicate=true" style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
                <img style="margin-bottom: 0em;display: inline;margin-top: -.25em;" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
                to use it privately, or use the <a href="https://huggingface.co/spaces/facebook/MusicGen">public demo</a>
                """)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Input Text", interactive=True)
                    melody = gr.Audio(source="upload", type="numpy", label="Melody Condition (optional)", interactive=True)
                    sample = gr.Audio(source="upload", type="numpy", label="Music Sample (optional)", interactive=True)
                with gr.Row():
                    submit = gr.Button("Generate", variant="primary")
                    gr.Button("Interrupt").click(fn=interrupt, queue=False)
                with gr.Row():
                    background = gr.Image(source="upload", label="Background", type="filepath", interactive=True)
                with gr.Row():
                    model = gr.Radio(["melody", "medium", "small", "large"], label="Model", value="melody", interactive=True)
                with gr.Row():
                    duration = gr.Slider(minimum=1, maximum=300, value=10, step=1, label="Duration", interactive=True)
                with gr.Row():
                    overlap = gr.Slider(minimum=1, maximum=29, value=5, step=1, label="Overlap", interactive=True)
                    recondition = gr.Checkbox(False, label='Condition next chunks with the first chunk')
                with gr.Row():
                    topk = gr.Number(label="Top-k", value=250, interactive=True)
                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                    temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                    cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
                with gr.Row():
                    seed = gr.Number(label="Seed", value=-1, precision=0, interactive=True)
                    gr.Button('\U0001f3b2\ufe0f').style(full_width=False).click(fn=lambda: -1, outputs=[seed], queue=False)
                    reuse_seed = gr.Button('\u267b\ufe0f').style(full_width=False)
            with gr.Column() as c:
                output = gr.Video(label="Generated Music")
                seed_used = gr.Number(label='Seed used', value=-1, interactive=False)

        reuse_seed.click(fn=lambda x: x, inputs=[seed_used], outputs=[seed], queue=False)
        submit.click(predict, inputs=[model, text, melody, sample, duration, topk, topp, temperature, cfg_coef, seed, overlap, recondition, background], outputs=[output, seed_used])
        def update_recondition(name: str):
            enabled = name == 'melody'
            return recondition.update(interactive=enabled, value=None if enabled else False)
        model.change(fn=update_recondition, inputs=[model], outputs=[recondition])
        gr.Examples(
            fn=predict,
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
            You can generate up to 30 seconds of audio.

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

        # Show the interface
        launch_kwargs = {}
        username = kwargs.get('username')
        password = kwargs.get('password')
        server_port = kwargs.get('server_port', 0)
        inbrowser = kwargs.get('inbrowser', False)
        share = kwargs.get('share', False)
        server_name = kwargs.get('listen')

        launch_kwargs['server_name'] = server_name

        if username and password:
            launch_kwargs['auth'] = (username, password)
        if server_port > 0:
            launch_kwargs['server_port'] = server_port
        if inbrowser:
            launch_kwargs['inbrowser'] = inbrowser
        if share:
            launch_kwargs['share'] = share

        interface.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='127.0.0.1',
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

    args = parser.parse_args()
    UNLOAD_MODEL = args.unload_model
    ui(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen
    )
