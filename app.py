"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from tempfile import NamedTemporaryFile
import argparse
import torch
import gradio as gr
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

MODEL = None
IS_SHARED_SPACE = "musicgen/MusicGen" in os.environ.get('SPACE_ID', '')


def load_model(version):
    print("Loading model", version)
    return MusicGen.get_pretrained(version)


def predict(model, text, melody, duration, topk, topp, temperature, cfg_coef):
    global MODEL
    topk = int(topk)
    if MODEL is None or MODEL.name != model:
        MODEL = load_model(model)

    if duration > MODEL.lm.cfg.dataset.segment_duration:
        raise gr.Error("MusicGen currently supports durations of up to 30 seconds!")
    MODEL.set_generation_params(
        use_sampling=True,
        top_k=topk,
        top_p=topp,
        temperature=temperature,
        cfg_coef=cfg_coef,
        duration=duration,
    )

    if melody:
        sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t().unsqueeze(0)
        print(melody.shape)
        if melody.dim() == 2:
            melody = melody[None]
        melody = melody[..., :int(sr * MODEL.lm.cfg.dataset.segment_duration)]
        output = MODEL.generate_with_chroma(
            descriptions=[text],
            melody_wavs=melody,
            melody_sample_rate=sr,
            progress=False
        )
    else:
        output = MODEL.generate(descriptions=[text], progress=False)

    output = output.detach().cpu().float()[0]
    with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
        audio_write(
            file.name, output, MODEL.sample_rate, strategy="loudness",
            loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
        waveform_video = gr.make_waveform(file.name)
    return waveform_video


def ui(**kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # MusicGen
            This is your private demo for [MusicGen](https://github.com/facebookresearch/audiocraft), a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284)
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
                with gr.Row():
                    submit = gr.Button("Submit")
                with gr.Row():
                    model = gr.Radio(["melody", "medium", "small", "large"], label="Model", value="melody", interactive=True)
                with gr.Row():
                    duration = gr.Slider(minimum=1, maximum=30, value=10, label="Duration", interactive=True)
                with gr.Row():
                    topk = gr.Number(label="Top-k", value=250, interactive=True)
                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                    temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                    cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
            with gr.Column():
                output = gr.Video(label="Generated Music")
        submit.click(predict, inputs=[model, text, melody, duration, topk, topp, temperature, cfg_coef], outputs=[output])
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

        interface.queue().launch(**launch_kwargs, max_threads=1)


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

    args = parser.parse_args()

    ui(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen
    )
