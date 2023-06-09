"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from tempfile import NamedTemporaryFile
import torch
import gradio as gr
from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from hf_loading import get_pretrained


MODEL = None


def load_model():
    print("Loading model")
    return get_pretrained("melody")


def predict(texts, melodies):
    global MODEL
    if MODEL is None:
        MODEL = load_model()

    duration = 12
    MODEL.set_generation_params(duration=duration)

    print(texts, melodies)
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

    outputs = MODEL.generate_with_chroma(
        descriptions=texts,
        melody_wavs=processed_melodies,
        melody_sample_rate=target_sr,
        progress=False
    )

    outputs = outputs.detach().cpu().float()
    out_files = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(file.name, output, MODEL.sample_rate, strategy="loudness", add_suffix=False)
            out_files.append([file.name])
    return out_files


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # MusicGen

        This is the demo for MusicGen, a simple and controllable model for music generation
        presented at: "Simple and Controllable Music Generation".

        Enter the description of the music you want and an optional audio used for melody conditioning.
        The model will extract the broad melody from the uploaded wav if provided.
        This will generate a 12s extract with the `melody` model.

        **Warning:** Due to high demand, the demo might get stuck, in that case please refresh
        the page! Normal processing time is ~30 seconds.

        For generating longer sequences (up to 30 seconds) and skipping queue, you can duplicate
        to full demo space, which contains more control and upgrade to GPU in the settings.
        <br/>
        <a href="https://huggingface.co/spaces/musicgen/MusicGen?duplicate=true">
        <img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
        </p>

        You can also use your own GPU or a Google Colab by following the instructions on our repo.

        See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
        for more details.
        """
    )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                text = gr.Text(label="Input Text", interactive=True)
                melody = gr.Audio(source="upload", type="numpy", label="Melody Condition (optional)", interactive=True)
            with gr.Row():
                submit = gr.Button("Submit")
        with gr.Column():
            output = gr.Audio(label="Generated Music", type="filepath", format="wav")
    submit.click(predict, inputs=[text, melody], outputs=[output], batch=True, max_batch_size=1)
    gr.Examples(
        fn=predict,
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

demo.queue(max_size=15).launch()
