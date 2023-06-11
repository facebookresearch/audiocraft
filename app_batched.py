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
from audiocraft.models import MusicGen


MODEL = None


def load_model():
    print("Loading model")
    return MusicGen.get_pretrained("melody")


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
            audio_write(
                file.name, output, MODEL.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
            waveform_video = gr.make_waveform(file.name)
            out_files.append(waveform_video)
    return [out_files]


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # MusicGen

        This is the demo for [MusicGen](https://github.com/facebookresearch/audiocraft), a simple and controllable model for music generation
        presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284).
        <br/>
        <a href="https://huggingface.co/spaces/musicgen/MusicGen?duplicate=true" style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
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
    submit.click(predict, inputs=[text, melody], outputs=[output], batch=True, max_batch_size=12)
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

demo.queue(max_size=15).launch()
