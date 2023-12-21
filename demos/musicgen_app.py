# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Updated to account for UI changes from https://github.com/rkfg/audiocraft/blob/long/app.py
# also released under the MIT license.

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

from einops import rearrange
import torch
import gradio as gr

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models.encodec import InterleaveStereoCompressionModel
from audiocraft.models import MusicGen, MultiBandDiffusion


MODEL = None  # Last used model
SPACE_ID = os.environ.get('SPACE_ID', '')
IS_BATCHED = "facebook/MusicGen" in SPACE_ID or 'musicgen-internal/musicgen_dev' in SPACE_ID
print(IS_BATCHED)
MAX_BATCH_SIZE = 12
BATCHED_DURATION = 15
INTERRUPTING = False
MBD = None
# We have to wrap subprocess call to clean a bit the log when using gr.make_waveform
_old_call = sp.call


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


def load_model(version='facebook/musicgen-melody'):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        # Clear PyTorch CUDA cache and delete model
        del MODEL
        torch.cuda.empty_cache()
        MODEL = None  # in case loading would crash
        MODEL = MusicGen.get_pretrained(version)


def load_diffusion():
    global MBD
    if MBD is None:
        print("loading MBD")
        MBD = MultiBandDiffusion.get_mbd_musicgen()


def _do_predictions(texts, melodies, duration, progress=False, gradio_progress=None, **gen_kwargs):
    MODEL.set_generation_params(duration=duration, **gen_kwargs)
    print("new batch", len(texts), texts, [None if m is None else (m[0], m[1].shape) for m in melodies])
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

    try:
        if any(m is not None for m in processed_melodies):
            outputs = MODEL.generate_with_chroma(
                descriptions=texts,
                melody_wavs=processed_melodies,
                melody_sample_rate=target_sr,
                progress=progress,
                return_tokens=USE_DIFFUSION
            )
        else:
            outputs = MODEL.generate(texts, progress=progress, return_tokens=USE_DIFFUSION)
    except RuntimeError as e:
        raise gr.Error("Error while generating " + e.args[0])
    if USE_DIFFUSION:
        if gradio_progress is not None:
            gradio_progress(1, desc='Running MultiBandDiffusion...')
        tokens = outputs[1]
        if isinstance(MODEL.compression_model, InterleaveStereoCompressionModel):
            left, right = MODEL.compression_model.get_left_right_codes(tokens)
            tokens = torch.cat([left, right])
        outputs_diffusion = MBD.tokens_to_wav(tokens)
        if isinstance(MODEL.compression_model, InterleaveStereoCompressionModel):
            assert outputs_diffusion.shape[1] == 1  # output is mono
            outputs_diffusion = rearrange(outputs_diffusion, '(s b) c t -> b (s c) t', s=2)
        outputs = torch.cat([outputs[0], outputs_diffusion], dim=0)
    outputs = outputs.detach().cpu().float()
    pending_videos = []
    out_wavs = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, MODEL.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
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
    load_model('facebook/musicgen-stereo-melody')
    res = _do_predictions(texts, melodies, BATCHED_DURATION)
    return res


def predict_full(model, model_path, decoder, text, melody, duration, topk, topp, temperature, cfg_coef, progress=gr.Progress()):
    global INTERRUPTING
    global USE_DIFFUSION
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
    if topk < 0:
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        raise gr.Error("Topp must be non-negative.")

    topk = int(topk)
    if decoder == "MultiBand_Diffusion":
        USE_DIFFUSION = True
        progress(0, desc="Loading diffusion model...")
        load_diffusion()
    else:
        USE_DIFFUSION = False
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
        [text], [melody], duration, progress=True,
        top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef,
        gradio_progress=progress)
    if USE_DIFFUSION:
        return videos[0], wavs[0], videos[1], wavs[1]
    return videos[0], wavs[0], None, None


def toggle_audio_src(choice):
    if choice == "mic":
        return gr.update(source="microphone", value=None, label="Microphone")
    else:
        return gr.update(source="upload", value=None, label="File")


def toggle_diffusion(choice):
    if choice == "MultiBand_Diffusion":
        return [gr.update(visible=True)] * 2
    else:
        return [gr.update(visible=False)] * 2


def ui_full(launch_kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # MusicGen
            This is your private demo for [MusicGen](https://github.com/facebookresearch/audiocraft),
            a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284)
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Input Text", interactive=True)
                    with gr.Column():
                        radio = gr.Radio(["file", "mic"], value="file",
                                         label="Condition on a melody (optional) File or Mic")
                        melody = gr.Audio(sources=["upload"], type="numpy", label="File",
                                          interactive=True, elem_id="melody-input")
                with gr.Row():
                    submit = gr.Button("Submit")
                    # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                    _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)
                with gr.Row():
                    model = gr.Radio(["facebook/musicgen-melody", "facebook/musicgen-medium", "facebook/musicgen-small",
                                      "facebook/musicgen-large", "facebook/musicgen-melody-large",
                                      "facebook/musicgen-stereo-small", "facebook/musicgen-stereo-medium",
                                      "facebook/musicgen-stereo-melody", "facebook/musicgen-stereo-large",
                                      "facebook/musicgen-stereo-melody-large"],
                                     label="Model", value="facebook/musicgen-stereo-melody", interactive=True)
                    model_path = gr.Text(label="Model Path (custom models)")
                with gr.Row():
                    decoder = gr.Radio(["Default", "MultiBand_Diffusion"],
                                       label="Decoder", value="Default", interactive=True)
                with gr.Row():
                    duration = gr.Slider(minimum=1, maximum=120, value=10, label="Duration", interactive=True)
                with gr.Row():
                    topk = gr.Number(label="Top-k", value=250, interactive=True)
                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                    temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                    cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
            with gr.Column():
                output = gr.Video(label="Generated Music")
                audio_output = gr.Audio(label="Generated Music (wav)", type='filepath')
                diffusion_output = gr.Video(label="MultiBand Diffusion Decoder")
                audio_diffusion = gr.Audio(label="MultiBand Diffusion Decoder (wav)", type='filepath')
        submit.click(toggle_diffusion, decoder, [diffusion_output, audio_diffusion], queue=False,
                     show_progress=False).then(predict_full, inputs=[model, model_path, decoder, text, melody, duration, topk, topp,
                                                                     temperature, cfg_coef],
                                               outputs=[output, audio_output, diffusion_output, audio_diffusion])
        radio.change(toggle_audio_src, radio, [melody], queue=False, show_progress=False)

        gr.Examples(
            fn=predict_full,
            examples=[
                [
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                    "facebook/musicgen-stereo-melody",
                    "Default"
                ],
                [
                    "A cheerful country song with acoustic guitars",
                    "./assets/bolero_ravel.mp3",
                    "facebook/musicgen-stereo-melody",
                    "Default"
                ],
                [
                    "90s rock song with electric guitar and heavy drums",
                    None,
                    "facebook/musicgen-stereo-medium",
                    "Default"
                ],
                [
                    "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions",
                    "./assets/bach.mp3",
                    "facebook/musicgen-stereo-melody",
                    "Default"
                ],
                [
                    "lofi slow bpm electro chill with organic samples",
                    None,
                    "facebook/musicgen-stereo-medium",
                    "Default"
                ],
                [
                    "Punk rock with loud drum and power guitar",
                    None,
                    "facebook/musicgen-stereo-medium",
                    "MultiBand_Diffusion"
                ],
            ],
            inputs=[text, melody, model, decoder],
            outputs=[output]
        )
        gr.Markdown(
            """
            ### More details

            The model will generate a short music extract based on the description you provided.
            The model can generate up to 30 seconds of audio in one pass.

            The model was trained with description from a stock music catalog, descriptions that will work best
            should include some level of details on the instruments present, along with some intended use case
            (e.g. adding "perfect for a commercial" can somehow help).

            Using one of the `melody` model (e.g. `musicgen-melody-*`), you can optionally provide a reference audio
            from which a broad melody will be extracted.
            The model will then try to follow both the description and melody provided.
            For best results, the melody should be 30 seconds long (I know, the samples we provide are not...)

            It is now possible to extend the generation by feeding back the end of the previous chunk of audio.
            This can take a long time, and the model might lose consistency. The model might also
            decide at arbitrary positions that the song ends.

            **WARNING:** Choosing long durations will take a long time to generate (2min might take ~10min).
            An overlap of 12 seconds is kept with the previously generated chunk, and 18 "new" seconds
            are generated each time.

            We present 10 model variations:
            1. facebook/musicgen-melody -- a music generation model capable of generating music condition
                on text and melody inputs. **Note**, you can also use text only.
            2. facebook/musicgen-small -- a 300M transformer decoder conditioned on text only.
            3. facebook/musicgen-medium -- a 1.5B transformer decoder conditioned on text only.
            4. facebook/musicgen-large -- a 3.3B transformer decoder conditioned on text only.
            5. facebook/musicgen-melody-large -- a 3.3B transformer decoder conditioned on and melody.
            6. facebook/musicgen-stereo-*: same as the previous models but fine tuned to output stereo audio.

            We also present two way of decoding the audio tokens
            1. Use the default GAN based compression model. It can suffer from artifacts especially
                for crashes, snares etc.
            2. Use [MultiBand Diffusion](https://arxiv.org/abs/2308.02560). Should improve the audio quality,
                at an extra computational cost. When this is selected, we provide both the GAN based decoded
                audio, and the one obtained with MBD.

            See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md)
            for more details.
            """
        )

        interface.queue().launch(**launch_kwargs)


def ui_batched(launch_kwargs):
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # MusicGen

            This is the demo for [MusicGen](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md),
            a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284).
            <br/>
            <a href="https://huggingface.co/spaces/facebook/MusicGen?duplicate=true"
                style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
            <img style="margin-bottom: 0em;display: inline;margin-top: -.25em;"
                src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
            for longer sequences, more control and no queue.</p>
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Describe your music", lines=2, interactive=True)
                    with gr.Column():
                        radio = gr.Radio(["file", "mic"], value="file",
                                         label="Condition on a melody (optional) File or Mic")
                        melody = gr.Audio(source="upload", type="numpy", label="File",
                                          interactive=True, elem_id="melody-input")
                with gr.Row():
                    submit = gr.Button("Generate")
            with gr.Column():
                output = gr.Video(label="Generated Music")
                audio_output = gr.Audio(label="Generated Music (wav)", type='filepath')
        submit.click(predict_batched, inputs=[text, melody],
                     outputs=[output, audio_output], batch=True, max_batch_size=MAX_BATCH_SIZE)
        radio.change(toggle_audio_src, radio, [melody], queue=False, show_progress=False)
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

        The model will generate 15 seconds of audio based on the description you provided.
        The model was trained with description from a stock music catalog, descriptions that will work best
        should include some level of details on the instruments present, along with some intended use case
        (e.g. adding "perfect for a commercial" can somehow help).

        You can optionally provide a reference audio from which a broad melody will be extracted.
        The model will then try to follow both the description and melody provided.
        For best results, the melody should be 30 seconds long (I know, the samples we provide are not...)

        You can access more control (longer generation, more models etc.) by clicking
        the <a href="https://huggingface.co/spaces/facebook/MusicGen?duplicate=true"
                style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
            <img style="margin-bottom: 0em;display: inline;margin-top: -.25em;"
                src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
        (you will then need a paid GPU from HuggingFace).
        If you have a GPU, you can run the gradio demo locally (click the link to our repo below for more info).
        Finally, you can get a GPU for free from Google
        and run the demo in [a Google Colab.](https://ai.honu.io/red/musicgen-colab).

        See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md)
        for more details. All samples are generated with the `stereo-melody` model.
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
    if IS_BATCHED:
        global USE_DIFFUSION
        USE_DIFFUSION = False
        ui_batched(launch_kwargs)
    else:
        ui_full(launch_kwargs)
