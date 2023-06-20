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
import glob
import re
from pathlib import Path

import torch
import gradio as gr
import numpy as np

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen
from audiocraft.utils import ui
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
        image = kwargs.get('bg_image', None)
        if image is None:
            resize_video(waveform_video, out, 900, 300)
        else:
            resize_video(waveform_video, out, 768, 512)
        print("Make a video took", time.time() - be)
        return out


def load_model(version='melody', custom_model=None, base_model='medium'):
    global MODEL, MODELS
    print("Loading model", version)
    if MODELS is None:
        if version == 'custom':
            MODEL = MusicGen.get_pretrained(base_model)
            MODEL.lm.load_state_dict(torch.load(custom_model))
        else:
            MODEL = MusicGen.get_pretrained(version)
        return
    else:
        t1 = time.monotonic()
        if MODEL is not None:
            MODEL.to('cpu') # move to cache
            print("Previous model moved to CPU in %.2fs" % (time.monotonic() - t1))
            t1 = time.monotonic()
        if version != 'custom' and MODELS.get(version) is None:
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

def _do_predictions(texts, melodies, sample, duration, image, background, bar1, bar2, progress=False, **gen_kwargs):
    maximum_size = 29.5
    cut_size = 0
    sampleP = None
    if sample is not None:
        globalSR, sampleM = sample[0], sample[1]
        sampleM = normalize_audio(sampleM)
        sampleM = torch.from_numpy(sampleM).t()
        if sampleM.dim() == 1:
            sampleM = sampleM.unsqueeze(0)
        sample_length = sampleM.shape[sampleM.dim() - 1] / globalSR
        if sample_length > maximum_size:
            cut_size = sample_length - maximum_size
            sampleP = sampleM[..., :int(globalSR * cut_size)]
            sampleM = sampleM[..., int(globalSR * cut_size):]
        if sample_length >= duration:
            duration = sample_length + 0.5
    global MODEL
    MODEL.set_generation_params(duration=(duration - cut_size), **gen_kwargs)
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
        if sampleP is None:
            outputs = MODEL.generate_continuation(
                prompt=sampleM,
                prompt_sample_rate=globalSR,
                descriptions=texts,
                progress=progress,
            )
        else:
            if sampleP.dim() > 1:
                sampleP = convert_audio(sampleP, globalSR, target_sr, target_ac)
            sampleP = sampleP.to(MODEL.device).float().unsqueeze(0)
            outputs = MODEL.generate_continuation(
                prompt=sampleM,
                prompt_sample_rate=globalSR,
                descriptions=texts,
                progress=progress,
            )
            outputs = torch.cat([sampleP, outputs], 2)
            
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
            out_files.append(pool.submit(make_waveform, file.name, bg_image=image, bg_color=background, bars_color=(bar1, bar2), fg_alpha=1.0, bar_count=75))
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


def predict_full(model, custom_model, base_model, prompt_amount, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, audio, mode, duration, topk, topp, temperature, cfg_coef, seed, overlap, image, background, bar1, bar2, progress=gr.Progress()):
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
        load_model(model, custom_model, base_model)
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

    melody = None
    sample = None
    if mode == "sample":
        sample = audio
    elif mode == "melody":
        melody = audio
    
    text_cat = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
    drag_cat = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9]
    texts = []
    ind = 0
    ind2 = 0
    while ind < prompt_amount:
        for ind2 in range(int(drag_cat[ind])):
            texts.append(text_cat[ind])
        ind2 = 0
        ind = ind + 1

    outs = _do_predictions(
        [texts], [melody], sample, duration, image, background, bar1, bar2, progress=True,
        top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef, extend_stride=MODEL.max_duration-overlap)
    return outs[0], seed

max_textboxes = 10

def get_available_models():
    return sorted([re.sub('.pt$', '', item.name) for item in list(Path('models/').glob('*')) if item.name.endswith('.pt')])

def ui_full(launch_kwargs):
    with gr.Blocks(title='MusicGen+') as interface:
        gr.Markdown(
            """
            # MusicGen+ V1.2.2

            Thanks to: facebookresearch, Camenduru, rkfg, oobabooga and GrandaddyShmax
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Tab("Generation"):
                    with gr.Row():
                        s = gr.Slider(1, max_textboxes, value=1, step=1, label="Prompt Segments:")
                    with gr.Column():
                        textboxes = []
                        prompts = []
                        repeats = []
                        with gr.Row():
                            text0 = gr.Text(label="Input Text", interactive=True, scale=3)
                            prompts.append(text0)
                            drag0 = gr.Number(label="Repeat", value=1, interactive=True, scale=1)
                            repeats.append(drag0)
                        for i in range(max_textboxes):
                            with gr.Row(visible=False) as t:
                                text = gr.Text(label="Input Text", interactive=True, scale=3)
                                repeat = gr.Number(label="Repeat", minimum=1, value=1, interactive=True, scale=1)
                            textboxes.append(t)
                            prompts.append(text)
                            repeats.append(repeat)
                    with gr.Row():
                        mode = gr.Radio(["melody", "sample"], label="Input Audio Mode", value="sample", interactive=True)
                        audio = gr.Audio(source="upload", type="numpy", label="Input Audio (optional)", interactive=True)
                    with gr.Row():
                        submit = gr.Button("Generate", variant="primary")
                        # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                        _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)
                    with gr.Row():
                        duration = gr.Slider(minimum=1, maximum=300, value=10, step=1, label="Duration", interactive=True)
                    with gr.Row():
                        overlap = gr.Slider(minimum=1, maximum=29, value=12, step=1, label="Overlap", interactive=True)
                    with gr.Row():
                        seed = gr.Number(label="Seed", value=-1, precision=0, interactive=True)
                        gr.Button('\U0001f3b2\ufe0f').style(full_width=False).click(fn=lambda: -1, outputs=[seed], queue=False)
                        reuse_seed = gr.Button('\u267b\ufe0f').style(full_width=False)
                with gr.Tab("Customization"):
                    with gr.Row():
                        with gr.Column():
                            background = gr.ColorPicker(value="#22A699", label="background color", interactive=True, scale=0)
                            bar1 = gr.ColorPicker(value="#F2BE22", label="bar color start", interactive=True, scale=0)
                            bar2 = gr.ColorPicker(value="#F29727", label="bar color end", interactive=True, scale=0)
                        image = gr.Image(label="Background Image", shape=(768,512), type="filepath", interactive=True, scale=4)
                with gr.Tab("Settings"):
                    with gr.Row():
                        model = gr.Radio(["melody", "small", "medium", "large", "custom"], label="Model", value="melody", interactive=True, scale=1)
                        with gr.Column():
                            dropdown = gr.Dropdown(choices=get_available_models(), value=("No models found" if len(get_available_models()) < 1 else get_available_models()[0]), label='Custom Model (models folder)', elem_classes='slim-dropdown', interactive=True)
                            ui.create_refresh_button(dropdown, lambda: None, lambda: {'choices': get_available_models()}, 'refresh-button')
                            basemodel = gr.Radio(["small", "medium", "large"], label="Base Model", value="medium", interactive=True, scale=1)
                    with gr.Row():
                        topk = gr.Number(label="Top-k", value=250, interactive=True)
                        topp = gr.Number(label="Top-p", value=0, interactive=True)
                        temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                        cfg_coef = gr.Number(label="Classifier Free Guidance", value=5.0, interactive=True)
            with gr.Column() as c:
                with gr.Tab("Output"):
                    output = gr.Video(label="Generated Music", scale=0)
                    seed_used = gr.Number(label='Seed used', value=-1, interactive=False)
                with gr.Tab("Wiki"):
                    gr.Markdown(
                        """
                        ### Generation Tab:

                        #### Multi-Prompt: 
                        
                        This feature allows you to control the music, adding variation to different time segments.  
                        You have up to 10 prompt segments. the first prompt will always be 30s long  
                        the other prompts will be [30s - overlap].  
                        for example if the overlap is 10s, each prompt segment will be 20s.

                        - **[Prompt Segments (number)]:**  
                        Amount of unique prompt to generate throughout the music generation.

                        - **[Prompt/Input Text (prompt)]:**  
                        Here describe the music you wish the model to generate.

                        - **[Repeat (number)]:**  
                        Write how many times this prompt will repeat (instead of wasting another prompt segment on the same prompt).

                        - **[Input Audio Mode (selection)]:**  
                        `Melody` mode only works with the melody model: it conditions the music generation to reference the melody  
                        `Sample` mode works with any model: it gives a music sample to the model to generate its continuation.

                        - **[Input Audio (audio file)]:**  
                        Input here the audio you wish to use with "melody" or "sample" mode.

                        - **[Generate (button)]:**  
                        Generates the music with the given settings and prompts.

                        - **[Interrupt (button)]:**  
                        Stops the music generation as soon as it can, providing an incomplete output.

                        - **[Duration (number)]:**  
                        How long you want the generated music to be (in seconds).

                        - **[Overlap (number)]:**  
                        How much each new segment will reference the previous segment (in seconds).  
                        For example, if you choose 20s: Each new segment after the first one will reference the previous segment 20s  
                        and will generate only 10s of new music. The model can only process 30s of music.

                        - **[Seed (number)]:**  
                        Your generated music id. If you wish to generate the exact same music,  
                        place the exact seed with the exact prompts  
                        (This way you can also extend specific song that was generated short).

                        - **[Random Seed (button)]:**  
                        Gives "-1" as a seed, which counts as a random seed.

                        - **[Copy Previous Seed (button)]:**  
                        Copies the seed from the output seed (if you don't feel like doing it manualy).

                        ---

                        ### Customization Tab:

                        - **[Background Color (color)]:**  
                        Works only if you don't upload image. Color of the background of the waveform.

                        - **[Bar Color Start (color)]:**  
                        First color of the waveform bars.

                        - **[Bar Color End (color)]:**  
                        Second color of the waveform bars.

                        - **[Background Image (image)]:**  
                        Background image that you wish to be attached to the generated video along with the waveform.

                        ---

                        ### Settings Tab:

                        - **[Model (selection)]:**  
                        Here you can choose which model you wish to use:  
                        `melody` model is based on the medium model with a unique feature that lets you use melody conditioning  
                        `small` model is trained on 300M parameters  
                        `medium` model is trained on 1.5B parameters  
                        `large` model is trained on 3.3B parameters  
                        `custom` model runs the custom model that you provided.

                        - **[Custom Model (selection)]:**  
                        This dropdown will show you models that are placed in the `models` folder  
                        you must select `custom` in the model options in order to use it.

                        - **[Refresh (button)]:**  
                        Refreshes the dropdown list for custom model.

                        - **[Base Model (selection)]:**  
                        Choose here the model that your custom model is based on.

                        - **[Top-k (number)]:**  
                        is a parameter used in text generation models, including music generation models. It determines the number of most likely next tokens to consider at each step of the generation process. The model ranks all possible tokens based on their predicted probabilities, and then selects the top-k tokens from the ranked list. The model then samples from this reduced set of tokens to determine the next token in the generated sequence. A smaller value of k results in a more focused and deterministic output, while a larger value of k allows for more diversity in the generated music.

                        - **[Top-p (number)]:**  
                        also known as nucleus sampling or probabilistic sampling, is another method used for token selection during text generation. Instead of specifying a fixed number like top-k, top-p considers the cumulative probability distribution of the ranked tokens. It selects the smallest possible set of tokens whose cumulative probability exceeds a certain threshold (usually denoted as p). The model then samples from this set to choose the next token. This approach ensures that the generated output maintains a balance between diversity and coherence, as it allows for a varying number of tokens to be considered based on their probabilities.
                        
                        - **[Temperature (number)]:**  
                        is a parameter that controls the randomness of the generated output. It is applied during the sampling process, where a higher temperature value results in more random and diverse outputs, while a lower temperature value leads to more deterministic and focused outputs. In the context of music generation, a higher temperature can introduce more variability and creativity into the generated music, but it may also lead to less coherent or structured compositions. On the other hand, a lower temperature can produce more repetitive and predictable music.

                        - **[Classifier Free Guidance (number)]:**  
                        refers to a technique used in some music generation models where a separate classifier network is trained to provide guidance or control over the generated music. This classifier is trained on labeled data to recognize specific musical characteristics or styles. During the generation process, the output of the generator model is evaluated by the classifier, and the generator is encouraged to produce music that aligns with the desired characteristics or style. This approach allows for more fine-grained control over the generated music, enabling users to specify certain attributes they want the model to capture.
                        """
                    )
                with gr.Tab("Changelog"):
                    gr.Markdown(
                        """
                        ## Changelog:

                        ### V1.2.2

                        - Added Wiki, Changelog and About tabs



                        ### V1.2.1

                        - Added tabs and organized the entire interface

                        - Added option to attach image to the output video

                        - Added option to load fine-tuned models (Yet to be tested)



                        ### V1.2.0

                        - Added Multi-Prompt



                        ### V1.1.3

                        - Added customization options for generated waveform



                        ### V1.1.2

                        - Removed sample length limit: now you can input audio of any length as music sample



                        ### V1.1.1

                        - Improved music sample audio quality when using music continuation



                        ### V1.1.0

                        - Rebuilt the repo on top of the latest structure of the main MusicGen repo
                        
                        - Improved Music continuation feature



                        ### V1.0.0 - Stable Version

                        - Added Music continuation
                        """
                    )
                with gr.Tab("About"):
                    gr.Markdown(
                        """
                        This is your private demo for [MusicGen](https://github.com/facebookresearch/audiocraft), a simple and controllable model for music generation
                        presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284)
                        
                        ## MusicGen+ is an extended version of the original MusicGen by facebookresearch. 
                        
                        ### Repo: https://github.com/GrandaddyShmax/audiocraft_plus/tree/plus

                        ---
                        
                        ### This project was possible thanks to:

                        #### GrandaddyShmax - https://github.com/GrandaddyShmax

                        #### Camenduru - https://github.com/camenduru

                        #### rkfg - https://github.com/rkfg

                        #### oobabooga - https://github.com/oobabooga
                        """
                    )
        reuse_seed.click(fn=lambda x: x, inputs=[seed_used], outputs=[seed], queue=False)
        submit.click(predict_full, inputs=[model, dropdown, basemodel, s, prompts[0], prompts[1], prompts[2], prompts[3], prompts[4], prompts[5], prompts[6], prompts[7], prompts[8], prompts[9], repeats[0], repeats[1], repeats[2], repeats[3], repeats[4], repeats[5], repeats[6], repeats[7], repeats[8], repeats[9], audio, mode, duration, topk, topp, temperature, cfg_coef, seed, overlap, image, background, bar1, bar2], outputs=[output, seed_used])

        def variable_outputs(k):
            k = int(k) - 1
            return [gr.Textbox.update(visible=True)]*k + [gr.Textbox.update(visible=False)]*(max_textboxes-k)

        s.change(variable_outputs, s, textboxes)
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
            inputs=[text0, audio, model],
            outputs=[output]
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
