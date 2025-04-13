import gradio as gr
import os
import zipfile
from pathlib import Path
import torch
from audiocraft.data.audio_dataset import AudioDataset
from audiocraft.models.musicgen import MusicGen

def prepare_audio_files(zip_path: str, output_dir: str, target_sample_rate: int = 32000, target_channels: int = 2):
    """Prepare audio files from a zip of full songs."""
    AudioDataset.prepare_audio_files(zip_path, output_dir, target_sample_rate, target_channels)

def fine_tune_musicgen(model_name: str, data_dir: str, epochs: int = 10, batch_size: int = 16, learning_rate: float = 1e-4):
    """Fine-tune MusicGen model with custom music."""
    model = MusicGen.get_pretrained(model_name)
    dataset = AudioDataset.from_path(data_dir, segment_duration=30, sample_rate=32000, channels=2)
    model.fine_tune(dataset, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)

def prepare_and_fine_tune(zip_path, output_dir, model_name, epochs, batch_size, learning_rate):
    prepare_audio_files(zip_path, output_dir)
    fine_tune_musicgen(model_name, output_dir, epochs, batch_size, learning_rate)
    return "Fine-tuning completed."

with gr.Blocks() as demo:
    gr.Markdown("# Fine-tune MusicGen with Custom Music")
    with gr.Row():
        with gr.Column():
            zip_path = gr.Textbox(label="Path to Zip File", placeholder="Enter the path to the zip file containing full songs")
            output_dir = gr.Textbox(label="Output Directory", placeholder="Enter the directory to save the prepared audio files")
            model_name = gr.Textbox(label="Model Name", placeholder="Enter the name of the pre-trained MusicGen model")
            epochs = gr.Number(label="Number of Epochs", value=10)
            batch_size = gr.Number(label="Batch Size", value=16)
            learning_rate = gr.Number(label="Learning Rate", value=1e-4)
            fine_tune_button = gr.Button("Fine-tune MusicGen")
        with gr.Column():
            output = gr.Textbox(label="Output")

    fine_tune_button.click(prepare_and_fine_tune, inputs=[zip_path, output_dir, model_name, epochs, batch_size, learning_rate], outputs=output)

demo.launch()
