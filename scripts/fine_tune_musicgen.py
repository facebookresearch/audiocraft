import argparse
import os
import zipfile
from pathlib import Path

import torch
from audiocraft.data.audio_dataset import AudioDataset
from audiocraft.models.musicgen import MusicGen


def prepare_audio_files(zip_path: str, output_dir: str, target_sample_rate: int = 32000, target_channels: int = 2):
    """Prepare audio files from a zip of full songs.

    Args:
        zip_path (str): Path to the zip file containing full songs.
        output_dir (str): Directory to save the prepared audio files.
        target_sample_rate (int): Target sample rate for the audio files.
        target_channels (int): Target number of channels for the audio files.
    """
    AudioDataset.prepare_audio_files(zip_path, output_dir, target_sample_rate, target_channels)


def fine_tune_musicgen(model_name: str, data_dir: str, epochs: int = 10, batch_size: int = 16, learning_rate: float = 1e-4):
    """Fine-tune MusicGen model with custom music.

    Args:
        model_name (str): Name of the pre-trained MusicGen model.
        data_dir (str): Directory containing the prepared audio files.
        epochs (int): Number of epochs to fine-tune the model.
        batch_size (int): Batch size for fine-tuning.
        learning_rate (float): Learning rate for fine-tuning.
    """
    model = MusicGen.get_pretrained(model_name)
    dataset = AudioDataset.from_path(data_dir, segment_duration=30, sample_rate=32000, channels=2)
    model.fine_tune(dataset, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MusicGen with custom music")
    parser.add_argument("--zip_path", type=str, required=True, help="Path to the zip file containing full songs")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the prepared audio files")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained MusicGen model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the prepared audio files")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to fine-tune the model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for fine-tuning")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for fine-tuning")
    args = parser.parse_args()

    prepare_audio_files(args.zip_path, args.output_dir)
    fine_tune_musicgen(args.model_name, args.data_dir, args.epochs, args.batch_size, args.learning_rate)


if __name__ == "__main__":
    main()
