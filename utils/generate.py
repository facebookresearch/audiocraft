import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('/home/maxwell/grimes_drums_model')
model.set_generation_params(duration=60)  # generate 8 seconds.
descriptions = ['dance drum beat, percussiveness high, business little, brightness maximal, variance little, bass little, mids great, highs ample, BPM tempo high, noisiness little', 'dance electronic drums, percussiveness little, business maximal, brightness just above mean, variance maximal, bass just above mean, mids great, highs ample, BPM tempo little, noisiness just below mean']
wav = model.generate(descriptions)  # generates 3 samples.

for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
            audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
