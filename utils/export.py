from audiocraft.utils import export
from audiocraft import train

# Export the MusicGen model
xp = train.main.get_xp_from_sig('5eda0d60')
export.export_lm(xp.folder / 'checkpoint.th', '../export/state_dict.bin')

# Export the pretrained EnCodec model
export.export_pretrained_compression_model('facebook/encodec_32khz', '/home/maxwell/grimes_keys_model/compression_state_dict.bin')
