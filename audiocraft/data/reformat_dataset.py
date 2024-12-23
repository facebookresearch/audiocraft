import json
import glob
import argparse
import os
import re


def reformat_dataset(in_path, source='odeon'):
    if source == 'odeon':
        by_filename = {'Cinematic', 'Crowd', 'Household', 'Humans', 'Jungle', 'Machines', 'Technology', 'Vehicles',
                       'Weather'}
        ext = 'Wav'
        files = glob.iglob(f'{in_path}/**/*.{ext}', recursive=True)
        for f in files:
            cls = f.split('ODEONSoundEffects/')[-1].split('/')[0].replace('_', ' ')
            if cls not in by_filename:
                tag = cls
            else:
                n = f.split('ODEONSoundEffects/')[-1].split('/')[-1][:-4]
                tag = cls + ', ' + ' '.join(re.findall(r'[A-Z][a-z]*', n)).lower()
            out = {'tags': [cls], 'description': tag}
            with open(f.replace(f'.{ext}', '.json'), 'w') as f:
                json.dump(out, f)
    elif source == 'macs':
        import yaml
        with open(f"{in_path}/MACS.yaml") as stream:
            try:
                labels = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        for l in labels['files']:
            l['descriptions'] = [a['sentence'] for a in l['annotations']]
            with open(f"{in_path}/{l['filename'].replace('.wav', '.json')}", 'w') as f:
                json.dump(l, f)
    elif source == 'audiocaps':
        import pandas as pd
        split = in_path.split('/')[-1]
        df = pd.read_csv(f'{in_path}/{split}.csv')
        df['path'] = df.apply(lambda x: f'{x.youtube_id}_{x.start_time*1000}_{(x.start_time + 10)*1000}.wav', axis=1)
        df['start_time'] = df['start_time'].astype(str)
        df['audiocap_id'] = df['audiocap_id'].astype(str)
        group_columns = ["youtube_id", "start_time", "path"]
        list_columns = ["caption", "audiocap_id"]

        # Group and aggregate
        df = (
            df.groupby(group_columns, as_index=False)
            .agg({col: list for col in list_columns})
        )

        out_path = f'{in_path}/captions/'
        for f in os.listdir(f'{in_path}/wav/'):
            if not f.endswith('.wav'):
                continue
            cur = df[df['path'] == f]
            if len(cur) != 1:
                print(f"Expected 1 file but got {len(cur)} for {f}")
                continue
            # assert len(cur) == 1, f"Expected 1 file but got {len(cur)} for {f}"
            cur = cur.iloc[0]
            out = {'description': cur['caption'], 'filename': cur['path'], 'audiocap_id': cur['audiocap_id'],
                   'youtube_id': cur['youtube_id'], 'start_time': cur['start_time']}
            with open(f'{out_path}/{f.replace(".wav", ".json")}', 'w') as f_out:
                json.dump(out, f_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SALMon')
    parser.add_argument("-i", "--in_path", type=str, help="Path to the data")
    parser.add_argument("-s", "--source", default='odeon', type=str, help="Which dataset")
    args = parser.parse_args()
    reformat_dataset(args.in_path, args.source)
