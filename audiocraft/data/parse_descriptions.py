import argparse
import json
import os
from glob import iglob


def parse_descriptions(sample_dict, source='bbc'):
    desc = None
    if source == 'bbc':
        desc = sample_dict['text']
    elif source == 'epidemic':
        desc = ', '.join(sample_dict['text'])
    elif source == 'audiostock':
        desc = ', '.join(sample_dict['text'])
    elif source == 'freesound':
        return sample_dict
    sample_dict['description'] = desc
    return sample_dict


def parse_file(in_path, out_path, source='bbc'):
    for f in iglob(f'{in_path}/*.json', recursive=True):
        with open(f, 'r') as f_in:
            sample_dict = json.load(f_in)
        sample_dict = parse_descriptions(sample_dict, source)
        with open(f'{out_path}/{f.split("/")[-1]}', 'w') as f_out:
            json.dump(sample_dict, f_out)

def filter_empty(in_path, out_path):
    for l in open(in_path, 'r'):
        f = json.load(open(json.loads(l)['path'].replace('flac', 'json'), 'r'))
        if f['description']:
            with open(out_path, 'a') as f_out:
                f_out.write(l)
        else:
            print('bad description', f['description'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse descriptions')
    parser.add_argument("-i", "--in_path", type=str, help="Path to the input file")
    parser.add_argument("-o", "--out_path", type=str, help="Path to the output file")
    parser.add_argument("-s", "--source", type=str, default='bbc', help="Source of the data")
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    parse_file(args.in_path, args.out_path, args.source)

    # if os.path.exists(args.out_path):
    #     os.remove(args.out_path)
    # filter_empty(args.in_path, args.out_path)