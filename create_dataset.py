import os
import argparse
import pandas as pd
from tqdm import tqdm
from src.utils import extract_features, features_dict_to_series


def create_from_folder(folder, out_csv='data/voice_features.csv', recursive=True, label_from='folder'):
    rows = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.wav', '.flac', '.mp3')):
                path = os.path.join(root, file)
                label = None
                if label_from == 'folder':
                    # assume immediate parent folder is the label
                    label = os.path.basename(root)
                elif label_from == 'filename':
                    # assumes filename like label_something.wav
                    label = file.split('_')[0]
                feats = extract_features(path)
                feats['filename'] = file
                feats['label'] = label
                rows.append(feats)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print('Wrote', out_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True, help='folder with audio files')
    parser.add_argument('--out', default='data/voice_features.csv')
    parser.add_argument('--label_from', choices=['folder','filename'], default='folder')
    args = parser.parse_args()
    create_from_folder(args.folder, out_csv=args.out, label_from=args.label_from)
