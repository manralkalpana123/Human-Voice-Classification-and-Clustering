import os
import json
import numpy as np
import librosa


def extract_features(file_path, sr=22050, n_mfcc=13):
    """
    Extract a consistent set of features from an audio file.
    Returns a dict {feature_name: value} (numeric scalars — mean/std aggregated).
    """
    y, sr = librosa.load(file_path, sr=sr)
    result = {}

    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for i in range(mfcc.shape[0]):
        result[f'mfcc_mean_{i+1}'] = float(np.mean(mfcc[i]))
        result[f'mfcc_std_{i+1}'] = float(np.std(mfcc[i]))

    
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        result['chroma_mean'] = float(np.mean(chroma))
        result['chroma_std'] = float(np.std(chroma))
    except Exception:
        result['chroma_mean'] = 0.0
        result['chroma_std'] = 0.0

   
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    result['spec_centroid_mean'] = float(np.mean(spec_centroid))
    result['spec_centroid_std'] = float(np.std(spec_centroid))

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    result['spec_bandwidth_mean'] = float(np.mean(spec_bw))
    result['spec_bandwidth_std'] = float(np.std(spec_bw))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    result['rolloff_mean'] = float(np.mean(rolloff))
    result['rolloff_std'] = float(np.std(rolloff))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    result['contrast_mean'] = float(np.mean(contrast))
    result['contrast_std'] = float(np.std(contrast))

    
    
    try:
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        result['tonnetz_mean'] = float(np.mean(tonnetz))
        result['tonnetz_std'] = float(np.std(tonnetz))
    except Exception:
        result['tonnetz_mean'] = 0.0
        result['tonnetz_std'] = 0.0

    
    zcr = librosa.feature.zero_crossing_rate(y)
    result['zcr_mean'] = float(np.mean(zcr))
    result['zcr_std'] = float(np.std(zcr))

    
    result['duration'] = float(librosa.get_duration(y=y, sr=sr))

    return result


def features_dict_to_series(d):
    import pandas as pd
    return pd.Series(d)


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)