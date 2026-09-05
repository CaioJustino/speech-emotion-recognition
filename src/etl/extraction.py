"""Extraction module: Extracts raw audio paths and emotions from CREMA-D, RAVDESS, TESS, and SAVEE."""

import os
import pandas as pd
from src.config import CREMAD_DIR, RAVDESS_DIR, TESS_DIR, SAVEE_DIR


def extract_cremad(cremad_dir: str = CREMAD_DIR) -> pd.DataFrame:
    """Extract emotion labels and file paths from CREMA-D dataset."""
    emotions = []
    paths = []
    
    if os.path.exists(cremad_dir):
        for file in os.listdir(cremad_dir):
            if not file.endswith('.wav'):
                continue
            parts = file.split('_')
            if len(parts) < 3:
                continue
            emotion_code = parts[2]
            
            emotion_map = {
                'SAD': 'sadness',
                'ANG': 'anger',
                'DIS': 'disgust',
                'FEA': 'fear',
                'HAP': 'happiness',
                'NEU': 'neutral',
                'SUR': 'surprise'
            }
            emotion = emotion_map.get(emotion_code, 'Unknown')
            path = os.path.join(cremad_dir, file)
            emotions.append(emotion)
            paths.append(path)

    df = pd.DataFrame({'Emotion': emotions, 'Path': paths})
    df['Dataset'] = 'CREMA-D'
    return df


def extract_ravdess(ravdess_dir: str = RAVDESS_DIR) -> pd.DataFrame:
    """Extract emotion labels and file paths from RAVDESS dataset."""
    emotions = []
    paths = []
    
    if os.path.exists(ravdess_dir):
        for dir_name in os.listdir(ravdess_dir):
            dir_path = os.path.join(ravdess_dir, dir_name)
            if not os.path.isdir(dir_path):
                continue
            for file in os.listdir(dir_path):
                if not file.endswith('.wav'):
                    continue
                parts = file.split('-')
                if len(parts) < 3:
                    continue
                emotion_code = parts[2]
                
                emotion_map = {
                    '01': 'neutral',
                    '02': 'calm',
                    '03': 'happiness',
                    '04': 'sadness',
                    '05': 'anger',
                    '06': 'fear',
                    '07': 'disgust',
                    '08': 'surprise'
                }
                emotion = emotion_map.get(emotion_code, 'Unknown')
                path = os.path.join(dir_path, file)
                emotions.append(emotion)
                paths.append(path)

    df = pd.DataFrame({'Emotion': emotions, 'Path': paths})
    df['Dataset'] = 'RAVDESS'
    return df


def extract_tess(tess_dir: str = TESS_DIR) -> pd.DataFrame:
    """Extract emotion labels and file paths from TESS dataset."""
    emotions = []
    paths = []
    
    if os.path.exists(tess_dir):
        for dir_name in os.listdir(tess_dir):
            dir_path = os.path.join(tess_dir, dir_name)
            if not os.path.isdir(dir_path):
                continue
            for file in os.listdir(dir_path):
                if not file.endswith('.wav'):
                    continue
                file_stem = file.split('.')[0]
                emotion_code = file_stem.split('_')[-1].lower()
                
                emotion_map = {
                    'ps': 'surprise',
                    'sad': 'sadness',
                    'disgust': 'disgust',
                    'angry': 'anger',
                    'happy': 'happiness',
                    'neutral': 'neutral',
                    'fear': 'fear'
                }
                emotion = emotion_map.get(emotion_code, 'Unknown')
                path = os.path.join(dir_path, file)
                emotions.append(emotion)
                paths.append(path)

    df = pd.DataFrame({'Emotion': emotions, 'Path': paths})
    df['Dataset'] = 'TESS'
    return df


def extract_savee(savee_dir: str = SAVEE_DIR) -> pd.DataFrame:
    """Extract emotion labels and file paths from SAVEE dataset."""
    emotions = []
    paths = []
    
    if os.path.exists(savee_dir):
        for file in os.listdir(savee_dir):
            if not file.endswith('.wav'):
                continue
            name_parts = file.split('.')[0].split('_')
            if len(name_parts) < 2:
                continue
            emotion_raw = name_parts[1]
            emotion_code = emotion_raw[:-2].lower()
            
            emotion_map = {
                'a': 'anger',
                'd': 'disgust',
                'f': 'fear',
                'h': 'happiness',
                'n': 'neutral',
                'sa': 'sadness',
                'su': 'surprise'
            }
            emotion = emotion_map.get(emotion_code, 'Unknown')
            path = os.path.join(savee_dir, file)
            emotions.append(emotion)
            paths.append(path)

    df = pd.DataFrame({'Emotion': emotions, 'Path': paths})
    df['Dataset'] = 'SAVEE'
    return df


def extract_all_datasets(
    cremad_dir: str = CREMAD_DIR,
    ravdess_dir: str = RAVDESS_DIR,
    tess_dir: str = TESS_DIR,
    savee_dir: str = SAVEE_DIR
) -> pd.DataFrame:
    """Extract and concatenate records from all four datasets."""
    print("\n[ETL - Extração] Carregando datasets brutos...")
    df_cremad = extract_cremad(cremad_dir)
    df_ravdess = extract_ravdess(ravdess_dir)
    df_tess = extract_tess(tess_dir)
    df_savee = extract_savee(savee_dir)

    print(f"  - CREMA-D: {len(df_cremad)} amostras")
    print(f"  - RAVDESS: {len(df_ravdess)} amostras")
    print(f"  - TESS:    {len(df_tess)} amostras")
    print(f"  - SAVEE:   {len(df_savee)} amostras")

    df_raw = pd.concat([df_cremad, df_ravdess, df_tess, df_savee], axis=0, ignore_index=True)
    print(f"[ETL - Extração] Total bruto coletado: {len(df_raw)} amostras")
    return df_raw

