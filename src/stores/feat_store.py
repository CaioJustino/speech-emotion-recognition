"""
    Module feat_store: Manages dataset metadata storage, in-memory audio waveforms, and MFCC feature extraction.
"""

import os
from typing import List, Optional
import numpy as np
import pandas as pd
import librosa
from joblib import Parallel, delayed

from src import (
    FEAT_STORE_DIR,
    SAMPLE_RATE,
    AUDIO_DURATION
)


def load_single_audio(path: str, sr: int = SAMPLE_RATE, duration: int = AUDIO_DURATION) -> np.ndarray:
    """
        Function load_single_audio: Load a single audio file and pad to exact target duration.
        
        Params:
            - path: The path to the audio file.
            - sr: The sample rate to use for loading.
            - duration: The target duration in seconds.
        
        Returns:
            - The loaded and padded audio waveform as a numpy array.
    """
    audio, _ = librosa.load(path, sr=sr, duration=duration, mono=True)
    target_length = duration * sr
    if len(audio) < target_length:
        audio = np.pad(audio, pad_width=(0, target_length - len(audio)), mode='constant')
    return audio


def extract_mfcc_from_array(audio_array: np.ndarray, sr: int, n_mfcc_val: int) -> np.ndarray:
    """
        Function extract_mfcc_from_array: Extract MFCC features from a pre-loaded audio waveform array in memory.
        
        Params:
            - audio_array: The audio waveform array.
            - sr: The sample rate of the audio.
            - n_mfcc_val: The number of MFCC features to extract.
        
        Returns:
            - The extracted MFCC features as a numpy array.
    """
    n_mels_val = max(128, n_mfcc_val)
    signal = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc_val, n_mels=n_mels_val)
    return np.array(signal, dtype=np.float32)


class FeatStore:
    """
        Class FeatStore: Feature Store abstraction for managing raw audio loading, feature caching, and extraction.
    """

    def __init__(self, store_dir: str = FEAT_STORE_DIR):
        """
            Function __init__: Initialize the FeatStore instance.
            
            Params:
                - store_dir: The directory path for the feature store.
            
            Returns:
                - None.
        """
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)
        self.metadata_path = os.path.join(self.store_dir, "metadata_balanced.csv")
        self._raw_audios: Optional[List[np.ndarray]] = None

    def save_metadata(self, df: pd.DataFrame) -> str:
        """
            Function save_metadata: Persist metadata dataframe to feature store.
            
            Params:
                - df: The metadata dataframe to save.
            
            Returns:
                - The path where the metadata was saved.
        """
        df.to_csv(self.metadata_path, index=False)
        return self.metadata_path

    def load_metadata(self) -> pd.DataFrame:
        """
            Function load_metadata: Load stored balanced metadata from feature store.
            
            Params:
                None.
            
            Returns:
                - The loaded metadata dataframe.
        """
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata not found in Feat Store at: {self.metadata_path}")
        return pd.read_csv(self.metadata_path)

    def load_audios_to_ram(self, paths: List[str], n_jobs: int = -1) -> List[np.ndarray]:
        """
            Function load_audios_to_ram: Load all audio files into RAM using parallel execution.
            
            Params:
                - paths: A list of paths to the audio files.
                - n_jobs: The number of jobs to run in parallel.
            
            Returns:
                - A list of loaded audio arrays.
        """
        print(f"\n[Feat Store] Loading {len(paths)} audios into RAM memory (n_jobs={n_jobs})...")
        self._raw_audios = Parallel(n_jobs=n_jobs)(
            delayed(load_single_audio)(path, SAMPLE_RATE, AUDIO_DURATION) for path in paths
        )
        print(f"[Feat Store] {len(self._raw_audios)} audio files successfully loaded into memory.")
        return self._raw_audios

    def get_raw_audios(self) -> List[np.ndarray]:
        """
            Function get_raw_audios: Return the pre-loaded raw audio list from memory.
            
            Params:
                None.
            
            Returns:
                - The list of pre-loaded raw audio arrays.
        """
        if self._raw_audios is None:
            raise RuntimeError("Audios have not been loaded into RAM yet. Call load_audios_to_ram() first.")
        return self._raw_audios

    def extract_mfccs(self, n_mfcc: int, n_jobs: int = -1) -> np.ndarray:
        """
            Function extract_mfccs: Extract MFCCs for all in-memory audios for a given n_mfcc.
            
            Params:
                - n_mfcc: The number of MFCC features to extract.
                - n_jobs: The number of jobs to run in parallel.
            
            Returns:
                - A numpy array containing the extracted MFCC features.
        """
        raw_audios = self.get_raw_audios()
        print(f"[Feat Store] Extracting MFCCs (n_mfcc={n_mfcc}) from RAM memory...")
        features = Parallel(n_jobs=n_jobs)(
            delayed(extract_mfcc_from_array)(audio, SAMPLE_RATE, n_mfcc) for audio in raw_audios
        )
        return np.array(features, dtype=np.float32)
