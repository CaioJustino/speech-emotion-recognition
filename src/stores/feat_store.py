"""FeatStore module: Manages dataset metadata storage, in-memory audio waveforms, and MFCC feature extraction."""

import os
from typing import List, Optional
import numpy as np
import pandas as pd
import librosa
from joblib import Parallel, delayed

from src.config import (
    FEAT_STORE_DIR,
    SAMPLE_RATE,
    AUDIO_DURATION
)


def load_single_audio(path: str, sr: int = SAMPLE_RATE, duration: int = AUDIO_DURATION) -> np.ndarray:
    """Load a single audio file and pad to exact target duration."""
    audio, _ = librosa.load(path, sr=sr, duration=duration, mono=True)
    target_length = duration * sr
    if len(audio) < target_length:
        audio = np.pad(audio, pad_width=(0, target_length - len(audio)), mode='constant')
    return audio


def extract_mfcc_from_array(audio_array: np.ndarray, sr: int, n_mfcc_val: int) -> np.ndarray:
    """Extract MFCC features from a pre-loaded audio waveform array in memory."""
    n_mels_val = max(128, n_mfcc_val)
    signal = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc_val, n_mels=n_mels_val)
    return np.array(signal, dtype=np.float32)


class FeatStore:
    """Feature Store abstraction for managing raw audio loading, feature caching, and extraction."""

    def __init__(self, store_dir: str = FEAT_STORE_DIR):
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)
        self.metadata_path = os.path.join(self.store_dir, "metadata_balanced.csv")
        self._raw_audios: Optional[List[np.ndarray]] = None

    def save_metadata(self, df: pd.DataFrame) -> str:
        """Persist metadata dataframe to feature store."""
        df.to_csv(self.metadata_path, index=False)
        return self.metadata_path

    def load_metadata(self) -> pd.DataFrame:
        """Load stored balanced metadata from feature store."""
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadados não encontrados no Feat Store em: {self.metadata_path}")
        return pd.read_csv(self.metadata_path)

    def load_audios_to_ram(self, paths: List[str], n_jobs: int = -1) -> List[np.ndarray]:
        """Load all audio files into RAM using parallel execution."""
        print(f"\n[Feat Store] Carregando {len(paths)} áudios na memória RAM (n_jobs={n_jobs})...")
        self._raw_audios = Parallel(n_jobs=n_jobs)(
            delayed(load_single_audio)(path, SAMPLE_RATE, AUDIO_DURATION) for path in paths
        )
        print(f"[Feat Store] {len(self._raw_audios)} arquivos de áudio carregados na memória com sucesso.")
        return self._raw_audios

    def get_raw_audios(self) -> List[np.ndarray]:
        """Return the pre-loaded raw audio list from memory."""
        if self._raw_audios is None:
            raise RuntimeError("Áudios ainda não foram carregados na RAM. Chame load_audios_to_ram() primeiro.")
        return self._raw_audios

    def extract_mfccs(self, n_mfcc: int, n_jobs: int = -1) -> np.ndarray:
        """Extract MFCCs for all in-memory audios for a given n_mfcc."""
        raw_audios = self.get_raw_audios()
        print(f"[Feat Store] Extraindo MFCCs (n_mfcc={n_mfcc}) da Memória RAM...")
        features = Parallel(n_jobs=n_jobs)(
            delayed(extract_mfcc_from_array)(audio, SAMPLE_RATE, n_mfcc) for audio in raw_audios
        )
        return np.array(features, dtype=np.float32)

