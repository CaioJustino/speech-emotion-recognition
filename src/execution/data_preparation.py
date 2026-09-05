"""Data Preparation module: Label encoding, K-fold splitting, feature normalization, padding, and tf.data pipelines."""

from typing import Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from src.config import (
    N_SPLITS,
    RANDOM_STATE,
    GLOBAL_BATCH_SIZE,
    TARGET_TIME_FRAMES,
    RAW_TIME_FRAMES
)


class DataPreparation:
    """Manages label encoding, cross-validation splitting, normalization, and tf.data generation."""

    def __init__(self, n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE, batch_size: int = GLOBAL_BATCH_SIZE):
        self.n_splits = n_splits
        self.random_state = random_state
        self.batch_size = batch_size
        self.encoder = LabelEncoder()
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.classes_ = None

    def fit_transform_labels(self, emotions: pd.Series) -> np.ndarray:
        """Fit LabelEncoder and transform emotion string labels to integers."""
        y_encoded = self.encoder.fit_transform(emotions)
        self.classes_ = self.encoder.classes_
        print("\n[Data Preparation] Mapeamento de Classes:")
        for idx, cls_name in enumerate(self.classes_):
            print(f"  {idx} -> {cls_name}")
        return np.array(y_encoded, dtype=np.int32)

    @property
    def num_classes(self) -> int:
        """Return number of unique emotion classes."""
        if self.classes_ is None:
            raise RuntimeError("LabelEncoder ainda não foi ajustado.")
        return len(self.classes_)

    def get_splits(self, X: np.ndarray, y: np.ndarray):
        """Generate StratifiedKFold train/test split indices."""
        return self.skf.split(X, y)

    def prepare_fold_tensors(
        self,
        X_mfcc: np.ndarray,
        y_mfcc: np.ndarray,
        train_index: np.ndarray,
        test_index: np.ndarray,
        n_mfcc: int
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray, np.ndarray]:
        """Normalize, pad, reshape and package fold data into tf.data.Datasets.
        
        Returns:
            train_dataset: Cached, shuffled, batched, prefetched tf.data.Dataset
            test_dataset: Cached, batched, prefetched tf.data.Dataset
            X_test: Padded and reshaped numpy array for model.predict evaluation
            y_test: Test ground truth numpy array
        """
        # Split com cast de segurança
        X_train = np.array(X_mfcc[train_index], dtype=np.float32)
        X_test = np.array(X_mfcc[test_index], dtype=np.float32)
        y_train = np.array(y_mfcc[train_index], dtype=np.int32)
        y_test = np.array(y_mfcc[test_index], dtype=np.int32)

        # Standardize com Epsilon para evitar divisão por zero
        mean_mfcc = np.mean(X_train, dtype=np.float32)
        std_mfcc = np.std(X_train, dtype=np.float32)

        X_train = ((X_train - mean_mfcc) / (std_mfcc + 1e-8)).astype(np.float32)
        X_test = ((X_test - mean_mfcc) / (std_mfcc + 1e-8)).astype(np.float32)

        # Pad e Reshape
        pad_width = TARGET_TIME_FRAMES - RAW_TIME_FRAMES
        X_train = np.pad(X_train, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        X_test = np.pad(X_test, ((0, 0), (0, 0), (0, pad_width)), mode='constant')

        X_train = X_train.reshape(X_train.shape[0], n_mfcc, TARGET_TIME_FRAMES, 1).astype(np.float32)
        X_test = X_test.reshape(X_test.shape[0], n_mfcc, TARGET_TIME_FRAMES, 1).astype(np.float32)

        # Otimização de I/O com tf.data.Dataset
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .cache()
            .shuffle(buffer_size=len(X_train))
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        test_dataset = (
            tf.data.Dataset.from_tensor_slices((X_test, y_test))
            .cache()
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        return train_dataset, test_dataset, X_test, y_test

