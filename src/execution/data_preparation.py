"""
    Module data_preparation: Label encoding, K-fold splitting, feature normalization, padding, and tf.data pipelines.
"""

from typing import Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from src import (
    N_SPLITS,
    RANDOM_STATE,
    GLOBAL_BATCH_SIZE,
    TARGET_TIME_FRAMES,
    RAW_TIME_FRAMES
)


class DataPreparation:
    """
        Class DataPreparation: Manages label encoding, cross-validation splitting, normalization, and tf.data generation.
    """

    def __init__(self, n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE, batch_size: int = GLOBAL_BATCH_SIZE):
        """
            Function __init__: Initializes the DataPreparation instance.
            
            Params:
                - n_splits: Number of cross-validation splits.
                - random_state: Random state for reproducibility.
                - batch_size: Batch size for tf.data pipelines.
            
            Returns:
                - None.
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.batch_size = batch_size
        self.encoder = LabelEncoder()
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.classes_ = None

    def fit_transform_labels(self, emotions: pd.Series) -> np.ndarray:
        """
            Function fit_transform_labels: Fit LabelEncoder and transform emotion string labels to integers.
            
            Params:
                - emotions: Pandas Series containing the emotion string labels.
            
            Returns:
                - Numpy array of encoded integer labels.
        """
        y_encoded = self.encoder.fit_transform(emotions)
        self.classes_ = self.encoder.classes_
        print("\n[Data Preparation] Class Mapping:")
        for idx, cls_name in enumerate(self.classes_):
            print(f"  {idx} -> {cls_name}")
        return np.array(y_encoded, dtype=np.int32)

    @property
    def num_classes(self) -> int:
        """
            Function num_classes: Return number of unique emotion classes.
            
            Params:
                - None.
            
            Returns:
                - Number of unique classes as an integer.
        """
        if self.classes_ is None:
            raise RuntimeError("LabelEncoder has not been fitted yet.")
        return len(self.classes_)

    def get_splits(self, X: np.ndarray, y: np.ndarray):
        """
            Function get_splits: Generate StratifiedKFold train/test split indices.
            
            Params:
                - X: Input features numpy array.
                - y: Target labels numpy array.
            
            Returns:
                - Generator of train and test indices splits.
        """
        return self.skf.split(X, y)

    def prepare_fold_tensors(
        self,
        X_mfcc: np.ndarray,
        y_mfcc: np.ndarray,
        train_index: np.ndarray,
        test_index: np.ndarray,
        n_mfcc: int
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray, np.ndarray]:
        """
            Function prepare_fold_tensors: Normalize, pad, reshape and package fold data into tf.data.Datasets.
            
            Params:
                - X_mfcc: Numpy array of MFCC features.
                - y_mfcc: Numpy array of corresponding labels.
                - train_index: Numpy array of training indices.
                - test_index: Numpy array of testing indices.
                - n_mfcc: Number of MFCC coefficients.
            
            Returns:
                - Tuple containing train_dataset, test_dataset, X_test, and y_test.
        """
        # Split with safety cast
        X_train = np.array(X_mfcc[train_index], dtype=np.float32)
        X_test = np.array(X_mfcc[test_index], dtype=np.float32)
        y_train = np.array(y_mfcc[train_index], dtype=np.int32)
        y_test = np.array(y_mfcc[test_index], dtype=np.int32)

        # Standardize with Epsilon to prevent division by zero
        mean_mfcc = np.mean(X_train, dtype=np.float32)
        std_mfcc = np.std(X_train, dtype=np.float32)

        X_train = ((X_train - mean_mfcc) / (std_mfcc + 1e-8)).astype(np.float32)
        X_test = ((X_test - mean_mfcc) / (std_mfcc + 1e-8)).astype(np.float32)

        # Pad and Reshape
        pad_width = TARGET_TIME_FRAMES - RAW_TIME_FRAMES
        X_train = np.pad(X_train, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        X_test = np.pad(X_test, ((0, 0), (0, 0), (0, pad_width)), mode='constant')

        X_train = X_train.reshape(X_train.shape[0], n_mfcc, TARGET_TIME_FRAMES, 1).astype(np.float32)
        X_test = X_test.reshape(X_test.shape[0], n_mfcc, TARGET_TIME_FRAMES, 1).astype(np.float32)

        # I/O Optimization with tf.data.Dataset
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
