"""
    Module ModelRegistry: Saves, tracks, and manages trained model artifacts.
"""

import os
import tensorflow as tf
from src import MODELS_DIR


class ModelRegistry:
    """
        Class ModelRegistry: Model Registry abstraction for saving and tracking trained Keras models.
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

    def save_model(self, model: tf.keras.Model, model_type: str, n_mfcc: int) -> str:
        """
            Function save_model: Save a trained model in Keras native format (.keras).
            
            Params:
                - model: Compiled/trained tf.keras Model.
                - model_type: 'cnn' or 'crnn'.
                - n_mfcc: Number of MFCC features.
            
            Returns:
                - Path where model was saved.
        """
        model_type_clean = model_type.lower()
        filename = f"emotion_recognition_mfcc_{model_type_clean}_{n_mfcc}.keras"
        save_path = os.path.join(self.models_dir, filename)
        model.save(save_path)
        print(f"[Model Registry] Model registered successfully: {save_path}")
        return save_path

    def get_model_path(self, model_type: str, n_mfcc: int) -> str:
        """
            Function get_model_path: Get expected path for a saved model.
            
            Params:
                - model_type: 'cnn' or 'crnn'.
                - n_mfcc: Number of MFCC features.
            
            Returns:
                - Expected path for a saved model.
        """
        model_type_clean = model_type.lower()
        filename = f"emotion_recognition_mfcc_{model_type_clean}_{n_mfcc}.keras"
        return os.path.join(self.models_dir, filename)
