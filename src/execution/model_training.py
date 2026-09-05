"""Model Training module: Hardware acceleration setup, model architectures (CNN, CRNN), callbacks and training procedures."""

import os
from typing import List
import tensorflow as tf
from tensorflow.keras import mixed_precision

from src.config import (
    TF_CUDNN_USE_AUTOTUNE,
    TF_XLA_FLAGS,
    MIXED_PRECISION_POLICY,
    LEARNING_RATE,
    PATIENCE_EARLY_STOPPING,
    PATIENCE_LR_SCHEDULER,
    LR_REDUCE_FACTOR,
    MIN_LR,
    TARGET_TIME_FRAMES
)


def setup_training_environment() -> tf.distribute.Strategy:
    """Initialize hardware acceleration flags, mixed precision, and distributed strategy."""
    # 1. ATIVADO: Permite que a GPU descubra o algoritmo mais rápido para as CNNs
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = TF_CUDNN_USE_AUTOTUNE
    # 2. ATIVADO: Liga o compilador XLA para fundir operações matemáticas
    os.environ['TF_XLA_FLAGS'] = TF_XLA_FLAGS
    try:
        tf.config.optimizer.set_jit(True)
    except Exception as e:
        print(f"Aviso ao ativar JIT XLA: {e}")

    # 3. ATIVADO: Mixed Precision para acelerar processamento na Tesla V100 / GPUs Tensor Core
    policy = mixed_precision.Policy(MIXED_PRECISION_POLICY)
    mixed_precision.set_global_policy(policy)
    print(f"\n[Hardware] Compute dtype: {policy.compute_dtype}")
    print(f"[Hardware] Variable dtype: {policy.variable_dtype}\n")

    physical_devices = tf.config.list_physical_devices('GPU')
    print("[Hardware] Num GPUs Available: ", len(physical_devices))
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Erro ao configurar memory growth: {e}")

    strategy = tf.distribute.MirroredStrategy()
    print('[Hardware] Num Devices in Strategy: {}'.format(strategy.num_replicas_in_sync))
    return strategy


def get_callbacks(
    patience_es: int = PATIENCE_EARLY_STOPPING,
    patience_lr: int = PATIENCE_LR_SCHEDULER,
    factor: float = LR_REDUCE_FACTOR,
    min_lr: float = MIN_LR
) -> List[tf.keras.callbacks.Callback]:
    """Return EarlyStopping and ReduceLROnPlateau callbacks."""
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience_es,
        restore_best_weights=True,
        mode='min'
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=factor,
        patience=patience_lr,
        min_lr=min_lr,
        mode='min'
    )
    return [early_stop, lr_scheduler]


def build_cnn_model(n_mfcc: int, num_classes: int, learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    """Build and compile 2D CNN model architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(n_mfcc, TARGET_TIME_FRAMES, 1), padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_crnn_model(n_mfcc: int, num_classes: int, learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    """Build and compile CRNN (CNN + Bidirectional LSTM) model architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(n_mfcc, TARGET_TIME_FRAMES, 1), padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Reshape((1, 128)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

