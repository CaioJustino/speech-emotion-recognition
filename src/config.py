"""Configuration module for Speech Emotion Recognition (SER) Pipeline."""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
CREMAD_DIR = os.path.join(DATA_DIR, "cremad", "AudioWAV")
RAVDESS_DIR = os.path.join(DATA_DIR, "ravdess")
TESS_DIR = os.path.join(DATA_DIR, "tess", "tess")
SAVEE_DIR = os.path.join(DATA_DIR, "savee")

# Storage & Output Paths
FEAT_STORE_DIR = os.path.join(DATA_DIR, "feat_store")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METADATA_DIR = os.path.join(RESULTS_DIR, "metadata")
RESULTS_CNN_DIR = os.path.join(RESULTS_DIR, "MFCC_CNN")
RESULTS_CRNN_DIR = os.path.join(RESULTS_DIR, "MFCC_CRNN")
CONSOLIDATED_RESULTS_PATH = os.path.join(RESULTS_DIR, "results_consolidated.csv")

# Audio Processing Settings
SAMPLE_RATE = 44100
AUDIO_DURATION = 4  # seconds
TARGET_TIME_FRAMES = 352
RAW_TIME_FRAMES = 345
PAD_WIDTH = TARGET_TIME_FRAMES - RAW_TIME_FRAMES  # 7 frames

# Undersampling & Dataset Settings
TARGET_SAMPLES_PER_CLASS = 1703
DROPPED_EMOTIONS = ["calm", "surprise"]
RANDOM_STATE = 42

# Training Hyperparameters
GLOBAL_BATCH_SIZE = 128
EPOCHS = 100
N_SPLITS = 10
LEARNING_RATE = 0.001
MIN_LR = 0.000001
PATIENCE_EARLY_STOPPING = 10
PATIENCE_LR_SCHEDULER = 3
LR_REDUCE_FACTOR = 0.2

MFCC_LIST = [12, 13, 14, 25, 39, 65, 96, 128, 192, 255, 256, 257]

# Hardware & TensorFlow Optimization
TF_CUDNN_USE_AUTOTUNE = "1"
TF_XLA_FLAGS = "--tf_xla_enable_xla_devices"
MIXED_PRECISION_POLICY = "mixed_float16"

