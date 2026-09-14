#!/usr/bin/env python3
"""
    Module main: Speech Emotion Recognition (SER) using CNNs and CRNNs Based on Mel Frequency Cepstral Coefficients (MFCCs). Main CLI Entrypoint for MLOps Pipeline.
"""

import sys
import argparse
from pathlib import Path

# Ensure the root directory is in PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import (
    MFCC_LIST,
    EPOCHS,
    GLOBAL_BATCH_SIZE,
    TARGET_SAMPLES_PER_CLASS,
    DROPPED_EMOTIONS,
    RANDOM_STATE
)
from src.pipeline import run_pipeline


def parse_args():
    """
        Function parse_args: Parse command line arguments for pipeline customization.
        
        Params:
            - None
        
        Returns:
            - Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Speech Emotion Recognition (SER) MLOps Pipeline"
    )
    parser.add_argument(
        "--mfcc-list",
        nargs="+",
        type=int,
        default=MFCC_LIST,
        help="List of MFCC coefficients to evaluate (default: all 12 original values)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs per fold (default: {EPOCHS})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=GLOBAL_BATCH_SIZE,
        help=f"Global batch size (default: {GLOBAL_BATCH_SIZE})"
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=TARGET_SAMPLES_PER_CLASS,
        help=f"Target amount for undersampling each emotion (default: {TARGET_SAMPLES_PER_CLASS})"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help=f"Random seed for sampling and stratification (default: {RANDOM_STATE})"
    )
    return parser.parse_args()


def main():
    """
        Function main: Main execution function.
        
        Params:
            - None
        
        Returns:
            - None
    """
    args = parse_args()
    
    run_pipeline(
        mfcc_list=args.mfcc_list,
        epochs=args.epochs,
        batch_size=args.batch_size,
        target_samples_per_class=args.samples_per_class,
        drop_emotions=DROPPED_EMOTIONS,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()
