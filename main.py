#!/usr/bin/env python3
"""Speech Emotion Recognition (SER) using CNNs and CRNNs Based on Mel Frequency Cepstral Coefficients (MFCCs).

Main CLI Entrypoint for MLOps Pipeline.
"""

import sys
import argparse
from pathlib import Path

# Garantir que o diretório raiz esteja no PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    MFCC_LIST,
    EPOCHS,
    GLOBAL_BATCH_SIZE,
    TARGET_SAMPLES_PER_CLASS,
    DROPPED_EMOTIONS,
    RANDOM_STATE
)
from src.pipeline import run_pipeline


def parse_args():
    """Parse command line arguments for pipeline customization."""
    parser = argparse.ArgumentParser(
        description="Speech Emotion Recognition (SER) MLOps Pipeline"
    )
    parser.add_argument(
        "--mfcc-list",
        nargs="+",
        type=int,
        default=MFCC_LIST,
        help="Lista de coeficientes MFCC a avaliar (padrão: todos os 12 valores originais)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Número de épocas de treino por fold (padrão: {EPOCHS})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=GLOBAL_BATCH_SIZE,
        help=f"Tamanho do batch global (padrão: {GLOBAL_BATCH_SIZE})"
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=TARGET_SAMPLES_PER_CLASS,
        help=f"Quantidade alvo para undersampling de cada emoção (padrão: {TARGET_SAMPLES_PER_CLASS})"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help=f"Semente aleatória para amostragem e estratificação (padrão: {RANDOM_STATE})"
    )
    return parser.parse_args()


def main():
    """Main execution function."""
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

