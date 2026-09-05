"""Transformation module: Cleans labels, removes unwanted emotions, and applies balanced undersampling."""

from typing import List, Optional
import pandas as pd
from src.config import TARGET_SAMPLES_PER_CLASS, DROPPED_EMOTIONS, RANDOM_STATE


def transform_and_undersample(
    df: pd.DataFrame,
    target_samples_per_class: int = TARGET_SAMPLES_PER_CLASS,
    drop_emotions: Optional[List[str]] = None,
    random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Clean labels, drop specified emotions, and apply random undersampling.
    
    Args:
        df: Raw DataFrame containing 'Emotion' and 'Path' columns.
        target_samples_per_class: Exact number of samples to keep per emotion.
        drop_emotions: List of emotion labels to remove completely (e.g. ['calm', 'surprise']).
        random_state: Seed for reproducibility of random sampling and shuffle.
        
    Returns:
        pd.DataFrame: Cleaned, balanced, and shuffled DataFrame.
    """
    if drop_emotions is None:
        drop_emotions = DROPPED_EMOTIONS

    print("\n[ETL - Transformação] Iniciando limpeza e undersampling dos dados...")
    print("\nContagem de amostras brutas por emoção:")
    print(df['Emotion'].value_counts())

    # 1. Remover emoções indesejadas (ex: calm, surprise) e amostras desconhecidas
    df_filtered = df[~df['Emotion'].isin(drop_emotions) & (df['Emotion'] != 'Unknown')].copy()
    df_filtered = df_filtered.reset_index(drop=True)

    print(f"\nContagem após remover {drop_emotions}:")
    print(df_filtered['Emotion'].value_counts())

    # 2. Aplicar Undersampling com escolha aleatória (shuffle) para cada emoção
    sampled_groups = []
    unique_emotions = sorted(df_filtered['Emotion'].unique())

    print(f"\nAplicando undersampling para {target_samples_per_class} amostras por classe:")
    for emotion in unique_emotions:
        group = df_filtered[df_filtered['Emotion'] == emotion]
        count = len(group)
        if count < target_samples_per_class:
            raise ValueError(
                f"A classe '{emotion}' possui {count} amostras, "
                f"quantidade insuficiente para o undersampling solicitado ({target_samples_per_class})."
            )
        # Seleção aleatória das 1703 amostras
        sampled = group.sample(n=target_samples_per_class, random_state=random_state)
        print(f"  - {emotion}: selecionadas {len(sampled)} de {count} amostras.")
        sampled_groups.append(sampled)

    # 3. Concatenar e embaralhar (shuffle) o dataset balanceado completo
    df_balanced = pd.concat(sampled_groups, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print(f"\n[ETL - Transformação] Dataset final balanceado: {len(df_balanced)} amostras no total.")
    print("Distribuição final por classe:")
    print(df_balanced['Emotion'].value_counts())

    return df_balanced

