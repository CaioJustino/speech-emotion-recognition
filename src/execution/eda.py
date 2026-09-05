"""EDA module: Exploratory Data Analysis for Speech Emotion Recognition dataset."""

from typing import Dict, Any
import pandas as pd


def run_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform exploratory data analysis on the balanced SER dataset.
    
    Args:
        df: Balanced DataFrame containing at least 'Emotion' and 'Path' columns.
        
    Returns:
        Dict[str, Any]: Summary dictionary containing counts, percentages, and distribution stats.
    """
    print("\n" + "=" * 55)
    print(" [EDA] Análise Exploratória de Dados")
    print("=" * 55)

    total_samples = len(df)
    class_counts = df['Emotion'].value_counts().to_dict()
    num_classes = len(class_counts)
    
    print(f"Total de amostras: {total_samples}")
    print(f"Número de classes (emoções): {num_classes}")
    print("\nDistribuição de Classes:")
    for emo, count in sorted(class_counts.items()):
        pct = (count / total_samples) * 100
        print(f"  - {emo:<12}: {count:>5} amostras ({pct:.2f}%)")

    dataset_breakdown = {}
    if 'Dataset' in df.columns:
        print("\nDistribuição por Dataset de Origem:")
        dataset_counts = df['Dataset'].value_counts().to_dict()
        for ds, count in sorted(dataset_counts.items()):
            pct = (count / total_samples) * 100
            print(f"  - {ds:<12}: {count:>5} amostras ({pct:.2f}%)")
        dataset_breakdown = dataset_counts

    eda_summary = {
        "total_samples": total_samples,
        "num_classes": num_classes,
        "class_distribution": class_counts,
        "dataset_breakdown": dataset_breakdown,
        "is_perfectly_balanced": len(set(class_counts.values())) == 1
    }

    print(f"Dataset perfeitamente balanceado? {'SIM' if eda_summary['is_perfectly_balanced'] else 'NÃO'}")
    print("=" * 55 + "\n")
    return eda_summary

