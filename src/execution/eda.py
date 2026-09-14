"""
    Module eda: Exploratory Data Analysis for Speech Emotion Recognition dataset.
"""

from typing import Dict, Any
import pandas as pd


def run_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Function run_eda: Perform exploratory data analysis on the balanced SER dataset.
    
    Params:
        - df: Balanced DataFrame containing at least 'Emotion' and 'Path' columns.
    
    Returns:
        - Summary dictionary containing counts, percentages, and distribution stats.
    """
    print("\n" + "=" * 55)
    print(" [EDA] Exploratory Data Analysis")
    print("=" * 55)

    total_samples = len(df)
    class_counts = df['Emotion'].value_counts().to_dict()
    num_classes = len(class_counts)
    
    print(f"Total samples: {total_samples}")
    print(f"Number of classes (emotions): {num_classes}")
    print("\nClass Distribution:")
    for emo, count in sorted(class_counts.items()):
        pct = (count / total_samples) * 100
        print(f"  - {emo:<12}: {count:>5} samples ({pct:.2f}%)")

    dataset_breakdown = {}
    if 'Dataset' in df.columns:
        print("\nDistribution by Origin Dataset:")
        dataset_counts = df['Dataset'].value_counts().to_dict()
        for ds, count in sorted(dataset_counts.items()):
            pct = (count / total_samples) * 100
            print(f"  - {ds:<12}: {count:>5} samples ({pct:.2f}%)")
        dataset_breakdown = dataset_counts

    eda_summary = {
        "total_samples": total_samples,
        "num_classes": num_classes,
        "class_distribution": class_counts,
        "dataset_breakdown": dataset_breakdown,
        "is_perfectly_balanced": len(set(class_counts.values())) == 1
    }

    print(f"Perfectly balanced dataset? {'YES' if eda_summary['is_perfectly_balanced'] else 'NO'}")
    print("=" * 55 + "\n")
    return eda_summary
