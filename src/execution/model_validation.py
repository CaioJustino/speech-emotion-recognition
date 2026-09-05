"""Model Validation module: Evaluates fold predictions, computes classification reports, and formats results."""

from typing import List, Tuple, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def evaluate_fold(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
    model_name: str,
    fold: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions for a fold, print classification report, and return predictions and targets."""
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(f"\nResultados Parciais: {model_name} (Fold {fold})")
    print(classification_report(y_test, y_pred, target_names=class_names))
    return y_test, y_pred


def generate_consolidated_report(
    true_labels: List[int],
    pred_labels: List[int],
    class_names: Sequence[str],
    model_label: str
) -> pd.DataFrame:
    """Generate consolidated 10-fold classification report DataFrame matching project standard format.
    
    Args:
        true_labels: Aggregated ground truth labels across all folds.
        pred_labels: Aggregated predicted labels across all folds.
        class_names: Names of emotion classes (from LabelEncoder).
        model_label: String label for the 'Model' column, e.g. 'MFCC CNN (128)'.
        
    Returns:
        pd.DataFrame: Formatted DataFrame ready to be stored in ArtifactStore.
    """
    print(f"\nCONSOLIDADO: {model_label}")
    print(classification_report(true_labels, pred_labels, target_names=class_names))

    report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)
    df_res = pd.DataFrame(report).transpose().reset_index().rename(columns={'index': 'Class/Metric'})
    df_res.insert(0, 'Model', model_label)
    return df_res

