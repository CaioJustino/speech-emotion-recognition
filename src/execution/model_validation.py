"""
    Module model_validation: Evaluates fold predictions, computes classification reports, and formats results.
"""

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
    """
    Function evaluate_fold: Generate predictions for a fold, print classification report, and return predictions and targets.
    
    Params:
        - model: The model to evaluate.
        - X_test: The test data.
        - y_test: The test labels.
        - class_names: The names of the emotion classes.
        - model_name: The name of the model.
        - fold: The fold number.
    
    Returns:
        - A tuple containing the true targets and the predicted labels.
    """
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(f"\nPartial Results: {model_name} (Fold {fold})")
    print(classification_report(y_test, y_pred, target_names=class_names))
    return y_test, y_pred


def generate_consolidated_report(
    true_labels: List[int],
    pred_labels: List[int],
    class_names: Sequence[str],
    model_label: str,
    energy_kwh: float = 0.0,
    co2e_lbs: float = 0.0
) -> pd.DataFrame:
    """
    Function generate_consolidated_report: Generate consolidated 10-fold classification report DataFrame matching project standard format.
    
    Params:
        - true_labels: Aggregated ground truth labels across all folds.
        - pred_labels: Aggregated predicted labels across all folds.
        - class_names: Names of emotion classes (from LabelEncoder).
        - model_label: String label for the 'Model' column, e.g. 'MFCC CNN (128)'.
        - energy_kwh: Total energy consumption in kWh during training across all folds.
        - co2e_lbs: Total CO2e emissions in lbs during training across all folds.
    
    Returns:
        - Formatted DataFrame ready to be stored in ArtifactStore.
    """
    print(f"\nCONSOLIDATED: {model_label}")
    print(classification_report(true_labels, pred_labels, target_names=class_names))

    report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)
    df_res = pd.DataFrame(report).transpose().reset_index().rename(columns={'index': 'Class/Metric'})
    df_res.insert(0, 'Model', model_label)
    
    if energy_kwh > 0 or co2e_lbs > 0:
        df_res['Energy_kWh'] = energy_kwh
        df_res['CO2e_lbs'] = co2e_lbs
        
    return df_res
