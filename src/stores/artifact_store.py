"""
    Module artifact_store: Manages metric reports, CSVs, and consolidated evaluation results.
"""

import os
from typing import List
import pandas as pd
from src import (
    RESULTS_DIR,
    RESULTS_CNN_DIR,
    RESULTS_CRNN_DIR,
    CONSOLIDATED_RESULTS_PATH
)


class ArtifactStore:
    """
        Class ArtifactStore: Artifact Store abstraction for persisting evaluation CSVs and results.
    """

    def __init__(self, results_dir: str = RESULTS_DIR):
        self.results_dir = results_dir
        self.cnn_dir = RESULTS_CNN_DIR
        self.crnn_dir = RESULTS_CRNN_DIR
        self.consolidated_path = CONSOLIDATED_RESULTS_PATH
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """
            Function _ensure_dirs: Create necessary result artifact directories.
            
            Params:
                - None.
            
            Returns:
                - None.
        """
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.cnn_dir, exist_ok=True)
        os.makedirs(self.crnn_dir, exist_ok=True)

    def save_model_report(self, df_report: pd.DataFrame, model_type: str, n_mfcc: int) -> str:
        """
            Function save_model_report: Save classification report DataFrame for a specific model type and n_mfcc.
            
            Params:
                - df_report: DataFrame formatted with 'Model', 'Class/Metric', precision, recall, f1-score, support.
                - model_type: 'CNN' or 'CRNN'.
                - n_mfcc: Number of MFCC coefficients.
            
            Returns:
                - Path to saved CSV file.
        """
        model_type_upper = model_type.upper()
        if "CRNN" in model_type_upper:
            target_dir = self.crnn_dir
        else:
            target_dir = self.cnn_dir

        target_path = os.path.join(target_dir, f"{n_mfcc}.csv")
        df_report.to_csv(target_path, index=False)
        print(f"[Artifact Store] Report saved: {target_path}")
        return target_path

    def save_consolidated_results(self, all_results: List[pd.DataFrame]) -> str:
        """
            Function save_consolidated_results: Concatenate and save all model results into results/results_consolidated.csv.
            
            Params:
                - all_results: List of all model result DataFrames.
            
            Returns:
                - Path to the consolidated CSV file.
        """
        if not all_results:
            raise ValueError("No results to consolidate.")
        
        final_results = pd.concat(all_results, ignore_index=True)
        final_results.to_csv(self.consolidated_path, index=False)
        print(f"\n[Artifact Store] Consolidated results successfully saved at: {self.consolidated_path}")
        return self.consolidated_path
