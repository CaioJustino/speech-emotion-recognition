"""ArtifactStore module: Manages metric reports, CSVs, and consolidated evaluation results."""

import os
from typing import List
import pandas as pd
from src.config import (
    RESULTS_DIR,
    RESULTS_CNN_DIR,
    RESULTS_CRNN_DIR,
    CONSOLIDATED_RESULTS_PATH
)


class ArtifactStore:
    """Artifact Store abstraction for persisting evaluation CSVs and results."""

    def __init__(self, results_dir: str = RESULTS_DIR):
        self.results_dir = results_dir
        self.cnn_dir = RESULTS_CNN_DIR
        self.crnn_dir = RESULTS_CRNN_DIR
        self.consolidated_path = CONSOLIDATED_RESULTS_PATH
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Create necessary result artifact directories."""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.cnn_dir, exist_ok=True)
        os.makedirs(self.crnn_dir, exist_ok=True)

    def save_model_report(self, df_report: pd.DataFrame, model_type: str, n_mfcc: int) -> str:
        """Save classification report DataFrame for a specific model type and n_mfcc.
        
        Args:
            df_report: DataFrame formatted with 'Model', 'Class/Metric', precision, recall, f1-score, support.
            model_type: 'CNN' or 'CRNN'.
            n_mfcc: Number of MFCC coefficients.
            
        Returns:
            str: Path to saved CSV file.
        """
        model_type_upper = model_type.upper()
        if "CRNN" in model_type_upper:
            target_dir = self.crnn_dir
        else:
            target_dir = self.cnn_dir

        target_path = os.path.join(target_dir, f"{n_mfcc}.csv")
        df_report.to_csv(target_path, index=False)
        print(f"[Artifact Store] Relatório salvo: {target_path}")
        return target_path

    def save_consolidated_results(self, all_results: List[pd.DataFrame]) -> str:
        """Concatenate and save all model results into results/results_consolidated.csv."""
        if not all_results:
            raise ValueError("Nenhum resultado para consolidar.")
        
        final_results = pd.concat(all_results, ignore_index=True)
        final_results.to_csv(self.consolidated_path, index=False)
        print(f"\n[Artifact Store] Resultados consolidados salvos com sucesso em: {self.consolidated_path}")
        return self.consolidated_path

