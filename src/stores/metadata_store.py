"""
    Module metadata_store: Persists run configurations, dataset statistics, and execution tracking.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from src import METADATA_DIR


class MetadataStore:
    """
        Class MetadataStore: Store for tracking and persisting pipeline execution metadata.
    """

    def __init__(self, metadata_dir: str = METADATA_DIR):
        self.metadata_dir = metadata_dir
        os.makedirs(self.metadata_dir, exist_ok=True)
        self.run_data: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "pipeline_status": "INITIALIZED",
            "parameters": {},
            "dataset_stats": {},
            "validation_checks": {},
            "metrics_summary": {}
        }

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """
            Function log_parameters: Log pipeline hyperparameters and settings.
            
            Params:
                - params: Pipeline hyperparameters and settings to log.
            
            Returns:
                - None.
        """
        self.run_data["parameters"].update(params)

    def log_dataset_stats(self, stats: Dict[str, Any]) -> None:
        """
            Function log_dataset_stats: Log dataset and EDA statistics.
            
            Params:
                - stats: Dataset and EDA statistics to log.
            
            Returns:
                - None.
        """
        self.run_data["dataset_stats"].update(stats)

    def log_validation(self, check_name: str, passed: bool, details: Optional[Dict[str, Any]] = None) -> None:
        """
            Function log_validation: Log data validation step outcomes.
            
            Params:
                - check_name: The name of the validation check.
                - passed: Boolean indicating whether the validation passed.
                - details: Optional dictionary containing additional details of the validation.
            
            Returns:
                - None.
        """
        self.run_data["validation_checks"][check_name] = {
            "passed": passed,
            "details": details or {}
        }

    def log_metrics_summary(self, key: str, summary: Any) -> None:
        """
            Function log_metrics_summary: Log summary metrics for a given model or n_mfcc.
            
            Params:
                - key: The key identifying the model or n_mfcc setting.
                - summary: The summary metrics to log.
            
            Returns:
                - None.
        """
        self.run_data["metrics_summary"][key] = summary

    def set_status(self, status: str) -> None:
        """
            Function set_status: Update overall pipeline execution status.
            
            Params:
                - status: The status to set.
            
            Returns:
                - None.
        """
        self.run_data["pipeline_status"] = status
        self.run_data["updated_at"] = datetime.now().isoformat()

    def save(self, filename: str = "run_metadata.json") -> str:
        """
            Function save: Save current metadata to JSON file.
            
            Params:
                - filename: The name of the file to save the metadata to.
            
            Returns:
                - The filepath where the metadata was saved.
        """
        self.run_data["saved_at"] = datetime.now().isoformat()
        filepath = os.path.join(self.metadata_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.run_data, f, indent=4, ensure_ascii=False)
        print(f"[Metadata Store] Execution metadata recorded at: {filepath}")
        return filepath
