"""MetadataStore module: Persists run configurations, dataset statistics, and execution tracking."""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from src.config import METADATA_DIR


class MetadataStore:
    """Store for tracking and persisting pipeline execution metadata."""

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
        """Log pipeline hyperparameters and settings."""
        self.run_data["parameters"].update(params)

    def log_dataset_stats(self, stats: Dict[str, Any]) -> None:
        """Log dataset and EDA statistics."""
        self.run_data["dataset_stats"].update(stats)

    def log_validation(self, check_name: str, passed: bool, details: Optional[Dict[str, Any]] = None) -> None:
        """Log data validation step outcomes."""
        self.run_data["validation_checks"][check_name] = {
            "passed": passed,
            "details": details or {}
        }

    def log_metrics_summary(self, key: str, summary: Any) -> None:
        """Log summary metrics for a given model or n_mfcc."""
        self.run_data["metrics_summary"][key] = summary

    def set_status(self, status: str) -> None:
        """Update overall pipeline execution status."""
        self.run_data["pipeline_status"] = status
        self.run_data["updated_at"] = datetime.now().isoformat()

    def save(self, filename: str = "run_metadata.json") -> str:
        """Save current metadata to JSON file."""
        self.run_data["saved_at"] = datetime.now().isoformat()
        filepath = os.path.join(self.metadata_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.run_data, f, indent=4, ensure_ascii=False)
        print(f"[Metadata Store] Metadados da execução registrados em: {filepath}")
        return filepath

