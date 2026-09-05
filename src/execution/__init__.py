"""Model Engineering package: EDA, data validation, data preparation, training, and validation."""

from .eda import run_eda
from .data_validation import validate_data
from .data_preparation import DataPreparation
from .model_training import (
    setup_training_environment,
    build_cnn_model,
    build_crnn_model,
    get_callbacks
)
from .model_validation import (
    evaluate_fold,
    generate_consolidated_report
)

__all__ = [
    "run_eda",
    "validate_data",
    "DataPreparation",
    "setup_training_environment",
    "build_cnn_model",
    "build_crnn_model",
    "get_callbacks",
    "evaluate_fold",
    "generate_consolidated_report",
]

