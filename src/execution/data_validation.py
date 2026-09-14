"""
    Module data_validation: Verifies data integrity, class balance, and audio file existence.
"""

import os
from typing import Optional, List
import pandas as pd
from src import TARGET_SAMPLES_PER_CLASS, DROPPED_EMOTIONS


class DataValidationError(Exception):
    """
        Class DataValidationError: Exception raised when dataset fails validation checks.
    """
    pass


def validate_data(
    df: pd.DataFrame,
    target_samples_per_class: int = TARGET_SAMPLES_PER_CLASS,
    forbidden_emotions: Optional[List[str]] = None,
    verify_files_exist: bool = True
) -> bool:
    """
        Function validate_data: Validate dataset integrity according to SER pipeline specifications.
        
        Params:
            - df: DataFrame to validate.
            - target_samples_per_class: Expected number of samples per emotion class.
            - forbidden_emotions: List of emotions that must not be in dataset (e.g. ['calm', 'surprise']).
            - verify_files_exist: If True, physically checks that every file in 'Path' exists on disk.
            
        Returns:
            - bool: True if all checks pass.
    """
    if forbidden_emotions is None:
        forbidden_emotions = DROPPED_EMOTIONS

    print("\n" + "=" * 55)
    print(" [Data Validation] Starting quality checks...")
    print("=" * 55)

    # 1. Check for empty DataFrame
    if df is None or len(df) == 0:
        raise DataValidationError("Validation failed: DataFrame is empty.")

    # 2. Check essential columns
    required_cols = ['Emotion', 'Path']
    for col in required_cols:
        if col not in df.columns:
            raise DataValidationError(f"Validation failed: Required column '{col}' missing from DataFrame.")

    # 3. Check for null or NaN values
    null_counts = df[required_cols].isnull().sum().to_dict()
    for col, count in null_counts.items():
        if count > 0:
            raise DataValidationError(f"Validation failed: {count} null values found in column '{col}'.")

    # 4. Check for absence of forbidden emotions (e.g., calm, surprise)
    found_forbidden = set(df['Emotion']).intersection(set(forbidden_emotions))
    if found_forbidden:
        raise DataValidationError(
            f"Validation failed: Emotions that should be removed are still present: {found_forbidden}"
        )

    # 5. Check if all classes have exactly the target number of samples
    counts = df['Emotion'].value_counts()
    for emotion, count in counts.items():
        if count != target_samples_per_class:
            raise DataValidationError(
                f"Validation failed: Class '{emotion}' has {count} samples (expected: {target_samples_per_class})."
            )

    # 6. Check physical integrity of audio file paths
    if verify_files_exist:
        missing_files = []
        for path in df['Path']:
            if not os.path.isfile(path):
                missing_files.append(path)
                if len(missing_files) >= 5:  # Limit to avoid overloading
                    break
        if missing_files:
            raise DataValidationError(
                f"Validation failed: Audio files not found on disk: {missing_files[:3]}..."
            )

    print(f"  [OK] Required columns present ({required_cols})")
    print(f"  [OK] No null or missing values")
    print(f"  [OK] Dropped emotions {forbidden_emotions} absent")
    print(f"  [OK] All {len(counts)} classes have exactly {target_samples_per_class} samples")
    print(f"  [OK] Total samples validated: {len(df)} (6 x {target_samples_per_class})")
    if verify_files_exist:
        print(f"  [OK] 100% physical integrity of audio paths confirmed on disk")
    
    print("[Data Validation] All checks completed SUCCESSFULLY!")
    print("=" * 55 + "\n")
    return True
