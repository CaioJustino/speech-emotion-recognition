"""Data Validation module: Verifies data integrity, class balance, and audio file existence."""

import os
from typing import Optional, List
import pandas as pd
from src.config import TARGET_SAMPLES_PER_CLASS, DROPPED_EMOTIONS


class DataValidationError(Exception):
    """Exception raised when dataset fails validation checks."""
    pass


def validate_data(
    df: pd.DataFrame,
    target_samples_per_class: int = TARGET_SAMPLES_PER_CLASS,
    forbidden_emotions: Optional[List[str]] = None,
    verify_files_exist: bool = True
) -> bool:
    """Validate dataset integrity according to SER pipeline specifications.
    
    Args:
        df: DataFrame to validate.
        target_samples_per_class: Expected number of samples per emotion class.
        forbidden_emotions: List of emotions that must not be in dataset (e.g. ['calm', 'surprise']).
        verify_files_exist: If True, physically checks that every file in 'Path' exists on disk.
        
    Returns:
        bool: True if all checks pass.
        
    Raises:
        DataValidationError: If any check fails.
    """
    if forbidden_emotions is None:
        forbidden_emotions = DROPPED_EMOTIONS

    print("\n" + "=" * 55)
    print(" [Validação dos Dados] Iniciando verificações de qualidade...")
    print("=" * 55)

    # 1. Verificar DataFrame vazio
    if df is None or len(df) == 0:
        raise DataValidationError("Validação falhou: DataFrame está vazio.")

    # 2. Verificar colunas essenciais
    required_cols = ['Emotion', 'Path']
    for col in required_cols:
        if col not in df.columns:
            raise DataValidationError(f"Validação falhou: Coluna obrigatória '{col}' ausente no DataFrame.")

    # 3. Verificar valores nulos ou NaN
    null_counts = df[required_cols].isnull().sum().to_dict()
    for col, count in null_counts.items():
        if count > 0:
            raise DataValidationError(f"Validação falhou: {count} valores nulos encontrados na coluna '{col}'.")

    # 4. Verificar ausência de emoções proibidas (ex: calm, surprise)
    found_forbidden = set(df['Emotion']).intersection(set(forbidden_emotions))
    if found_forbidden:
        raise DataValidationError(
            f"Validação falhou: Emoções que deveriam ser removidas ainda estão presentes: {found_forbidden}"
        )

    # 5. Verificar se todas as classes possuem exatamente o número alvo de amostras
    counts = df['Emotion'].value_counts()
    for emotion, count in counts.items():
        if count != target_samples_per_class:
            raise DataValidationError(
                f"Validação falhou: Classe '{emotion}' possui {count} amostras (esperado: {target_samples_per_class})."
            )

    # 6. Verificar integridade física dos caminhos dos arquivos de áudio
    if verify_files_exist:
        missing_files = []
        for path in df['Path']:
            if not os.path.isfile(path):
                missing_files.append(path)
                if len(missing_files) >= 5:  # Limita para não sobrecarregar
                    break
        if missing_files:
            raise DataValidationError(
                f"Validação falhou: Arquivos de áudio não encontrados no disco: {missing_files[:3]}..."
            )

    print(f"  [OK] Colunas requeridas presentes ({required_cols})")
    print(f"  [OK] Sem valores nulos ou ausentes")
    print(f"  [OK] Emoções descartadas {forbidden_emotions} ausentes")
    print(f"  [OK] Todas as {len(counts)} classes possuem exatamente {target_samples_per_class} amostras")
    print(f"  [OK] Total de amostras validado: {len(df)} (6 x {target_samples_per_class})")
    if verify_files_exist:
        print(f"  [OK] Integridade física de 100% dos caminhos de áudio confirmada no disco")
    
    print("[Validação dos Dados] Todas as verificações foram concluídas com SUCESSO!")
    print("=" * 55 + "\n")
    return True

