"""Loading module: Persists cleaned/transformed metadata into the Feature Store."""

import os
import pandas as pd
from src.config import FEAT_STORE_DIR


def load_to_feat_store(df_balanced: pd.DataFrame, feat_store_dir: str = FEAT_STORE_DIR) -> str:
    """Save balanced dataset metadata into the Feature Store directory.
    
    Args:
        df_balanced: Transformed and balanced DataFrame.
        feat_store_dir: Path to the feature store directory.
        
    Returns:
        str: Filepath to the saved metadata file in the Feature Store.
    """
    os.makedirs(feat_store_dir, exist_ok=True)
    target_path = os.path.join(feat_store_dir, "metadata_balanced.csv")
    df_balanced.to_csv(target_path, index=False)
    print(f"\n[ETL - Carregamento] Dados transformados persistidos no Feat Store: {target_path}")
    return target_path

