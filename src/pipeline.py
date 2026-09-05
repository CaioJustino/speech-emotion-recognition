"""Pipeline Orchestrator module: Coordinates Data Engineering, Stores, and Model Engineering."""

import gc
import warnings
from typing import List, Optional
import tensorflow as tf

from src.config import (
    MFCC_LIST,
    EPOCHS,
    GLOBAL_BATCH_SIZE,
    TARGET_SAMPLES_PER_CLASS,
    DROPPED_EMOTIONS,
    RANDOM_STATE
)
from src.etl.extraction import extract_all_datasets
from src.etl.transformation import transform_and_undersample
from src.etl.loading import load_to_feat_store
from src.stores.feat_store import FeatStore
from src.stores.metadata_store import MetadataStore
from src.stores.artifact_store import ArtifactStore
from src.stores.model_registry import ModelRegistry
from src.execution.eda import run_eda
from src.execution.data_validation import validate_data
from src.execution.data_preparation import DataPreparation
from src.execution.model_training import (
    setup_training_environment,
    build_cnn_model,
    build_crnn_model,
    get_callbacks
)
from src.execution.model_validation import (
    evaluate_fold,
    generate_consolidated_report
)

warnings.filterwarnings('ignore')


def run_pipeline(
    mfcc_list: Optional[List[int]] = None,
    epochs: int = EPOCHS,
    batch_size: int = GLOBAL_BATCH_SIZE,
    target_samples_per_class: int = TARGET_SAMPLES_PER_CLASS,
    drop_emotions: Optional[List[str]] = None,
    random_state: int = RANDOM_STATE
) -> None:
    """Execute the complete Speech Emotion Recognition MLOps Pipeline."""
    if mfcc_list is None:
        mfcc_list = MFCC_LIST
    if drop_emotions is None:
        drop_emotions = DROPPED_EMOTIONS

    print("\n" + "=" * 65)
    print(" INICIANDO PIPELINE DE SPEECH EMOTION RECOGNITION (MLOps)")
    print("=" * 65)

    # -------------------------------------------------------------
    # 0. Inicialização dos Stores e Hardware
    # -------------------------------------------------------------
    metadata_store = MetadataStore()
    artifact_store = ArtifactStore()
    model_registry = ModelRegistry()
    feat_store = FeatStore()

    metadata_store.log_parameters({
        "mfcc_list": mfcc_list,
        "epochs": epochs,
        "batch_size": batch_size,
        "target_samples_per_class": target_samples_per_class,
        "drop_emotions": drop_emotions,
        "random_state": random_state
    })

    strategy = setup_training_environment()

    # -------------------------------------------------------------
    # 1. Pipeline de Engenharia de Dados (ETL)
    # -------------------------------------------------------------
    # 1.1 Extração
    df_raw = extract_all_datasets()
    
    # 1.2 Transformação (Limpeza, Remoção de 'surprise' e 'calm', Undersampling para 1703)
    df_balanced = transform_and_undersample(
        df=df_raw,
        target_samples_per_class=target_samples_per_class,
        drop_emotions=drop_emotions,
        random_state=random_state
    )

    # 1.3 Carregamento no Feat Store
    load_to_feat_store(df_balanced)
    feat_store.save_metadata(df_balanced)

    # -------------------------------------------------------------
    # 2. Pipeline de Engenharia de Modelos: EDA e Validação dos Dados
    # -------------------------------------------------------------
    # 2.1 EDA
    eda_stats = run_eda(df_balanced)
    metadata_store.log_dataset_stats(eda_stats)

    # 2.2 Validação dos Dados
    validation_ok = validate_data(
        df=df_balanced,
        target_samples_per_class=target_samples_per_class,
        forbidden_emotions=drop_emotions,
        verify_files_exist=True
    )
    metadata_store.log_validation("data_integrity_check", validation_ok)

    # -------------------------------------------------------------
    # 3. Preparação dos Dados e Carregamento de Áudios
    # -------------------------------------------------------------
    prep = DataPreparation(batch_size=batch_size, random_state=random_state)
    y_encoded = prep.fit_transform_labels(df_balanced['Emotion'])
    num_classes = prep.num_classes
    class_names = prep.classes_

    # Carregar áudios na RAM (usando FeatStore)
    feat_store.load_audios_to_ram(df_balanced['Path'].tolist())

    all_results_df = []

    # -------------------------------------------------------------
    # 4. Treinamento e Validação do Modelo (10-Fold para cada n_mfcc)
    # -------------------------------------------------------------
    for n_mfcc in mfcc_list:
        print("\n" + "=" * 65)
        print(f" Executando extração e validação 10-Fold para n_mfcc = {n_mfcc}")
        print("=" * 65)

        X_mfcc = feat_store.extract_mfccs(n_mfcc)
        y_mfcc = y_encoded

        cnn_true_labels, cnn_pred_labels = [], []
        crnn_true_labels, crnn_pred_labels = [], []

        for fold, (train_index, test_index) in enumerate(prep.get_splits(X_mfcc, y_mfcc), 1):
            tf.keras.backend.clear_session()
            gc.collect()

            print(f"\n--- Fold {fold}/10 (n_mfcc={n_mfcc}) ---")

            train_ds, test_ds, X_test, y_test = prep.prepare_fold_tensors(
                X_mfcc=X_mfcc,
                y_mfcc=y_mfcc,
                train_index=train_index,
                test_index=test_index,
                n_mfcc=n_mfcc
            )

            # ---------------- 4.1 Treinamento MFCC CNN ----------------
            print(f"\nTreinando MFCC CNN - Fold {fold}...")
            with strategy.scope():
                model_cnn = build_cnn_model(n_mfcc=n_mfcc, num_classes=num_classes)

            model_cnn.fit(
                train_ds,
                epochs=epochs,
                validation_data=test_ds,
                callbacks=get_callbacks(),
                verbose=2
            )

            y_t, y_p = evaluate_fold(model_cnn, X_test, y_test, class_names, "CNN", fold)
            cnn_true_labels.extend(y_t)
            cnn_pred_labels.extend(y_p)

            if fold == 10:
                model_registry.save_model(model_cnn, "cnn", n_mfcc)

            del model_cnn
            gc.collect()

            # ---------------- 4.2 Treinamento MFCC CRNN ----------------
            print(f"\nTreinando MFCC CRNN - Fold {fold}...")
            with strategy.scope():
                model_crnn = build_crnn_model(n_mfcc=n_mfcc, num_classes=num_classes)

            model_crnn.fit(
                train_ds,
                epochs=epochs,
                validation_data=test_ds,
                callbacks=get_callbacks(),
                verbose=2
            )

            y_t, y_p = evaluate_fold(model_crnn, X_test, y_test, class_names, "CRNN", fold)
            crnn_true_labels.extend(y_t)
            crnn_pred_labels.extend(y_p)

            if fold == 10:
                model_registry.save_model(model_crnn, "crnn", n_mfcc)

            del model_crnn, train_ds, test_ds, X_test, y_test
            gc.collect()

        # -------------------------------------------------------------
        # 5. Consolidação de Resultados por n_mfcc
        # -------------------------------------------------------------
        print("\n" + "=" * 65)
        print(f" RESULTADOS CONSOLIDADOS (10-Fold) PARA N_MFCC = {n_mfcc}")
        print("=" * 65)

        # CNN
        df_res_cnn = generate_consolidated_report(
            cnn_true_labels, cnn_pred_labels, class_names, f"MFCC CNN ({n_mfcc})"
        )
        artifact_store.save_model_report(df_res_cnn, "CNN", n_mfcc)
        all_results_df.append(df_res_cnn)

        # CRNN
        df_res_crnn = generate_consolidated_report(
            crnn_true_labels, crnn_pred_labels, class_names, f"MFCC CRNN ({n_mfcc})"
        )
        artifact_store.save_model_report(df_res_crnn, "CRNN", n_mfcc)
        all_results_df.append(df_res_crnn)

        del X_mfcc
        gc.collect()

    # -------------------------------------------------------------
    # 6. Finalização e Consolidação Global
    # -------------------------------------------------------------
    artifact_store.save_consolidated_results(all_results_df)
    metadata_store.set_status("COMPLETED")
    metadata_store.save()

    print("\nMetrics successfully saved to individual folders and consolidated in results/results_consolidated.csv")
    print("\nPipeline Execution Complete!")