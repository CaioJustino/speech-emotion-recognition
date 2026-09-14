"""
    Module pipeline: Coordinates Data Engineering, Stores, and Model Engineering.
"""

import gc
import os
import warnings
from typing import List, Optional
import tensorflow as tf

from src import (
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
    """
    Function run_pipeline: Execute the complete Speech Emotion Recognition MLOps Pipeline.
    
    Params:
        - mfcc_list: Optional list of MFCC counts to process.
        - epochs: Number of training epochs.
        - batch_size: Batch size for training.
        - target_samples_per_class: Number of samples to target per class during undersampling.
        - drop_emotions: Optional list of emotions to remove from the dataset.
        - random_state: Random seed for reproducibility.
    
    Returns:
        - None
    """
    if mfcc_list is None:
        mfcc_list = MFCC_LIST
    if drop_emotions is None:
        drop_emotions = DROPPED_EMOTIONS

    # Detect if running as a task in a SLURM job array (see execute_array.sh).
    # In this mode, each task processes only a subset of mfcc_list, so it MUST NOT
    # overwrite the shared global artifacts (results_consolidated.csv,
    # run_metadata.json) — this is done later by src/aggregate_results.py, reading the
    # files per n_mfcc that each task already saves individually (without conflict).
    array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    is_array_task = array_task_id is not None

    print("\n" + "=" * 65)
    print(" STARTING SPEECH EMOTION RECOGNITION PIPELINE (MLOps)")
    if is_array_task:
        print(f" [Job Array] Task {array_task_id} — mfcc_list processed in this task: {mfcc_list}")
    print("=" * 65)

    # -------------------------------------------------------------
    # 0. Initialization of Stores and Hardware
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
        "random_state": random_state,
        "slurm_array_task_id": array_task_id
    })

    strategy = setup_training_environment()

    # -------------------------------------------------------------
    # 1. Data Engineering Pipeline (ETL)
    # -------------------------------------------------------------
    # 1.1 Extraction
    df_raw = extract_all_datasets()
    
    # 1.2 Transformation (Cleaning, Removal of 'surprise' and 'calm', Undersampling to 1703)
    df_balanced = transform_and_undersample(
        df=df_raw,
        target_samples_per_class=target_samples_per_class,
        drop_emotions=drop_emotions,
        random_state=random_state
    )

    # 1.3 Loading into Feat Store
    load_to_feat_store(df_balanced)
    feat_store.save_metadata(df_balanced)

    # -------------------------------------------------------------
    # 2. Model Engineering Pipeline: EDA and Data Validation
    # -------------------------------------------------------------
    # 2.1 EDA
    eda_stats = run_eda(df_balanced)
    metadata_store.log_dataset_stats(eda_stats)

    # 2.2 Data Validation
    validation_ok = validate_data(
        df=df_balanced,
        target_samples_per_class=target_samples_per_class,
        forbidden_emotions=drop_emotions,
        verify_files_exist=True
    )
    metadata_store.log_validation("data_integrity_check", validation_ok)

    # -------------------------------------------------------------
    # 3. Data Preparation and Audio Loading
    # -------------------------------------------------------------
    prep = DataPreparation(batch_size=batch_size, random_state=random_state)
    y_encoded = prep.fit_transform_labels(df_balanced['Emotion'])
    num_classes = prep.num_classes
    class_names = prep.classes_

    # Load audios into RAM (using FeatStore)
    feat_store.load_audios_to_ram(df_balanced['Path'].tolist())

    all_results_df = []

    # -------------------------------------------------------------
    # 4. Model Training and Validation (10-Fold for each n_mfcc)
    # -------------------------------------------------------------
    for n_mfcc in mfcc_list:
        print("\n" + "=" * 65)
        print(f" Executing extraction and 20-Fold validation for n_mfcc = {n_mfcc}")
        print("=" * 65)

        # IMPORTANT: clear_session() here, ONCE per n_mfcc (12x total) — not for each
        # fold (which would be 240x). The MirroredStrategy is created a single time at the beginning of the
        # pipeline and reused by all models; calling clear_session() too
        # frequently while repeatedly rebuilding models under the same `strategy.scope()`
        # is a combination known to corrupt/leak internal TF resources (collective
        # communicators between GPUs in the case of MirroredStrategy) — it was the cause of the error
        # "pthread_create() failed" accumulating throughout the training.
        tf.keras.backend.clear_session()
        gc.collect()

        X_mfcc = feat_store.extract_mfccs(n_mfcc)
        y_mfcc = y_encoded

        cnn_true_labels, cnn_pred_labels = [], []
        crnn_true_labels, crnn_pred_labels = [], []

        from src.execution.energy_tracker import EnergyTracker
        total_energy_cnn_kwh = 0.0
        total_co2e_cnn_lbs = 0.0
        total_energy_crnn_kwh = 0.0
        total_co2e_crnn_lbs = 0.0

        for fold, (train_index, test_index) in enumerate(prep.get_splits(X_mfcc, y_mfcc), 1):
            print(f"\n--- Fold {fold}/20 (n_mfcc={n_mfcc}) ---")

            train_X, train_y, X_test, y_test = prep.prepare_fold_tensors(
                X_mfcc=X_mfcc,
                y_mfcc=y_mfcc,
                train_index=train_index,
                test_index=test_index,
                n_mfcc=n_mfcc
            )

            # ---------------- 4.1 MFCC CNN Training ----------------
            print(f"\nTraining MFCC CNN - Fold {fold}...")
            with strategy.scope():
                model_cnn = build_cnn_model(n_mfcc=n_mfcc, num_classes=num_classes)

            tracker_cnn = EnergyTracker()
            tracker_cnn.start()

            model_cnn.fit(
                x=train_X,
                y=train_y,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                callbacks=get_callbacks(),
                verbose=2
            )

            pt_cnn, co2_cnn = tracker_cnn.stop()
            total_energy_cnn_kwh += pt_cnn
            total_co2e_cnn_lbs += co2_cnn

            y_t, y_p = evaluate_fold(model_cnn, X_test, y_test, class_names, "CNN", fold)
            cnn_true_labels.extend(y_t)
            cnn_pred_labels.extend(y_p)

            if fold == 20:
                model_registry.save_model(model_cnn, "cnn", n_mfcc)

            del model_cnn
            gc.collect()

            # ---------------- 4.2 MFCC CRNN Training ----------------
            print(f"\nTraining MFCC CRNN - Fold {fold}...")
            with strategy.scope():
                model_crnn = build_crnn_model(n_mfcc=n_mfcc, num_classes=num_classes)

            tracker_crnn = EnergyTracker()
            tracker_crnn.start()

            model_crnn.fit(
                x=train_X,
                y=train_y,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                callbacks=get_callbacks(),
                verbose=2
            )

            pt_crnn, co2_crnn = tracker_crnn.stop()
            total_energy_crnn_kwh += pt_crnn
            total_co2e_crnn_lbs += co2_crnn

            y_t, y_p = evaluate_fold(model_crnn, X_test, y_test, class_names, "CRNN", fold)
            crnn_true_labels.extend(y_t)
            crnn_pred_labels.extend(y_p)

            if fold == 20:
                model_registry.save_model(model_crnn, "crnn", n_mfcc)

            del model_crnn, train_X, train_y, X_test, y_test
            gc.collect()

        # -------------------------------------------------------------
        # 5. Consolidation of Results by n_mfcc
        # -------------------------------------------------------------
        print("\n" + "=" * 65)
        print(f" CONSOLIDATED RESULTS (20-Fold) FOR N_MFCC = {n_mfcc}")
        print("=" * 65)

        # CNN
        df_res_cnn = generate_consolidated_report(
            cnn_true_labels, cnn_pred_labels, class_names, f"MFCC CNN ({n_mfcc})",
            energy_kwh=total_energy_cnn_kwh, co2e_lbs=total_co2e_cnn_lbs
        )
        artifact_store.save_model_report(df_res_cnn, "CNN", n_mfcc)
        all_results_df.append(df_res_cnn)

        # CRNN
        df_res_crnn = generate_consolidated_report(
            crnn_true_labels, crnn_pred_labels, class_names, f"MFCC CRNN ({n_mfcc})",
            energy_kwh=total_energy_crnn_kwh, co2e_lbs=total_co2e_crnn_lbs
        )
        artifact_store.save_model_report(df_res_crnn, "CRNN", n_mfcc)
        all_results_df.append(df_res_crnn)

        del X_mfcc
        gc.collect()

    # -------------------------------------------------------------
    # 6. Finalization and Global Consolidation
    # -------------------------------------------------------------
    if is_array_task:
        # Each array task only processed a slice of mfcc_list — DO NOT overwrite the
        # global consolidated CSV or the shared run_metadata.json. Final consolidation
        # must be done by running src/aggregate_results.py after all tasks finish.
        metadata_filename = f"run_metadata_task{array_task_id}.json"
        print(f"\n[Job Array] Skipping global consolidation in this task "
              f"(run src/aggregate_results.py after all array tasks finish).")
    else:
        artifact_store.save_consolidated_results(all_results_df)
        metadata_filename = "run_metadata.json"

    metadata_store.set_status("COMPLETED")
    metadata_store.save(metadata_filename)

    print("\nMetrics successfully saved to individual folders.")
    if not is_array_task:
        print("Consolidated in results/results_consolidated.csv")
    print("\nPipeline Execution Complete!")
