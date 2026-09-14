# Speech Emotion Recognition (SER) Pipeline

## Overview
This repository contains a complete Machine Learning Operations (MLOps) pipeline for Speech Emotion Recognition (SER). The project evaluates Convolutional Neural Networks (CNN) and Convolutional Recurrent Neural Networks (CRNN) architectures trained on Mel Frequency Cepstral Coefficients (MFCCs). 

The pipeline encapsulates the entire machine learning lifecycle, including data extraction, transformation, undersampling, model training, rigorous 20-fold cross-validation, and automated evaluation reporting.

## Key Features
- **Modular MLOps Architecture**: Distinct modules for ETL (Extraction, Transformation, Loading), Execution (EDA, Training, Validation), and Stores (Feature, Metadata, Artifact, Model registries).
- **Automated Data Processing**: Cleans labels, removes unwanted emotions (e.g., calm, surprise), and ensures balanced classes via targeted random undersampling.
- **Hardware Optimization**: Implements TensorFlow mixed precision and XLA compilation to maximize GPU throughput and training efficiency.
- **Energy Consumption Tracking**: Features integrated real-time monitoring of energy consumption (kWh) and estimated carbon emissions (CO2e) during model training, following the methodology proposed by Strubell et al. (2019).
- **Robust Cross-Validation**: Reliable performance evaluation across all implemented architectures using 20-fold cross-validation.

## Project Structure
- `config.yaml`: Centralized configuration file defining local paths, hardware settings, and model hyperparameters.
- `main.py`: The primary Command Line Interface (CLI) entrypoint for triggering the pipeline.
- `src/`: Contains the core source code divided into:
  - `etl/`: Data engineering workflows.
  - `execution/`: Model engineering, training, tracking, and evaluation loops.
  - `stores/`: Management of datasets, models, metrics, and execution artifacts.
  - `aggregate_results.py`: Consolidation script for aggregating outputs from distributed job array executions.

## Usage
The pipeline is designed to be executed via the main entry point. Default hyperparameters and paths are automatically read from `config.yaml`.

To start the standard training pipeline:
```bash
python main.py
```

To run the pipeline with customized arguments (e.g., evaluating specific MFCC values or altering the number of epochs):
```bash
python main.py --mfcc-list 128 256 --epochs 50
```

For high-performance computing (HPC) environments using Slurm, you can utilize the provided shell scripts:
- `execute.sh`: Standard sequential execution.
- `execute_array.sh`: Distributed execution across multiple nodes using Slurm job arrays.

## Results and Artifacts
Upon completion, the pipeline automatically generates and stores consolidated classification reports (Accuracy, Precision, Recall, F1-Score) along with the recorded energy consumption and CO2e emissions metrics. All generated artifacts are saved directly into the `results/` directory.

