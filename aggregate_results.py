#!/usr/bin/env python3
"""Aggregate Results: Consolida os resultados de todas as tarefas do job array
(execute_array.sh) em um único results_consolidated.csv e um único metadata
resumido — rodar SOMENTE depois que todas as tarefas do array tiverem terminado.

Uso:
    python -u aggregate_results.py
"""

import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    RESULTS_CNN_DIR,
    RESULTS_CRNN_DIR,
    CONSOLIDATED_RESULTS_PATH,
    METADATA_DIR,
    MFCC_LIST,
)


def aggregate_model_reports() -> pd.DataFrame:
    """Lê todos os {n_mfcc}.csv de MFCC_CNN/ e MFCC_CRNN/ e concatena em um único DataFrame.

    Cada arquivo já é o relatório consolidado dos 10 folds daquele n_mfcc/modelo
    (gerado por ArtifactStore.save_model_report), então não há necessidade de
    reagregar folds aqui — só juntar os arquivos, um por n_mfcc/modelo.
    """
    all_dfs = []
    missing = []

    for results_dir, model_label in [(RESULTS_CNN_DIR, "CNN"), (RESULTS_CRNN_DIR, "CRNN")]:
        for n_mfcc in MFCC_LIST:
            csv_path = os.path.join(results_dir, f"{n_mfcc}.csv")
            if os.path.exists(csv_path):
                all_dfs.append(pd.read_csv(csv_path))
            else:
                missing.append(csv_path)

    if missing:
        print("\n[Aviso] Os seguintes arquivos esperados não foram encontrados "
              "(a tarefa correspondente do array pode não ter terminado ainda "
              "ou ter falhado):")
        for path in missing:
            print(f"  - {path}")

    if not all_dfs:
        raise FileNotFoundError(
            "Nenhum resultado encontrado em MFCC_CNN/ ou MFCC_CRNN/. "
            "Verifique se as tarefas do array já rodaram e geraram resultados."
        )

    return pd.concat(all_dfs, ignore_index=True)


def aggregate_metadata() -> dict:
    """Junta os run_metadata_task{N}.json de cada tarefa num único resumo."""
    pattern = os.path.join(METADATA_DIR, "run_metadata_task*.json")
    task_files = sorted(glob.glob(pattern))

    summary = {
        "num_tasks_found": len(task_files),
        "tasks": {}
    }

    for path in task_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        task_id = os.path.basename(path).replace("run_metadata_task", "").replace(".json", "")
        summary["tasks"][task_id] = {
            "pipeline_status": data.get("pipeline_status"),
            "mfcc_list": data.get("parameters", {}).get("mfcc_list"),
            "saved_at": data.get("saved_at"),
        }

    return summary


def main():
    print("=" * 65)
    print(" Agregando resultados de todas as tarefas do job array")
    print("=" * 65)

    final_results = aggregate_model_reports()
    final_results.to_csv(CONSOLIDATED_RESULTS_PATH, index=False)
    print(f"\n[OK] Resultados consolidados salvos em: {CONSOLIDATED_RESULTS_PATH}")
    print(f"     Total de linhas: {len(final_results)}")

    metadata_summary = aggregate_metadata()
    summary_path = os.path.join(METADATA_DIR, "run_metadata_array_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metadata_summary, f, indent=4, ensure_ascii=False)
    print(f"[OK] Resumo de metadados das tarefas salvo em: {summary_path}")

    expected_tasks = 12
    if metadata_summary["num_tasks_found"] < expected_tasks:
        print(f"\n[Aviso] Encontrados metadados de apenas "
              f"{metadata_summary['num_tasks_found']}/{expected_tasks} tarefas. "
              f"Confirme que todas as tarefas do array terminaram com sucesso "
              f"antes de considerar essa consolidação completa.")

    print("\nAgregação concluída!")


if __name__ == "__main__":
    main()
