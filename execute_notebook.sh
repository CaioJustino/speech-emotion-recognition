#!/bin/bash
#SBATCH --job-name=speech-emotion-recognition-notebook
#SBATCH --output=logs/results/result_%j.txt
#SBATCH --error=logs/errors/error_%j.txt     
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=8              
#SBATCH --mem=32G                      
#SBATCH --time=24:00:00                
#SBATCH --partition=amd-512

mkdir -p logs/results
mkdir -p logs/errors

cd $SLURM_SUBMIT_DIR

# Inicializando o ambiente Conda.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate speech-emotion-recognition

# Convertendo o Notebook para script Python.
echo "Convertendo o notebook para script Python..."
jupyter nbconvert --to script speech_emotion_recogniton.ipynb

# Executando o script Python gerado.
echo "Iniciando o treinamento..."
python speech_emotion_recogniton.py