#!/bin/bash
#SBATCH --job-name=speech-emotion-recognition
#SBATCH --output=logs/results_%j.txt
#SBATCH --error=logs/errors_%j.txt     
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=8              
#SBATCH --mem=32G                      
#SBATCH --time=24:00:00                
#SBATCH --partition=amd-512

# 1. Criando a pasta de "logs", caso ela não exista.
mkdir -p logs

# 2. Inicializando o Conda no script.
source $(conda info --base)/etc/profile.d/conda.sh

# 3. Ativando o ambiente Conda.
conda activate speech_emotion_recognition

# 4. Navegando para o diretório de trabalho onde o job foi submetido.
cd $SLURM_SUBMIT_DIR

# 5. Executando o projeto.
python main.py