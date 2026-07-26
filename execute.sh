#!/bin/bash
#SBATCH --job-name=speech-emotion-recognition
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=8              
#SBATCH --mem=32G                      
#SBATCH --time=24:00:00                
#SBATCH --partition=amd-512

# Inicializando o ambiente Conda.
eval "$(conda shell.bash hook)"
conda activate speech-emotion-recognition

# Acessando o diretório do Slurm.
cd $SLURM_SUBMIT_DIR

# Executando o script Python gerado.
echo "Iniciando o treinamento..."
python speech-emotion-recognition.py