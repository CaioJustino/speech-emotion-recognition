#!/bin/bash
#SBATCH --job-name=speech-emotion-recognition
#SBATCH --partition=gpu-8-v100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-0:0

# Inicializando o ambiente Conda.
eval "$(conda shell.bash hook)"
conda activate speech-emotion-recognition

# Acessando o diretório do Slurm.
cd $SLURM_SUBMIT_DIR

# Testando a conexão com a GPU.
echo "Placa de vídeo alocada:"
nvidia-smi

# Executando o script Python gerado.
echo "Iniciando o treinamento na GPU..."
python -u speech-emotion-recognition.py