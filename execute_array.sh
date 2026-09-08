#!/bin/bash
#SBATCH --job-name=speech-emotion-recognition
#SBATCH --partition=gpu-8-v100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-6:0
#SBATCH --array=0-11%4
#SBATCH --output=logs/ser_%A_%a.out
#SBATCH --error=logs/ser_%A_%a.err

# ---------------------------------------------------------------------------
# Job array: cada tarefa ($SLURM_ARRAY_TASK_ID de 0 a 11) processa UM valor de
# n_mfcc, com 1 GPU dedicada. O "%4" no --array limita a 4 tarefas rodando ao
# mesmo tempo (uma por GPU V100 disponível simultaneamente no cluster).
#
# Ajustes em relação ao execute.sh original (que rodava tudo sequencialmente
# em 1 job com 4 GPUs via MirroredStrategy):
#   --gpus-per-node=4 -> 1   (cada tarefa usa só 1 GPU; sem overhead de
#                             sincronização multi-GPU para um modelo pequeno)
#   --cpus-per-task=32 -> 8  (cada tarefa processa 1/12 do trabalho de CPU;
#                             8 cpus é suficiente para a extração de MFCC de
#                             uma fatia do dataset. Ajuste se notar gargalo.)
#   --mem=128G -> 32G        (idem: cada tarefa carrega o dataset inteiro em
#                             RAM uma vez, não 4x simultâneas; 32G é margem
#                             segura, ajuste conforme necessário)
#   --time=3-0:0 -> 0-6:0    (cada tarefa treina só 1 n_mfcc: 10 folds x 2
#                             modelos, não os 12 valores inteiros. Ajuste
#                             conforme o tempo real observado por n_mfcc.)
# ---------------------------------------------------------------------------

mkdir -p logs

# Lista de n_mfcc na MESMA ordem de src/config.py:MFCC_LIST — se mudar lá,
# mude aqui também (índice $SLURM_ARRAY_TASK_ID -> valor de n_mfcc).
MFCC_LIST=(12 13 14 25 39 65 96 128 192 255 256 257)
N_MFCC=${MFCC_LIST[$SLURM_ARRAY_TASK_ID]}

echo "=== Tarefa do array: $SLURM_ARRAY_TASK_ID -> n_mfcc=$N_MFCC ==="

# Inicializando o ambiente Conda.
eval "$(conda shell.bash hook)"
conda activate speech-emotion-recognition

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)" 2>/dev/null) 2>/dev/null)
CUBLAS_PATH=$(dirname $(python -c "import nvidia.cublas;print(nvidia.cublas.__file__)" 2>/dev/null) 2>/dev/null)
NVVM_PATH=$(dirname $(python -c "import nvidia.cuda_nvrtc;print(nvidia.cuda_nvrtc.__file__)" 2>/dev/null) 2>/dev/null)

if [ ! -z "$CUDNN_PATH" ]; then export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$LD_LIBRARY_PATH; fi
if [ ! -z "$CUBLAS_PATH" ]; then export LD_LIBRARY_PATH=$CUBLAS_PATH/lib:$LD_LIBRARY_PATH; fi
if [ ! -z "$NVVM_PATH" ]; then export LD_LIBRARY_PATH=$NVVM_PATH/lib:$LD_LIBRARY_PATH; fi

# Acessando o diretório do Slurm.
cd $SLURM_SUBMIT_DIR

# Testando a conexão com a GPU.
echo "Placa de vídeo alocada para esta tarefa (n_mfcc=$N_MFCC):"
nvidia-smi

# Executando o script Python, restringindo a este único valor de n_mfcc.
echo "Iniciando o treinamento na GPU para n_mfcc=$N_MFCC..."
python -u main.py --mfcc-list $N_MFCC
