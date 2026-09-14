#!/bin/bash
#SBATCH --job-name=speech-emotion-recognition
#SBATCH --partition=gpu-8-v100
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=3-0:0

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
echo "Placas de vídeo alocadas:"
nvidia-smi

# Executando o script Python gerado.
echo "Iniciando o treinamento na GPU..."
python -u main.py