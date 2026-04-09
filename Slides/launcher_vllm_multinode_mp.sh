#!/bin/bash -l
#SBATCH -A lxp
#SBATCH -q default
#SBATCH -p gpu
#SBATCH -t 48:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=4
#SBATCH --error="vllm-mp-%j.err"
#SBATCH --output="vllm-mp-%j.out"


module load Apptainer

# Fix pmix error (munge)
export PMIX_MCA_psec=native
# Choose a directory for the cache 
export LOCAL_HF_CACHE="/project/home/lxp/ekieffer/HF_cache"
mkdir -p ${LOCAL_HF_CACHE}
export HF_TOKEN="<TOKEN>"
# Make sure the path to the SIF image is correct
# Here, the SIF image is in the same directory as this script

# --> Here I should take the latest vllm with ray inside 
export SIF_IMAGE="vllm-ray.sif"
export APPTAINER_ARGS=" --nvccli -B /project/home/lxp/ekieffer/.cache:/root/.cache -B /project/home/lxp/ekieffer/HF_cache:/root/.cache/huggingface --env HF_HOME=/root/.cache/huggingface --env HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} --env XDG_CACHE_HOME=/root/.cache --env TORCHINDUCTOR_CACHE_DIR=/root/.cache --env FLASHINFER_WORKSPACE_BASE=/root/.cache"  
# Make sure your have been granted access to the model
#export HF_MODEL="mistralai/Mixtral-8x22B-Instruct-v0.1"
export HF_MODEL="Qwen/Qwen3.5-397B-A17B-GPTQ-Int4"

export HEAD_HOSTNAME="$(hostname)"
export HEAD_IPADDRESS="$(hostname --ip-address)"

echo "HEAD NODE: ${HEAD_HOSTNAME}"
echo "IP ADDRESS: ${HEAD_IPADDRESS}"
echo "SSH TUNNEL (Execute on your local machine): ssh -p 8822 ${USER}@login.lxp.lu  -NL 8000:${HEAD_IPADDRESS}:8000"  

# We need to get an available random port
export RANDOM_PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

export TENSOR_PARALLEL_SIZE=4 # Set it to the number of GPU per node
export PIPELINE_PARALLEL_SIZE=${SLURM_NNODES} # Set it to the number of allocated GPU nodes 

export CMD_BASE="vllm serve ${HF_MODEL} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} --nnodes ${SLURM_NNODES} --master-addr ${HEAD_IPADDRESS} "

export MODEL_PARAMS="--max-model-len 262144 --reasoning-parser qwen3 --quantization moe_wna16"

# Command to start the head node
export CMD_HEAD="${CMD_BASE} ${MODEL_PARAMS} "
# Command to start workers
export CMD_WORKER="${CMD_BASE} ${MODEL_PARAMS} --headless"

echo "Starting vLLM inference with multiprocessing distributed backend"
srun  -l -N ${SLURM_NNODES} --ntasks-per-node ${SLURM_NTASKS_PER_NODE}  -c ${SLURM_CPUS_PER_TASK}  ./run_cluster_mp.sh 
