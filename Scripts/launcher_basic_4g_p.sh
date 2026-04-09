#!/bin/bash -l
#SBATCH --job-name=monai_basic_4g_p
#SBATCH --output=monai_basic_4g_p.out
#SBATCH --error=monai_basic_4g_p.err
#SBATCH --time=02:00:00
#SBATCH --qos=dev
#SBATCH --reservation=gpudev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --account=lxp
#SBATCH --partition=gpu

module load env/release/2025.1
module load Nsight-Systems
module load Python
NUM_GPUS=4
timestamp=$(date +"%Y-%m-%d-%H:%M:%S")
export PROJECT_DIR=${PWD}
export OMP_NUM_THREADS=16
export PROFDIR=${PWD}/${timestamp}_nsys_output_basic_4g
export MONAI_DATA_DIRECTORY=${PWD}/dataset_for_training
export RANDOM_PORT=$(shuf -i 20000-65000 -n 1)
export output_file="profile.%h.%p"
source setup_environment.sh
mkdir -p "$PROFDIR"
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
endpoint="${head_node_ip}:${RANDOM_PORT}"
srun --cpu-bind=cores -N1 --gpus=4 --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c "
     nsys profile \
         --cuda-memory-usage=true \
         --capture-range=cudaProfilerApi \
         --capture-range-end=stop \
         --output=${output_file} \
         -t cuda,nvtx \
         torchrun \
            --nnodes ${SLURM_NNODES} \
            --nproc_per_node ${NUM_GPUS} \
            --rdzv_id 10000 \
            --rdzv_backend c10d \
            --rdzv_endpoint $endpoint \
            --log_dir ${PROJECT_DIR}/log_torch \
            ${PROJECT_DIR}/script_basic_4g_p.py"
