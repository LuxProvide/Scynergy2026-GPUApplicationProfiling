#!/bin/bash -l
#SBATCH --job-name=monai_basic_1g_no_p
#SBATCH --output=monai_basic_1g_no_p.out
#SBATCH --error=monai_basic_1g_no_p.err
#SBATCH --time=02:00:00
#SBATCH --qos=dev
#SBATCH --reservation=gpudev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --account=lxp
#SBATCH --partition=gpu

module load env/release/2025.1
module load Python
export PROJECT_DIR=${PWD}
export OMP_NUM_THREADS=1
export MONAI_DATA_DIRECTORY=${PWD}/dataset_for_training
source setup_environment.sh

srun --cpu-bind=cores -N1 --gpus=1 --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c "
    torchrun \
    --nproc_per_node 1 \
    --log_dir ${PROJECT_DIR}/log_torch \
    ${PROJECT_DIR}/script_basic_1g_no_p.py"