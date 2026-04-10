#!/bin/bash -l
#SBATCH --job-name=monai_modded_4g_no_p
#SBATCH --output=monai_modded_4g_no_p.out
#SBATCH --error=monai_modded_4g_no_p.err
#SBATCH --time=00:30:00
#SBATCH --qos=dev
#SBATCH --reservation=gpudev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --account=lxp
#SBATCH --partition=gpu

module load env/release/2025.1
module load Python
NUM_GPUS=4
export case_name="modded_4g_no_p"
export USE_PROFILER="false"
export USE_DISTRIBUTED="true"
source setup_environment.sh

srun --cpu-bind=cores -N1 --gpus=4 --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c "
    torchrun \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_id 10000 \
    --rdzv_backend c10d \
    --rdzv_endpoint $endpoint \
    --log_dir ${PROJECT_DIR}/log_torch \
    ${PROJECT_DIR}/script_modded_4g_no_p.py"