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

export BATCH_SIZE=100
export NUM_WORKERS=8
export PRE_FETCH_FACTOR=2

TORCHRUN_COMMAND="torchrun \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_id 10000 \
    --rdzv_backend c10d \
    --rdzv_endpoint $endpoint \
    --log_dir ${PROJECT_DIR}/log_torch \
    ${PROJECT_DIR}/script_modded_4g.py"

srun --cpu-bind=cores -N1 --gpus=4 --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c "
    ${TORCHRUN_COMMAND}"

srun \
  --ntasks-per-node=1 \
  --gpus-per-task=4 \
  --cpus-per-task=8 \
  --hint=nomultithread \
  --gpu-bind=map_gpu:0,1,2,3 \
  --cpu-bind=cores \
  --mem-bind=local bash -c "${TORCHRUN_COMMAND}"
   