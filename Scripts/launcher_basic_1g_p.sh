#!/bin/bash -l
# Author: Marco Magliulo (LuxProvide)
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
export OMP_NUM_THREADS=1
export NUM_GPUS=1
export USE_PROFILER="true"
export USE_DISTRIBUTED="false"
export case_name="basic_1g_p"
source setup_environment.sh

TORCHRUN_COMMAND="torchrun \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node ${NUM_GPUS} \
    --log_dir ${PROJECT_DIR}/log_torch \
    ${PROJECT_DIR}/script_basic_1g.py"


NSYS_OPTIONS="--cuda-memory-usage=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --output=${output_file} \
    -t cuda,nvtx"

if command -v srun /dev/null 2>&1 && [[ -n "$SLURM_JOB_ID" ]]; then
    srun --cpu-bind=cores -N1 --gpus=4 \
            --ntasks-per-node=1 --kill-on-bad-exit=1 \
            bash -c "nsys profile ${NSYS_OPTIONS} ${TORCHRUN_COMMAND}" 
           
else
    nsys profile ${NSYS_OPTIONS} ${TORCHRUN_COMMAND}
fi