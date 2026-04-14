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
module load Python
NUM_GPUS=4
export case_name="basic_4g_p"
export USE_PROFILER="true"
export USE_DISTRIBUTED="true"
source setup_environment.sh


NSYS_OPTIONS="--cuda-memory-usage=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --output=${output_file} \
    -t cuda,nvtx"

TORCHRUN_COMMAND="torchrun \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_id 10000 \
    --rdzv_backend c10d \
    --rdzv_endpoint $endpoint \
    --log_dir ${PROJECT_DIR}/log_torch \
    ${PROJECT_DIR}/script_basic_4g.py"



if command -v srun /dev/null 2>&1 && [[ -n "$SLURM_JOB_ID" ]]; then
    srun --cpu-bind=cores -N1 --gpus=4 \
            --ntasks-per-node=1 --kill-on-bad-exit=1 \
            bash -c "nsys profile ${NSYS_OPTIONS} ${TORCHRUN_COMMAND}" 
           
else
    nsys profile ${NSYS_OPTIONS} ${TORCHRUN_COMMAND}
fi