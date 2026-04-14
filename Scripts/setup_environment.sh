#!/bin/bash

set -u

echo "----------------------------------"

# ALREADY_IN_VENV=$(python -c "import sys; print('venv' in sys.prefix)")
# if [ "$ALREADY_IN_VENV" = "venv" ]; then
#     echo "Already in a virtual environment. Skipping setup."
#     exit 0
# fi
module load Python

ALREADY_SETUP=${ALREADY_SETUP:-"false"}

if [ "$ALREADY_SETUP" = "true" ]; then
    echo "Environment already set up. Skipping setup."
else

    export COLLECTED_TRACES_DIR=/mnt/tier2/project/p201259/materials/15April_GPUApp_Profiling/ProfilingTraces
    export TRACE_1GPU_BASE=${COLLECTED_TRACES_DIR}/single_gpu_base.nsys-rep
    export TRACE_4GPU_BASE=${COLLECTED_TRACES_DIR}/multi_gpu_base.nsys-rep
    export TRACE_GPU_OPTIM=${COLLECTED_TRACES_DIR}/multi_gpu_optim.nsys-rep

    VENV_DIR=$(realpath ..)

    VENV_NAME=Scynergy_venv

    if [ ! -d "$VENV_DIR/$VENV_NAME" ]; then
        cd "$VENV_DIR"
        echo "Creating virtual environment..."
        python -m venv --without-pip "$VENV_NAME"
        source "$VENV_NAME/bin/activate"
        echo "Installing pip..."
        curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python get-pip.py
        rm get-pip.py
    fi

    source "$VENV_DIR/$VENV_NAME/bin/activate"
    pip install --upgrade pip
    check_that_packages_are_installed="true"
    if [ "$check_that_packages_are_installed" = "true" ]; then
        packages=("monai-weekly[pillow,tqdm]" "matplotlib" "tensorboard" "scikit-learn" "numpy" "requests" "torchvision" "black" "pylint" "autonvtx")

        is_installed() {
            python -c "import $1" &> /dev/null
        }

        missing_package=false
        for package in "${packages[@]}"; do
            if [ "$package" = "monai-weekly[pillow, tqdm]" ]; then
                echo "Checking if mon ai is installed. This can take some time..."
                if ! python -c "import monai.config" &> /dev/null ; then
                    missing_package=true
                    echo "${package} is not installed"
                    pip install $package
                fi
            else 
                echo "Checking if ${package} is installed"
                if ! is_installed "$package"; then
                    missing_package=true
                    echo "${package} is not installed"
                    pip install $package
                fi
            fi
        done
    fi

    export ALREADY_SETUP="true"

fi


timestamp=$(date +"%Y_%m_%d_%H_%M_%S")
if [ "$USE_PROFILER" = "true" ]; then
    module load Nsight-Systems
    export PROFDIR=${PWD}/${timestamp}_${case_name}
    export output_file=$PROFDIR/${case_name}".%h.%p"
    mkdir -p "$PROFDIR"
    export TORCH_NVTX_RANGE=1
fi

export PROJECT_DIR=${PWD}
export OMP_NUM_THREADS=16
export MONAI_DATA_DIRECTORY=${PWD}/dataset_for_training

if [ "$USE_DISTRIBUTED" = "true" ]; then
    export RANDOM_PORT=$(shuf -i 20000-65000 -n 1)
    export head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
    export endpoint="${head_node_ip}:${RANDOM_PORT}"
else
    echo "Running in non-distributed mode. No need to set up distributed environment variables."
fi

export MAX_EPOCHS=1
export VAL_INTERVAL=1
export PERFORM_MODEL_EVALUATION="false"
echo "the number of epochs is set to ${MAX_EPOCHS} and the validation interval is set to ${VAL_INTERVAL}"

echo "Environment setup complete."

echo "----------------------------------"


set +u