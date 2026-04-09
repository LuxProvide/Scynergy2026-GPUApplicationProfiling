#!/bin/bash

# set -u

# ALREADY_IN_VENV=$(python -c "import sys; print('venv' in sys.prefix)")
# if [ "$ALREADY_IN_VENV" = "venv" ]; then
#     echo "Already in a virtual environment. Skipping setup."
#     exit 0
# fi

ALREADY_SETUP=${ALREADY_SETUP:-"false"}

if [ "$ALREADY_SETUP" = "true" ]; then
    echo "Environment already set up. Skipping setup."
else

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
        packages=("monai-weekly[pillow,tqdm]" "matplotlib" "tensorboard" "scikit-learn" "numpy" "requests" "torchvision" "black" "pylint")

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