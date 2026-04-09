#!/bin/bash

VENV_NAME=Scynergy_venv

if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment..."
    python -m venv --without-pip "$VENV_NAME"
    source "$VENV_NAME/bin/activate"
    echo "Installing pip..."
    curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
    rm get-pip.py
else
    source "$VENV_NAME/bin/activate"
fi
pip install --upgrade pip
check_that_packages_are_installed="true"
if [ "$check_that_packages_are_installed" = "true" ]; then
    packages=("monai-weekly[pillow,tqdm]" "matplotlib" "tensorboard" "scikit-learn" "numpy" "requests" "torchvision")

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
