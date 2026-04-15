
"""
Author: Marco Magliulo
Affiliation: LuxProvide
"""


import os
import torch.distributed as dist
from monai.apps import download_and_extract
from distribute_utils import is_main_process
import torch



def get_data(
    resource: str = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz",
    md5: str = "0bc7306e7427e00ad1c5526a6677552d",
    env_var: str = "MONAI_DATA_DIRECTORY",
):
    """
    Download (if needed) and return paths for the MedNIST dataset.

    Returns
    -------
    data_dir : str
        Path to the extracted dataset.
    root_dir : str
        Root directory where data is stored.
    """
    monai_data_directory = os.environ.get(env_var)

    if monai_data_directory is None:
        raise RuntimeError(
            f"Environment variable {env_var} is not set."
        )

    os.makedirs(monai_data_directory, exist_ok=True)
    root_dir = monai_data_directory

    if not os.path.exists(root_dir):
        raise RuntimeError(f"The path '{root_dir}' does not exist.")

    compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
    data_dir = os.path.join(root_dir, "MedNIST")

    if not os.path.exists(data_dir):
        if is_main_process():
            print(f"Downloading and extracting the data to {data_dir}")
            download_and_extract(resource, compressed_file, root_dir, md5)
    else:
        if is_main_process():
            print(f"The directory containing the data {data_dir} already exists")

    # Synchronize in distributed runs
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    return data_dir, root_dir



