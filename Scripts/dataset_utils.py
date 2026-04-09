import torch
import os
import PIL
from distribute_utils import is_main_process


class MedNISTDataset(torch.utils.data.Dataset):
    """
    Simple Dataset for MedNIST images.
    """

    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

        if len(self.image_files) != len(self.labels):
            raise ValueError(
                "image_files and labels must have the same length"
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = self.image_files[index]
        label = self.labels[index]

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label


def build_mednist_index(data_dir, verbose: bool = True):
    """
    Scan a MedNIST directory and build image paths, labels, and metadata.

    Parameters
    ----------
    data_dir : str
        Root directory of the MedNIST dataset.
    verbose : bool
        Whether to print dataset statistics.

    Returns
    -------
    image_files_list : list[str]
        Flat list of image file paths.
    image_class : list[int]
        Class index for each image.
    class_names : list[str]
        Sorted class names.
    num_each : list[int]
        Number of images per class.
    image_size : tuple[int, int]
        (width, height) of images.
    """
    # Discover class names
    class_names = sorted(
        x for x in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, x))
    )
    num_class = len(class_names)

    # Collect image paths per class
    image_files = [
        [
            os.path.join(data_dir, class_names[i], x)
            for x in os.listdir(os.path.join(data_dir, class_names[i]))
        ]
        for i in range(num_class)
    ]

    # Count per class
    num_each = [len(files) for files in image_files]

    # Flatten lists and build labels
    image_files_list = []
    image_class = []
    for i, files in enumerate(image_files):
        image_files_list.extend(files)
        image_class.extend([i] * len(files))

    num_total = len(image_class)

    # Read image size from first image
    image_width, image_height = PIL.Image.open(image_files_list[0]).size

    if verbose:
        print(f"Total image count: {num_total}")
        print(f"Image dimensions: {image_width} x {image_height}")
        print(f"Label names: {class_names}")
        print(f"Label counts: {num_each}")

    if is_main_process():
        print(f"Total image count: {num_total}")
        print(f"Image dimensions: {image_width} x {image_height}")
        print(f"Label names: {class_names}")
        print(f"Label counts: {num_each}")

    return (
        image_files_list,
        image_class,
        class_names,
        num_each,
        (image_width, image_height),
        len(class_names),
    )



import numpy as np


def split_dataset(
    image_files_list,
    image_class,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int | None = None,
    verbose: bool = True,
):
    """
    Split dataset into train / validation / test sets.

    Parameters
    ----------
    image_files_list : list[str]
        Flat list of image paths.
    image_class : list[int]
        Class labels.
    val_frac : float
        Fraction of data used for validation.
    test_frac : float
        Fraction of data used for testing.
    seed : int | None
        Random seed for reproducibility.
    verbose : bool
        Whether to print split statistics.

    Returns
    -------
    train_x, train_y, val_x, val_y, test_x, test_y
    """
    if len(image_files_list) != len(image_class):
        raise ValueError("image_files_list and image_class must have the same length")

    length = len(image_files_list)

    if seed is not None:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(length)
    else:
        indices = np.random.permutation(length)

    test_split = int(test_frac * length)
    val_split = test_split + int(val_frac * length)

    test_indices = indices[:test_split]
    val_indices = indices[test_split:val_split]
    train_indices = indices[val_split:]

    train_x = [image_files_list[i] for i in train_indices]
    train_y = [image_class[i] for i in train_indices]

    val_x = [image_files_list[i] for i in val_indices]
    val_y = [image_class[i] for i in val_indices]

    test_x = [image_files_list[i] for i in test_indices]
    test_y = [image_class[i] for i in test_indices]

    if verbose:
        if is_main_process():
            print(
                f"Training count: {len(train_x)}, "
                f"Validation count: {len(val_x)}, "
                f"Test count: {len(test_x)}"
            )

    return train_x, train_y, val_x, val_y, test_x, test_y