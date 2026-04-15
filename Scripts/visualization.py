import numpy as np
import matplotlib.pyplot as plt
import PIL


def show_example_images(
    image_files_list,
    image_class,
    class_names,
    num_examples: int = 9,
    grid_size: tuple[int, int] = (3, 3),
    cmap: str = "gray",
):
    """
    Show a grid of random example images.

    Parameters
    ----------
    image_files_list : list[str]
        List of image file paths.
    image_class : list[int] or np.ndarray
        Class index for each image.
    class_names : list[str]
        Mapping from class index to class name.
    num_examples : int
        Number of images to show.
    grid_size : (int, int)
        Grid layout (rows, cols).
    cmap : str
        Matplotlib colormap.
    """
    NUM_TOTAL = len(image_files_list)

    plt.figure(figsize=(8, 8))

    for i, k in enumerate(np.random.randint(NUM_TOTAL, size=num_examples)):
        im = PIL.Image.open(image_files_list[k])
        arr = np.array(im)

        plt.subplot(grid_size[0], grid_size[1], i + 1)
        plt.imshow(arr, cmap=cmap, vmin=0, vmax=255)
        plt.xlabel(class_names[image_class[k]])
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("example_from_mednist_dataset.png")



def write_convergence_plots(
    epoch_loss_values,
    metric_values,
    VAL_INTERVAL: int,
    job_id: str | None = None,
    output_dir: str = ".",
):
    """
    Write training convergence plots (loss + validation metric).

    Parameters
    ----------
    epoch_loss_values : list[float]
        Average training loss per epoch.
    metric_values : list[float]
        Validation metric values.
    VAL_INTERVAL : int
        Validation interval in epochs.
    job_id : str | None
        Optional job ID (e.g. SLURM_JOBID). If None, read from environment.
    output_dir : str
        Directory where the plot is saved.
    """
    if job_id is None:
        job_id = os.environ.get("SLURM_JOBID", "default_job_id")

    plt.figure("train", (12, 6))

    # ---- Loss ----
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x_loss = range(1, len(epoch_loss_values) + 1)
    plt.xlabel("epoch")
    plt.plot(x_loss, epoch_loss_values)

    # ---- Validation metric ----
    plt.subplot(1, 2, 2)
    plt.title("Val AUC")
    x_metric = [(i + 1) * VAL_INTERVAL for i in range(len(metric_values))]
    plt.xlabel("epoch")
    plt.plot(x_metric, metric_values)

    filename = os.path.join(output_dir, f"training_plot_{job_id}.png")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    return filename
