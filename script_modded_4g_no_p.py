"""
------------------------------------------------------------------------------
This script is the intellectual property of LuxProvide.

Authors: Marco Magliulo and Tom Walter

This code is proprietary and confidential. Unauthorized copying, distribution,
or modification of this file, via any medium, is strictly prohibited without
the express written consent of LuxProvide.

All rights reserved.
------------------------------------------------------------------------------
"""
import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader, Dataset, CacheDataset
#from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
#Using dicitonary transforms
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    RandFlip,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    EnsureTyped,
    RandAffine, 
    EnsureType
)
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from datetime import timedelta

class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]



def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=local_rank,
        )
        print(f">>> Initialized rank {dist.get_rank()} on GPU {local_rank}")
    else:
        print(">>> Running in single‑GPU mode (dist not initialized)")

def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_data():
    # Retrieve the environment variable
    monai_data_directory = os.environ.get("MONAI_DATA_DIRECTORY")
    # Assert that the environment variable is defined
    assert monai_data_directory is not None, "Environment variable MONAI_DATA_DIRECTORY is not set."

    if monai_data_directory is not None:
        os.makedirs(monai_data_directory, exist_ok=True)
    root_dir = tempfile.mkdtemp() if monai_data_directory is None else monai_data_directory 
    assert os.path.exists(monai_data_directory), f"The path '{monai_data_directory}' does not exist."

    resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
    md5 = "0bc7306e7427e00ad1c5526a6677552d"

    compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
    data_dir = os.path.join(root_dir, "MedNIST")
    if not os.path.exists(data_dir):
        if is_main_process():
            print(f"Downloading and extracting the data to {data_dir}")
        download_and_extract(resource, compressed_file, root_dir, md5)
    else:
        if is_main_process():
            print(f"The directory containing the data {data_dir} already exists")

    return data_dir, root_dir

def show_example_images():
    plt.subplots(3, 3, figsize=(8, 8))
    for i, k in enumerate(np.random.randint(num_total, size=9)):
        im = PIL.Image.open(image_files_list[k])
        arr = np.array(im)
        plt.subplot(3, 3, i + 1)
        plt.xlabel(class_names[image_class[k]])
        plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

def write_convergence_plots(epoch_loss_values, metric_values, val_interval):
    # Retrieve the SLURM job ID from the environment variable
    slurm_job_id = os.environ.get("SLURM_JOBID", "default_job_id")
    # Create the figure and subplots
    plt.figure("train", (12, 6))
    # Plot the Epoch Average Loss
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    # Plot the Validation AUC
    plt.subplot(1, 2, 2)
    plt.title("Val AUC")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    # Save the figure to a file named with the SLURM job ID
    filename = f"training_plot_{slurm_job_id}.png"
    plt.savefig(filename)
    # Optionally, close the plot to free up memory
    plt.close()


def main():
    init_distributed()
    data_dir, root_dir = get_data()

    set_determinism(seed=0)

    class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
    num_class = len(class_names)
    image_files = [
        [os.path.join(data_dir, class_names[i], x) for x in os.listdir(os.path.join(data_dir, class_names[i]))]
        for i in range(num_class)
    ]
    num_each = [len(image_files[i]) for i in range(num_class)]
    image_files_list = []
    image_class = []
    for i in range(num_class):
        image_files_list.extend(image_files[i])
        image_class.extend([i] * num_each[i])
    num_total = len(image_class)
    image_width, image_height = PIL.Image.open(image_files_list[0]).size
    if is_main_process():
        print(f"Total image count: {num_total}")
        print(f"Image dimensions: {image_width} x {image_height}")
        print(f"Label names: {class_names}")
        print(f"Label counts: {num_each}")



    val_frac = 0.1
    test_frac = 0.1
    length = len(image_files_list)
    indices = np.arange(length)
    np.random.shuffle(indices)

    test_split = int(test_frac * length)
    val_split = int(val_frac * length) + test_split
    test_indices = indices[:test_split]
    val_indices = indices[test_split:val_split]
    train_indices = indices[val_split:]

    train_x = [image_files_list[i] for i in train_indices]
    train_y = [image_class[i] for i in train_indices]
    val_x = [image_files_list[i] for i in val_indices]
    val_y = [image_class[i] for i in val_indices]
    test_x = [image_files_list[i] for i in test_indices]
    test_y = [image_class[i] for i in test_indices]
    if is_main_process():
        print(f"Training count: {len(train_x)}, Validation count: " f"{len(val_x)}, Test count: {len(test_x)}")


    val_transforms = Compose([
        LoadImaged(keys="img"),
        EnsureChannelFirstd(keys="img"),
        ScaleIntensityd(keys="img"),
        EnsureTyped(keys="img", track_meta=False)
    ])

    y_pred_trans = Compose([Activations(softmax=True)])
    y_trans = Compose([AsDiscrete(to_onehot=num_class)])

    cpu_transforms = Compose([
        LoadImaged(keys="img"),
        EnsureChannelFirstd(keys="img"),
        ScaleIntensityd(keys="img"),
        EnsureTyped(keys="img", track_meta=False),
    ])

    gpu_aug = RandAffine(
        prob=0.5,
        rotate_range=np.pi / 12,
        scale_range=0.1,
        padding_mode="zeros",
    )

    train_cached = CacheDataset(
        data=[{"img": x, "label": y} for x,y in zip(train_x,train_y)],
        transform=cpu_transforms,
        cache_rate=1.0,
        num_workers=8,
    )
    train_ds = Dataset(
        data=train_cached
    )
    val_ds = Dataset(
        data=[{"img": x, "label": y} for x,y in zip(val_x,val_y)],
        transform=val_transforms
    )
    test_ds = Dataset(
        data=[{"img": x, "label": y} for x,y in zip(test_x,test_y)],
        transform=val_transforms
    )


    train_sampler = (
        DistributedSampler(train_ds)
        if dist.is_initialized()
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=100,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(val_ds, batch_size=100, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=100, num_workers=8, pin_memory=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_class).to(device, non_blocking=True)
    model = model.to(memory_format=torch.channels_last)
    if dist.is_initialized():
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
        model.register_comm_hook(None, default_hooks.fp16_compress_hook)

    eval_model = model.module if dist.is_initialized() else model
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    max_epochs = 4
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    writer = SummaryWriter() if is_main_process() else None
    for epoch in range(max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main_process():
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
        for batch_data in train_loader:
            step += 1
            inputs = batch_data["img"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True)
            inputs = gpu_aug(inputs)
            inputs = inputs.to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            epoch_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if is_main_process():
                print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
            epoch_len = len(train_ds) // train_loader.batch_size
            if writer is not None:
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        if is_main_process():
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            if is_main_process():
                print(f"Epoch {epoch + 1}: validation phase")
            if is_main_process():
                eval_model.eval()
                with torch.no_grad():
                    y_pred = torch.tensor([], dtype=torch.float32, device=device)
                    y = torch.tensor([], dtype=torch.long, device=device)
                    for val_data in val_loader:
                        val_images = val_data["img"].to(device, non_blocking=True)
                        val_labels = val_data["label"].to(device, non_blocking=True)
                        y_pred = torch.cat([y_pred, eval_model(val_images)], dim=0)
                        y = torch.cat([y, val_labels], dim=0)
                    y_prob = torch.softmax(y_pred, dim=1).detach().cpu().numpy()
                    y_true_np = y.detach().cpu().numpy()

                    result = roc_auc_score(
                        y_true_np,
                        y_prob,
                        multi_class="ovr",
                        average="macro",
                    )
                    metric_values.append(result) 
                    acc_value = torch.eq(y_pred.argmax(dim=1), y)
                    acc_metric = acc_value.sum().item() / len(acc_value)
                    if result > best_metric:
                        best_metric = result
                        best_metric_epoch = epoch + 1
                        state_dict = model.module.state_dict() if dist.is_initialized() else model.state_dict()
                        torch.save(state_dict, os.path.join(root_dir, "best_metric_model.pth"))
                        if is_main_process():
                            print("saved new best metric model")
                    if is_main_process():
                        print(
                        f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                        f" current accuracy: {acc_metric:.4f}"
                        f" best AUC: {best_metric:.4f}"
                        f" at epoch: {best_metric_epoch}"
                    )
                    if writer is not None:
                        writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
            if dist.is_initialized():
                dist.barrier()
    if is_main_process():
        print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
    if writer is not None:
        writer.close()
    if is_main_process():
        write_convergence_plots(epoch_loss_values, metric_values, val_interval)
    if dist.is_initialized():
        dist.barrier()
    ckpt = torch.load(os.path.join(root_dir, "best_metric_model.pth"), weights_only=True)
    eval_model.load_state_dict(ckpt)
    eval_model.eval()
    if is_main_process():
        print("Testing the trained model", flush=True)
    if dist.is_initialized():
        dist.barrier()

    ckpt = torch.load(
        os.path.join(root_dir, "best_metric_model.pth"),
        map_location=device,
        weights_only=True,
    )

    eval_model.load_state_dict(ckpt)
    eval_model.eval()

    if is_main_process():
        print("Testing the trained model", flush=True)
        y_true = []
        y_pred = []

        with torch.no_grad():
            for test_data in test_loader:
                test_images = test_data["img"].to(device, non_blocking=True)
                test_labels = test_data["label"].to(device, non_blocking=True)
                pred = eval_model(test_images).argmax(dim=1)
                y_true.extend(test_labels.cpu().tolist())
                y_pred.extend(pred.cpu().tolist())

    if dist.is_initialized():
        dist.barrier()

    cleanup()


if __name__ == "__main__":
    main()
