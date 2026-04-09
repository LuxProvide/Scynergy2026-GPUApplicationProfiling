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
import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, roc_auc_score
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader, Dataset, CacheDataset
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    EnsureTyped,
    RandAffine, 
)
from monai.utils import set_determinism
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import default_pg_timeout
from dataset_utils import build_mednist_index, MedNISTDataset, split_dataset
from visualization import show_example_images, write_convergence_plots
from distribute_utils import init_distributed, cleanup, is_main_process
from data_utils import get_data
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

def main():
    init_distributed()
    data_dir, root_dir = get_data()

    set_determinism(seed=0)
    (
        image_files_list,
        image_class,
        class_names,
        num_each,
        image_size,
        num_class,
    ) = build_mednist_index(data_dir)



    show_example_images(
        image_files_list=image_files_list,
        image_class=image_class,
        class_names=class_names,
    )

    VAL_FRAC = 0.1
    TEST_FRAC = 0.1

    train_x, train_y, val_x, val_y, test_x, test_y = split_dataset(
        image_files_list,
        image_class,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
        seed=42,
    )

    if is_main_process():
        print(
            f"Training count: {len(train_x)}, Validation count: "
            f"{len(val_x)}, Test count: {len(test_x)}"
        )

    val_transforms = Compose(
        [
            LoadImaged(keys="img"),
            EnsureChannelFirstd(keys="img"),
            ScaleIntensityd(keys="img"),
            EnsureTyped(keys="img", track_meta=False),
        ]
    )

    y_pred_trans = Compose([Activations(softmax=True)])
    y_trans = Compose([AsDiscrete(to_onehot=num_class)])

    cpu_transforms = Compose(
        [
            LoadImaged(keys="img"),
            EnsureChannelFirstd(keys="img"),
            ScaleIntensityd(keys="img"),
            EnsureTyped(keys="img", track_meta=False),
        ]
    )

    gpu_aug = RandAffine(
        prob=0.5,
        rotate_range=np.pi / 12,
        scale_range=0.1,
        padding_mode="zeros",
    )

    train_cached = CacheDataset(
        data=[{"img": x, "label": y} for x, y in zip(train_x, train_y)],
        transform=cpu_transforms,
        cache_rate=1.0,
        num_workers=8,
    )
    train_ds = Dataset(data=train_cached)

    val_ds = Dataset(
        data=[{"img": x, "label": y} for x, y in zip(val_x, val_y)],
        transform=val_transforms,
    )
    test_ds = Dataset(
        data=[{"img": x, "label": y} for x, y in zip(test_x, test_y)],
        transform=val_transforms,
    )

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=100,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(val_ds, batch_size=100, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=100, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_class).to(
        device, non_blocking=True
    )
    model = model.to(memory_format=torch.channels_last)
    if dist.is_initialized():
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
        model.register_comm_hook(None, default_hooks.fp16_compress_hook)

    eval_model = model.module if dist.is_initialized() else model
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    MAX_EPOCHS = 4
    VAL_INTERVAL = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    writer = SummaryWriter() if is_main_process() else None
    current_epoch = 0
    for epoch in range(MAX_EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        nvtx.range_push(f"epoch_{epoch + 1}")
        if is_main_process():
            current_epoch = current_epoch + 1
            print("-" * 10)
            print(f"epoch {epoch + 1}/{MAX_EPOCHS}")
            if current_epoch == 1:
                profiler.start()
        nvtx.range_push("training")
        model.train()
        nvtx.range_pop()
        epoch_loss = 0
        step = 0
        scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
        for batch_data in train_loader:
            nvtx.range_push("training_step")
            step += 1
            nvtx.range_push("wait_for_batch")
            inputs = batch_data["img"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True)
            inputs = gpu_aug(inputs)
            inputs = inputs.to(memory_format=torch.channels_last)
            nvtx.range_pop()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")
            ):
                nvtx.range_push("forward")
                outputs = model(inputs)
                nvtx.range_pop()
                nvtx.range_push("loss")
                loss = loss_function(outputs, labels)
                nvtx.range_pop()
            epoch_loss += loss.item()
            nvtx.range_push("ddp_backward_sync")
            scaler.scale(loss).backward()
            nvtx.range_pop()
            nvtx.range_push("optimizer_step")
            scaler.step(optimizer)
            nvtx.range_pop()
            scaler.update()
            if is_main_process():
                print(
                    f"{step}/{len(train_ds) // train_loader.batch_size}, "
                    f"train_loss: {loss.item():.4f}"
                )
            epoch_len = len(train_ds) // train_loader.batch_size
            if writer is not None:
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            nvtx.range_pop()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        if is_main_process():
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % VAL_INTERVAL == 0:
            if is_main_process():
                print(f"Epoch {epoch + 1}: validation phase")
            if is_main_process():
                nvtx.range_push("validation")
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
                        state_dict = (
                            model.module.state_dict()
                            if dist.is_initialized()
                            else model.state_dict()
                        )
                        torch.save(
                            state_dict, os.path.join(root_dir, "best_metric_model.pth")
                        )
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
                nvtx.range_pop()
            if dist.is_initialized():
                dist.barrier()
        nvtx.range_pop()
        if is_main_process() and current_epoch == 1:
            profiler.stop()
    if is_main_process():
        print(
            f"train completed, best_metric: {best_metric:.4f} "
            f"at epoch: {best_metric_epoch}"
        )
    if writer is not None:
        writer.close()
    profiler.stop()
    write_convergence_plots(
        epoch_loss_values=epoch_loss_values,
        metric_values=metric_values,
        VAL_INTERVAL=VAL_INTERVAL,
    )
    if dist.is_initialized():
        dist.barrier()
    ckpt = torch.load(
        os.path.join(root_dir, "best_metric_model.pth"), weights_only=True
    )
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
