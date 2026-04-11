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
import torch

USE_PROFILER = os.getenv("USE_PROFILER", "false").lower() in ("true", "1")

if USE_PROFILER:
    import torch.cuda.profiler as profiler
    import torch.cuda.nvtx as nvtx
else:
    class MockProfiler:
        @staticmethod
        def start(): pass
        @staticmethod
        def stop(): pass
    class MockNVTX:
        @staticmethod
        def range_push(name): pass
        @staticmethod
        def range_pop(): pass
    profiler = MockProfiler()
    nvtx = MockNVTX()

import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from monai.data import DataLoader, Dataset, CacheDataset
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
from dataset_utils import build_mednist_index, split_dataset
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



    # if is_main_process():
    #     show_example_images(
    #         image_files_list=image_files_list,
    #         image_class=image_class,
    #         class_names=class_names,
    #     )

    VAL_FRAC = 0.1
    TEST_FRAC = 0.1

    train_x, train_y, val_x, val_y, test_x, test_y = split_dataset(
        image_files_list,
        image_class,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
        seed=42,
    )

    val_transforms = Compose(
        [
            LoadImaged(keys="img"),
            EnsureChannelFirstd(keys="img"),
            ScaleIntensityd(keys="img"),
            EnsureTyped(keys="img", track_meta=False),
        ]
    )

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
        progress=is_main_process(),
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
    
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", 8))
    PREFETCH_FACTOR = int(os.getenv("PRE_FETCH_FACTOR", 2))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
    )

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

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

    MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", "1"))
    VAL_INTERVAL = int(os.getenv("VAL_INTERVAL", "1"))
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    writer = SummaryWriter() if is_main_process() else None
    current_epoch = 0

    profiler.start()
    for epoch in range(MAX_EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        nvtx.range_push(f"epoch_{epoch + 1}")
        if is_main_process():
            print("-" * 10)
            print(f"epoch {epoch + 1}/{MAX_EPOCHS}")
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
            
            if step % 50 == 0 and is_main_process():
                 print(f"Epoch {epoch + 1} Step {step}/{len(train_loader)} - loss: {loss.item():.4f}")

            nvtx.range_pop()

        epoch_loss /= step
        if dist.is_initialized():
             dist.all_reduce(torch.tensor(epoch_loss, device=device)) # Aggregate only once per epoch
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
                    y_pred_list = []
                    y_list = []
                    for val_data in val_loader:
                        val_images = val_data["img"].to(device, non_blocking=True)
                        val_labels = val_data["label"].to(device, non_blocking=True)
                        y_pred_list.append(eval_model(val_images))
                        y_list.append(val_labels)
                    
                    y_pred = torch.cat(y_pred_list, dim=0)
                    y = torch.cat(y_list, dim=0)
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
        current_epoch += 1
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

    if os.getenv("PERFORM_MODEL_EVALUATION", "false").lower() == "true":
        from evaluate_trained_model_utils import test_best_checkpoint
        if is_main_process():
            write_convergence_plots(
                epoch_loss_values=epoch_loss_values,
                metric_values=metric_values,
                VAL_INTERVAL=VAL_INTERVAL,
            )

            from evaluate_trained_model_utils import test_best_checkpoint
            y_true, y_pred = test_best_checkpoint(
                eval_model=model,
                test_loader=test_loader,
                root_dir=root_dir,
                device=device,
                is_main_process=lambda: True,  # Not running distributed, so always main process
            )

            print(classification_report(y_true, y_pred, target_names=class_names))

    cleanup()


if __name__ == "__main__":
    main()
