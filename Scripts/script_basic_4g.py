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

import torch.distributed as dist
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, roc_auc_score
from monai.data import DataLoader
from monai.networks.nets import DenseNet121
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel as DDP
from data_utils import get_data
from distribute_utils import init_distributed, cleanup, is_main_process
from dataset_utils import build_mednist_index, MedNISTDataset
from visualization import show_example_images, write_convergence_plots
from dataset_utils import build_mednist_index, split_dataset


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


    # show_example_images(
    #     image_files_list=image_files_list,
    #     image_class=image_class,
    #     class_names=class_names,
    # )


    VAL_FRAC = 0.1
    TEST_FRAC = 0.1
   

    train_x, train_y, val_x, val_y, test_x, test_y = split_dataset(
        image_files_list,
        image_class,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
        seed=42,
    )


    train_transforms = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        ]
    )

    val_transforms = Compose(
        [LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()]
    )


    train_ds = MedNISTDataset(train_x, train_y, train_transforms)
    val_ds = MedNISTDataset(val_x, val_y, val_transforms)
    test_ds = MedNISTDataset(test_x, test_y, val_transforms)


    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=300,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
    )

    val_loader = DataLoader(val_ds, batch_size=300, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=300, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_class).to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", "1"))
    VAL_INTERVAL = int(os.getenv("VAL_INTERVAL", "1"))

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
        for batch_data in train_loader:
            nvtx.range_push("training_step")
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            nvtx.range_push("forward_backward")
            optimizer.zero_grad()
            nvtx.range_pop()
            nvtx.range_push("model_inference")
            outputs = model(inputs)
            nvtx.range_pop()
            nvtx.range_push("loss_computation")
            loss = loss_function(outputs, labels)
            nvtx.range_pop()
            nvtx.range_push("backward_pass")
            loss.backward()
            nvtx.range_pop()
            nvtx.range_push("optimizer_step")
            optimizer.step()
            nvtx.range_pop()
            epoch_loss += loss.item()
            if is_main_process():
                print(f"{step}/{len(train_loader)}, " f"train_loss: {loss.item():.4f}")
            epoch_len = len(train_loader)
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
                nvtx.range_push("validation")

                eval_model = model.module if dist.is_initialized() else model
                eval_model.eval()

                with torch.no_grad():
                    y_pred_list = []
                    y_true_list = []

                    for val_data in val_loader:
                        val_images = val_data[0].to(device)
                        val_labels = val_data[1].to(device)

                        logits = eval_model(val_images)
                        y_pred_list.append(logits)
                        y_true_list.append(val_labels)

                    y_pred = torch.cat(y_pred_list, dim=0)
                    y = torch.cat(y_true_list, dim=0)

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
                        print("saved new best metric model")

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


    if os.getenv("PERFORM_MODEL_EVALUATION", "false").lower() == "true":

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