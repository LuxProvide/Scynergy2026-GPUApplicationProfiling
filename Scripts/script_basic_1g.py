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
import PIL
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism
from data_utils import get_data
from dataset_utils import build_mednist_index, split_dataset, MedNISTDataset
from visualization import show_example_images, write_convergence_plots
from sklearn.metrics import classification_report

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


def main():
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

    y_pred_trans = Compose([Activations(softmax=True)])
    y_trans = Compose([AsDiscrete(to_onehot=num_class)])

    train_ds = MedNISTDataset(train_x, train_y, train_transforms)
    val_ds = MedNISTDataset(val_x, val_y, val_transforms)
    test_ds = MedNISTDataset(test_x, test_y, val_transforms)

    train_loader = DataLoader( train_ds, batch_size=300, shuffle=False, num_workers=0,)
    val_loader = DataLoader(val_ds, batch_size=300, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=300, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_class).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", "1"))
    VAL_INTERVAL = int(os.getenv("VAL_INTERVAL", "1"))
    auc_metric = ROCAUCMetric()

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    writer = SummaryWriter()

    ckpt_path = os.path.join(root_dir, "best_metric_model.pth")

    if USE_PROFILER:
        torch.cuda.profiler.start()

    for epoch in range(MAX_EPOCHS):
        nvtx.range_push(f"epoch_{epoch + 1}")
        print("-" * 10)
        print(f"epoch {epoch + 1}/{MAX_EPOCHS}")
        nvtx.range_push("training")
        model.train()
        nvtx.range_pop()
        epoch_loss = 0
        step = 0


        from torch.profiler import record_function

        data_iter = iter(train_loader)

        while True:
            try:
                if USE_PROFILER:
                    with record_function("dataloader"):
                        batch_data = next(data_iter)
                else:
                    batch_data = next(data_iter)
            except StopIteration:
                break

        # for batch_data in train_loader:
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
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}"
            )
            epoch_len = len(train_ds) // train_loader.batch_size
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            nvtx.range_pop()

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % VAL_INTERVAL == 0:
            print(f"Epoch {epoch + 1}: validation phase")
            nvtx.range_push("validation")
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                    )
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot
                metric_values.append(result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                if result > best_metric:
                    best_metric = result
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), ckpt_path)
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                    f" current accuracy: {acc_metric:.4f}"
                    f" best AUC: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
                writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
            nvtx.range_pop()
        nvtx.range_pop()
    print(
        f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}"
    )
    writer.close()

    if os.getenv("PERFORM_MODEL_EVALUATION", "false").lower() == "true":

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

    if data_dir is None:
        shutil.rmtree(root_dir)

    if USE_PROFILER:
        torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()