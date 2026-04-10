# evaluation_utils.py

import os
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from distribute_utils import init_distributed, cleanup, is_main_process


def test_best_checkpoint(
    eval_model: torch.nn.Module,
    test_loader,
    root_dir: str,
    device: torch.device,
    is_main_process,
    checkpoint_name: str = "best_metric_model.pth",
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """
    Load the best checkpoint and run test inference.

    Parameters
    ----------
    eval_model : torch.nn.Module
        Model used for evaluation.
    test_loader :
        DataLoader for the test dataset.
    root_dir : str
        Directory containing the checkpoint.
    device : torch.device
        Target device for inference.
    is_main_process :
        Callable returning True only on the main process.
    checkpoint_name : str
        Name of the checkpoint file to load.

    Returns
    -------
    (y_true, y_pred) on main process
    (None, None) on non-main processes
    """

    checkpoint_path = os.path.join(root_dir, checkpoint_name)

    # # If running distributed, make sure checkpoint writing is finished
    # # before any process tries to read it.
    # if dist.is_available() and dist.is_initialized():
    #     print("Waiting for checkpoint to be available...", flush=True)
    #     dist.barrier()

    y_true: Optional[List[int]] = None
    y_pred: Optional[List[int]] = None


    # this works for both model wrapped with DDP or not 
    target_model = eval_model.module if hasattr(eval_model, "module") else eval_model

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    target_model.load_state_dict(ckpt)
    target_model.to(device)
    target_model.eval()

    print("Testing the trained model", flush=True)

    y_true = []
    y_pred = []

    with torch.inference_mode():
        for batch_data in test_loader:
            images, labels = batch_data
            test_images = images.to(device, non_blocking=True)
            test_labels = labels.to(device, non_blocking=True)

            pred = target_model(test_images).argmax(dim=1)

            y_true.extend(test_labels.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    return y_true, y_pred