import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch



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