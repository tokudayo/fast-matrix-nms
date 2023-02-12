import torch
import numpy as np
from torchvision.ops import nms
from tqdm import trange
from matrix_nms import fast_matrix_nms
from torchvision.ops import nms
from matrix_nms_torch import matrix_nms as naive_matrix_nms
import torch.utils.benchmark as benchmark

device = "cuda"


def create_bbox_data_generator(size=16 * 200, rep=100000, seed=0, device="cuda"):
    np.random.seed(seed)
    torch.manual_seed(seed)
    for _ in range(rep):
        bbox = torch.randint(0, 1000, (size, 4), dtype=torch.float32)
        bbox[:, 2:] += bbox[:, :2]
        scores = torch.rand(size, dtype=torch.float32)
        yield bbox.to(device).contiguous(), scores.to(device).contiguous()


def random_box(size=16 * 200, device="cuda"):
    bbox = torch.randint(0, 1000, (size, 4), dtype=torch.float32)
    bbox[:, 2:] += bbox[:, :2]
    scores = torch.rand(size, dtype=torch.float32)
    return bbox.to(device).contiguous(), scores.to(device).contiguous()


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    results = []
    for size in trange(160, 3201, 160):
        boxes, scores = random_box(size)
        results.append(
            benchmark.Timer(
                "matrix_nms(boxes, scores, 0.4)",
                globals={"matrix_nms": fast_matrix_nms, "boxes": boxes, "scores": scores},
                label=f"Non-max suppression",
                sub_label=f"{size}",
                description="Fast Matrix NMS",
            ).blocked_autorange(min_run_time=2)
        )
        results.append(
            benchmark.Timer(
                "nms(boxes, scores, 0.1)",
                globals={"nms": nms, "boxes": boxes, "scores": scores},
                label=f"Non-max suppression",
                sub_label=f"{size}",
                description="Torchvision NMS CUDA",
            ).blocked_autorange(min_run_time=2)
        )
        results.append(
            benchmark.Timer(
                "naive_matrix_nms(boxes, scores, 0.4)",
                globals={
                    "naive_matrix_nms": naive_matrix_nms,
                    "boxes": boxes,
                    "scores": scores,
                },
                label=f"Non-max suppression",
                sub_label=f"{size}",
                description="Naive Matrix NMS",
            ).blocked_autorange(min_run_time=2)
        )
        results.append(
            benchmark.Timer(
                "nms(boxes, scores, 0.1)",
                globals={"nms": nms, "boxes": boxes.cpu(), "scores": scores.cpu()},
                label=f"Non-max suppression",
                sub_label=f"{size}",
                description="Torchvision NMS CPU",
            ).blocked_autorange(min_run_time=2)
        )


    compare = benchmark.Compare(results)
    compare.print()
