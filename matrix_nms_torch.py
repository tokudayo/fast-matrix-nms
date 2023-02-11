import torch


def iou_matrix(a, b, norm=True):
    tl_i = torch.maximum(a[:, None, :2], b[:, :2])
    br_i = torch.minimum(a[:, None, 2:], b[:, 2:])

    pad = not norm and 1 or 0

    area_i = torch.prod(br_i - tl_i + pad, axis=2) * (tl_i < br_i).all(axis=2)
    area_a = torch.prod(a[:, 2:] - a[:, :2] + pad, axis=1)
    area_b = torch.prod(b[:, 2:] - b[:, :2] + pad, axis=1)
    area_o = area_a[:, None] + area_b - area_i
    return area_i / (area_o + 1e-10)


def matrix_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    use_gaussian=True,
    gaussian_sigma=0.5,
    threshold=0.4,
):
    """
    Matrix NMS.
    Args:
        boxes: [N, 4] float array.
        scores: [N] float array.
        conf_threshold: scalar, the confidence threshold.
        post_threshold: scalar, the post-processing threshold.
        max_keep: scalar, the maximum number of detections.
        normalized: bool, whether boxes is in normalized coordinates.
        use_gaussian: bool, whether to use gaussian filter to decay scores.
        gaussian_sigma: scalar, sigma of gaussian kernel.
    Returns:
        selected_indices: [M] int array.
    """
    N = boxes.shape[0]
    # Sort by conf
    scores, sorted_indices = scores.sort(descending=True)
    boxes = boxes[sorted_indices]

    # IoU matrix
    ious = iou_matrix(boxes, boxes)
    ious = torch.triu(ious, diagonal=1)
    ious_cmax = ious.max(0)[0].view(-1, 1)

    # Post threshold
    if use_gaussian:
        decay = torch.exp((ious_cmax.square() - ious.square()) / gaussian_sigma)
    else:
        decay = (1 - ious) / (1 - ious_cmax)
    decay = decay.min(0)[0]
    decayed_scores = scores * decay

    indices = torch.where(decayed_scores > threshold)[0]
    return sorted_indices[indices]
