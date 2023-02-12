import torch


def self_iou_matrix(boxes):
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    lt = torch.max(boxes[:, None, :2], boxes[:, :2])  # [N,M,2]
    rb = torch.min(boxes[:, None, 2:], boxes[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area[:, None] + area - inter

    return torch.triu(inter / (union + 1e-10), diagonal=1)


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
    # Sort by conf
    scores, sorted_indices = scores.sort(descending=True)
    boxes = boxes[sorted_indices]

    # IoU matrix
    ious = self_iou_matrix(boxes)

    # p(f_i) estimation & enable broadcasting
    ious_cmax = ious.max(0)[0].view(-1, 1)

    # Decay factor
    if use_gaussian:
        decay = torch.exp((ious_cmax.square() - ious.square()) / gaussian_sigma)
    else:
        decay = (1 - ious) / (1 - ious_cmax)
    decay = decay.min(0)[0]

    # Select boxes
    decayed_scores = scores * decay
    indices = torch.where(decayed_scores > threshold)[0]

    return sorted_indices[indices]
