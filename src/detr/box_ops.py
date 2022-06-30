"""
Utilities for bounding box manipulation and GIoU.

  - Copyright Holder: Facebook, Inc. and its affiliates
  - Source: https://github.com/facebookresearch/detr
  - License: Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0
"""
import torch
from torchvision.ops.boxes import box_area
from typing import List, Tuple


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """

    Args:
        x(torch.Tensor): 
        
    Returns:
        torch.Tensor: 

    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """

    Args:
        x(torch.Tensor): 
        
    Returns:
        torch.Tensor: 

    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def batch_box_xyxy_to_cxcywh(targets: List[dict], img_size: int) -> None:
    """Formats targets bbox for DE:TR model

    Args:
        targets(List[dict]): 
        img_size(int): 

    """
    for target in targets:
        boxes = target["boxes"]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # Normalizing the bboxes
        target["boxes"] = box_xyxy_to_cxcywh(boxes) / img_size


# modified from torchvision to also return the union
def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    Args:
        boxes1(torch.Tensor): 
        boxes2(torch.Tensor): 

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:

    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Generalized IoU from https://giou.stanford.edu/
    
    Args:
        boxes1(torch.Tensor): 
        boxes2(torch.Tensor): 
      
    Notes:
        The boxes should be in [x0, y0, x1, y1] format

    Returns:
        torch.Tensor: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)

    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area
