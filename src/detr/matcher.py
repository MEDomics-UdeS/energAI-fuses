"""
Modules to compute the matching cost and solve the corresponding LSAP.

  - Copyright Holder: Facebook, Inc. and its affiliates
  - Source: https://github.com/facebookresearch/detr
  - License: Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects)."""

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Args:
            cost_class(float, optional): This is the relative weight of the classification error in the matching cost (Default value = 1)
            cost_bbox(float, optional): This is the relative weight of the L1 error of the bounding box coordinates in the matching cost (Default value = 1)
            cost_giou(float, optional): This is the relative weight of the giou loss of the bounding box in the matching cost (Default value = 1)

        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Args:
            outputs: This is a dict that contains at least these entries
            targets: This is a list of targets

        Notes:
            index_i is the indices of the selected predictions (in order)
            index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)

        Returns:

        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(cost_class: float, cost_bbox: float, cost_giou: float):
    """

    Args:
        cost_class(float): 
        cost_bbox(float): 
        cost_giou(float): 

    Returns:

    """
    return HungarianMatcher(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)
