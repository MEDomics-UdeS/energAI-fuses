#import os
#import sys
from src.coco.coco_utils import get_coco
from src.coco.coco_eval import CocoEvaluator


if __name__ == '__main__':
    base_ds = get_coco(image_set='val',
                       transforms=None)

    # sys.stdout = open(os.devnull, 'w')  # block prints

    coco_evaluator = CocoEvaluator(coco_gt=base_ds.coco,
                                   iou_types=['bbox'])

    # sys.stdout = sys.__stdout__  # enable prints
