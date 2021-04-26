import torch
from torch import index_select
from torchvision.ops import nms


def filter_by_nms(preds_list, iou_threshold):
    keep_nms = [nms(pred['boxes'], pred['scores'], iou_threshold) for pred in preds_list]

    preds_nms = []

    for pred, keep in zip(preds_list, keep_nms):
        preds_nms.append({key: index_select(val, dim=0, index=keep) for key, val in pred.items()})

    return preds_nms


def filter_by_score(preds_list, score_threshold):
    preds_filt = []

    device = None

    for pred in preds_list:
        keep = []

        for index, score in enumerate(pred['scores']):
            if score.greater(score_threshold):
                keep.append(index)

                if device is None:
                    device = score.device

        preds_filt.append({key: index_select(val, dim=0, index=torch.tensor(keep, device=device))
                           for key, val in pred.items()})

    return preds_filt


def print_args(args_dict):
    print('\n=== Arguments & Hyperparameters ===\n')

    for key, value in args_dict.items():
        print(f'{key}:{" " * (27 - len(key))}{value}')

    print('\n')
