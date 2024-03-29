"""
File:
    src/utils/helper_functions.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Helper functions used in the project
"""

import csv
import json
import os
import sys
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from google_images_download import google_images_download
from torchvision.ops import nms
from typing import List
import torch.nn.functional as F
from pathlib import PurePosixPath, PureWindowsPath

from src.detr.box_ops import box_cxcywh_to_xyxy
from src.utils.constants import REQUIRED_PYTHON, IMAGE_EXT


def count_images(path: str,
                 minimum: int = 20) -> None:
    """Function to count images in some folders

    Args:
        path(str): directory in which to seek images
        minimum(int, optional): minimum number of images required for the number of images to be plotted (Default value = 20)

    """
    fuse_dict = {}

    folders = ([name for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name))])

    for folder in folders:
        contents = os.listdir(os.path.join(path, folder))

        if len(contents) > minimum:
            fuse_dict[folder] = len(contents)

    x = sorted(fuse_dict, key=fuse_dict.get, reverse=True)
    y = sorted(fuse_dict.values(), reverse=True)

    plt.bar(x, y)
    plt.show()
    plt.xticks(rotation=90)
    plt.ylabel("# of images")
    plt.grid(axis='y')
    plt.tight_layout()


def env_tests() -> None:
    """Environment tests"""
    system_major = sys.version_info.major

    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(
            REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version))


def cp_split(filepath: str) -> List[str]:
    """Cross-platform filepath splitting. Supports Windows, OS X and Linux.

    Args:
        filepath(str): The complete windows or posix filepath

    Returns:
        List[str]: A list of every individual elements in the filepath
        
    """
    # Check for the system's platform
    if sys.platform == "win32" or sys.platform == "cygwin":
        return PureWindowsPath(filepath).parts
    else:
        return PurePosixPath(filepath).parts


def filter_by_nms(preds_list: List[dict],
                  iou_threshold: float) -> List[dict]:
    """Function to filter a bounding boxes predictions list by using non-maximum suppression

    Args:
        preds_list(List[dict]): predicted bounding boxes
        iou_threshold(float): iou threshold for non-maximum suppression

    Returns: filtered predictions list

    """
    keep_nms = [nms(pred['boxes'], pred['scores'], iou_threshold) for pred in preds_list]

    preds_nms = []

    for pred, keep in zip(preds_list, keep_nms):
        preds_nms.append({key: torch.index_select(val, dim=0, index=keep) for key, val in pred.items()})

    return preds_nms


def filter_by_score(preds_list: List[dict],
                    score_threshold: float) -> List[dict]:
    """Function to filter a bounding boxes predictions list by using a confidence score threshold

    Args:
        preds_list(List[dict]): predicted bounding boxes
        score_threshold(float): confidence score threshold above which predictions are to be saved

    Returns: filtered predictions list

    """
    preds_filt = []

    for pred in preds_list:
        keep = []

        for index, score in enumerate(pred['scores']):
            if score.greater(score_threshold):
                keep.append(index)

        preds_filt.append({key: torch.index_select(val, dim=0, index=torch.tensor(keep,
                                                                                  device=pred['scores'].device).long())
                           for key, val in pred.items()})

    return preds_filt


def google_image_scraper(chrome_driver_path: str = '../chromedriver.exe',
                         prefix: str = 'English Electric C',
                         postfix: str = 'J') -> None:
    """Function to scrape images from Google images

    Args:
        chrome_driver_path(str, optional): chromedriver.exe path (Default value = '../chromedriver.exe')
        prefix(str, optional): prefix for fuse search criterion (Default value = 'English Electric C')
        postfix(str, optional): postfix for fuse search criterion (Default value = 'J')

    """
    response = google_images_download.googleimagesdownload()

    amps = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80, 90, 100,
            110, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 600]

    for i in amps:
        keywords = f'{prefix}{i}{postfix}'

        arguments = {
            'keywords': keywords,
            'limit': 100,
            'save_source': f'source_{keywords}',
            'chromedriver': chrome_driver_path
        }
        paths = response.download(arguments)


def json_to_csv(dir_list: List[str]) -> None:
    """Function to convert multiple json files into a single csv file

    Args:
        dir_list(List[str]): list containing the directories in which to fetch the jsons

    """
    for directory in dir_list:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        print(directory)
        for i in tqdm(files):
            json_file = open(directory + i)
            json_data = json.load(json_file)
            csv_file = open('ground_truth.csv', 'a', newline='')
            csv_writer = csv.writer(csv_file)
            try:
                for x in json_data['outputs']['object']:
                    path = i
                    label = x['name']
                    xmin = x['bndbox']['xmin']
                    ymin = x['bndbox']['ymin']
                    xmax = x['bndbox']['xmax']
                    ymax = x['bndbox']['ymax']
                    item = [path, label, xmin, ymin, xmax, ymax]
                    csv_writer.writerow(item)
                csv_file.close()
            except Exception:
                split_path = json_data['path'].split('\\')
                path = split_path[-1]
                label = split_path[-2]
                item = [path, label]
                csv_writer.writerow(item)
                csv_file.close()


def print_dict(dictionary: dict,
               n_spaces: int,
               str_format: str = None) -> None:
    """Function to print dictionary keys and values

    Args:
        dictionary(dict): dictionary
        n_spaces(int): number of spaces between keys and values
        str_format(str, optional): format string for floating point values (Default value = None)

    """
    max_key_length = max(map(len, dictionary)) + n_spaces

    if str_format is None:
        for key, value in dictionary.items():
            print(f'{key}:{" " * (max_key_length - len(key))}{value}')
    else:
        for key, value in dictionary.items():
            print(f'{key}:{" " * (max_key_length - len(key))}{value:{str_format}}')


def rename_photos(root_dir: str) -> None:
    """Rename all photos in a folder's subfolders

    Args:
        root_dir(str): str, root directory

    """
    for subdir, dirs, files in os.walk(root_dir):
        for i, file in enumerate(files, start=1):
            os.rename(subdir + os.sep + file, subdir + "-" + str(i) + f".{IMAGE_EXT}")


def format_detr_outputs(outputs: List[dict], target_sizes: torch.Tensor, device: torch.device) -> List[dict]:
    """Function to format DETR outputs

    Args:
        outputs(List[dict]): DETR outputs
        target_sizes(torch.Tensor): target sizes
        device(torch.device): device (cpu or CUDA)

    Returns: formatted DETR outputs

    """

    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

    assert len(out_logits) == len(target_sizes), "target_sizes tensor must be of size 'batch_size' in 1st dimension"
    assert target_sizes.shape[1] == 2, "target_sizes tensor must be of size '2' in 2nd dimension"

    prob = F.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)

    # convert to [x0, y0, x1, y1] format
    boxes = box_cxcywh_to_xyxy(out_bbox)
    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(device)
    boxes = boxes * scale_fct[:, None, :]

    results = [{'boxes': b, 'labels': l, 'scores': s}
                for b, l, s in zip(boxes, labels, scores)]

    return results


def enter_default_json(file) -> None:
    """Function to enter default JSON

    Args:
        file: file name

    """
    # Loading in the default values for inference
    iou_threshold = "0.5"
    score_threshold = "0.5"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Creating the settings dictionnary
    settings_dict = {"iou_threshold": iou_threshold,
                     "score_threshold": score_threshold,
                     "device": device}

    # Saving the settings in json file
    json.dump(settings_dict, file)
