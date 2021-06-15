"""
File:
    src/utils/helper_functions.py

Authors:
    - Simon Giard-Leroux
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
from argparse import Namespace

from src.utils.constants import REQUIRED_PYTHON, IMAGE_EXT


def count_images(path: str = "C:/Users/gias2402/Google Drive/"
                             "Maîtrise SGL CIMA+/General/Fuses Survey Dataset 2",
                 minimum: int = 20) -> None:
    """
    Function to count images in some folders

    :param path: str, directory in which to seek images
    :param minimum: int, minimum number of images required for the number of images to be plotted
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
    """
    Environment tests
    """
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
    else:
        print(">>> Development environment passes all tests!")


def filter_by_nms(preds_list: List[dict], iou_threshold: float) -> List[dict]:
    """
    Function to filter a bounding boxes predictions list by using non-maximum suppression

    :param preds_list: list, predicted bounding boxes
    :param iou_threshold: float, iou threshold for non-maximum suppression
    """
    keep_nms = [nms(pred['boxes'], pred['scores'], iou_threshold) for pred in preds_list]

    preds_nms = []

    for pred, keep in zip(preds_list, keep_nms):
        preds_nms.append({key: torch.index_select(val, dim=0, index=keep) for key, val in pred.items()})

    return preds_nms


def filter_by_score(preds_list: List[dict], score_threshold: float) -> List[dict]:
    """
    Function to filter a bounding boxes predictions list by using a confidence score threshold

    :param preds_list: list, predicted bounding boxes
    :param score_threshold: float, confidence score threshold above which predictions are to be saved
    :return:
    """
    preds_filt = []

    for pred in preds_list:
        keep = []

        for index, score in enumerate(pred['scores']):
            if score.greater(score_threshold):
                keep.append(index)

        preds_filt.append({key: torch.index_select(val, dim=0, index=torch.tensor(keep, device=pred['scores'].device))
                           for key, val in pred.items()})

    return preds_filt


def google_image_scraper(chrome_driver_path: str = 'C:/Users/simon.giard-leroux/Google Drive/'
                                                   'Maîtrise/Python/fuseFinder/chromedriver.exe',
                         prefix: str = 'English Electric C',
                         postfix: str = 'J') -> None:
    """
    Function to scrape images from Google images

    :param chrome_driver_path: str, chromedriver.exe path
    :param prefix: str, prefix for fuse search criterion
    :param postfix: str, postfix for fuse search criterion
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
    """
    Function to convert multiple json files into a single csv file

    :param dir_list: list, containing the directories in which to fetch the jsons
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


def print_dict(dictionary: dict, n_spaces: int, str_format: str = None) -> None:
    """

    :param dictionary:
    :param n_spaces:
    :param str_format:
    :return:
    """
    max_key_length = max(map(len, dictionary)) + n_spaces

    if str_format is None:
        for key, value in dictionary.items():
            print(f'{key}:{" " * (max_key_length - len(key))}{value}')
    else:
        for key, value in dictionary.items():
            print(f'{key}:{" " * (max_key_length - len(key))}{value:{str_format}}')


def rename_photos(root_dir: str = 'C:/Users/simon.giard-leroux/Google Drive/'
                                  'Maîtrise SGL CIMA+/General/Fuses Survey Dataset 2') -> None:
    """
    Rename all photos in a folder's subfolders

    :param root_dir: str, root directory
    """
    for subdir, dirs, files in os.walk(root_dir):
        for i, file in enumerate(files, start=1):
            os.rename(subdir + os.sep + file, subdir + "-" + str(i) + f".{IMAGE_EXT}")
