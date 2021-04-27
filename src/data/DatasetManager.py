"""
File:
    src/data/DatasetManager.py

Authors:
    - Simon Giard-Leroux
    - Shreyas Sunil Kulkarni

Description:
    Generate datasets for training, validation and testing.
"""

import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import ray
import requests
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm import tqdm
from tqdm import trange
from typing import Tuple, List

from src.utils.constants import *
from src.data.FuseDataset import FuseDataset


class DatasetManager:
    """
    Dataset Manager class, handles the creation of the training, validation and testing datasets.
    """
    def __init__(self,
                 images_path: str,
                 annotations_path: str,
                 max_image_size: int,
                 num_workers: int,
                 data_aug: float,
                 validation_size: float,
                 test_size: float,
                 mean_std: bool) -> None:

        if max_image_size > 0:
            image_ext = '.JPG'

            if any(file.endswith(image_ext) for file in os.listdir(RESIZED_PATH)):
                for file in os.listdir(RESIZED_PATH):
                    if file.endswith(image_ext):
                        img = Image.open(f'{RESIZED_PATH}{file}')
                        break

                if img.size != (max_image_size, max_image_size):
                    print(f'Max image size argument is {(max_image_size, max_image_size)} '
                          f'but a resized image of {img.size} was found')
                    print(f'All images will be resized to {(max_image_size, max_image_size)}')

                    self.resize_images(max_image_size, num_workers)
            else:
                if any(file.endswith(image_ext) for file in os.listdir(RAW_PATH)):
                    self.resize_images(max_image_size, num_workers)
                else:
                    if input('Raw data folder contains no images. Do you want to download them? (~ 12 GB) (y/n): ') == 'y':
                        self.fetch_data(IMAGES_ID, ANNOTATIONS_ID)
                        self.resize_images(max_image_size, num_workers)
                    else:
                        sys.exit(1)

        # Declare training, validation and testing datasets
        self.dataset_train = FuseDataset(images_path, annotations_path, num_workers)
        self.dataset_valid = FuseDataset()
        self.dataset_test = FuseDataset()

        total_size = len(self.dataset_train)

        self.dataset_train, self.dataset_valid = self.split_dataset(self.dataset_train, self.dataset_valid,
                                                                    validation_size, total_size)
        self.dataset_train, self.dataset_test = self.split_dataset(self.dataset_train, self.dataset_test,
                                                                   test_size, total_size)

        if mean_std:
            mean, std = self.calculate_mean_std(num_workers)
        else:
            mean, std = MEAN, STD

        self.dataset_train.transforms = self.transforms_train(mean, std, data_aug)
        self.dataset_valid.transforms = self.transforms_base(mean, std)
        self.dataset_test.transforms = self.transforms_base(mean, std)

    @staticmethod
    def download_file_from_google_drive(file_id: str, dest: str, chunk_size: int = 32768) -> None:
        """

        Inspired from :
        https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

        :param file_id:
        :param dest:
        :param chunk_size:
        :return:
        """
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={'id': file_id}, stream=True)

        token = None

        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        if response.ok:
            with open(dest, "wb") as f:
                for chunk in tqdm(response.iter_content(chunk_size), desc='Downloading'):
                    if chunk:
                        f.write(chunk)
        else:
            raise Exception(f'Error {response.status_code}: {response.reason}')

    def fetch_data(self, images_id: str, annotations_id: str) -> None:
        images_zip = os.path.join(RAW_PATH, 'images.zip')

        print('\nDownloading images to:\t\t', RAW_PATH)
        self.download_file_from_google_drive(images_id, images_zip)
        print('Done!\nUnzipping images...')

        with zipfile.ZipFile(images_zip, 'r') as zip_ref:
            zip_ref.extractall(RAW_PATH)

        os.remove(images_zip)

        print('Done!\n\nDownloading annotations to:\t', ANNOTATIONS_PATH)
        self.download_file_from_google_drive(annotations_id, ANNOTATIONS_PATH)
        print('Done!')

    @staticmethod
    def transforms_train(mean: float, std: float, data_aug: float) -> transforms.Compose:
        transforms_list = [
            transforms.ColorJitter(brightness=data_aug,
                                   contrast=data_aug,
                                   saturation=data_aug,
                                   hue=data_aug),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

        return transforms.Compose(transforms_list)

    @staticmethod
    def transforms_base(mean: float, std: float) -> transforms.Compose:
        transforms_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

        return transforms.Compose(transforms_list)

    @staticmethod
    def split_dataset(dataset_in: FuseDataset, dataset_out: FuseDataset, split: float, total_size: int) \
            -> Tuple[FuseDataset, FuseDataset]:
        dataset_size = len(dataset_in)
        indices = list(range(dataset_size))
        split_idx = int(np.floor(split * total_size))
        np.random.shuffle(indices)
        indices = indices[0:split_idx]

        image_paths, images, targets = dataset_in.extract_data(index_list=indices)
        dataset_out.add_data(image_paths, images, targets)

        return dataset_in, dataset_out

    def calculate_mean_std(self, num_workers: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        # Calculate dataset mean & std for normalization
        ray.init(include_dashboard=False)

        image_paths = self.dataset_train.image_paths

        ids = [ray_get_rgb.remote(image_paths, i) for i in range(num_workers)]
        size = len(image_paths)
        r = [None] * size
        g = [None] * size
        b = [None] * size
        nb_job_left = size - num_workers

        for _ in trange(size, desc='Getting RGB values of each pixel...'):
            ready, ids = ray.wait(ids, num_returns=1)
            r_val, g_val, b_val, idx = ray.get(ready)[0]
            r[idx] = r_val
            g[idx] = g_val
            b[idx] = b_val

            if nb_job_left > 0:
                ids.extend([ray_get_rgb.remote(image_paths, size - nb_job_left)])
                nb_job_left -= 1

        ray.shutdown()

        r = np.array(r)
        g = np.array(g)
        b = np.array(b)

        print('Calculating RGB means...')
        mean = (np.mean(r) / 255, np.mean(g) / 255, np.mean(b) / 255)

        print('Calculating RGB standard deviations...')
        std = (np.std(r) / 255, np.std(g) / 255, np.std(b) / 255)

        return mean, std

    @staticmethod
    def resize_images(max_image_size: int, num_workers: int) -> None:
        ray.init(include_dashboard=False)

        imgs = [img for img in sorted(os.listdir(RAW_PATH)) if img.startswith('.') is False]
        image_paths = [os.path.join(RAW_PATH, img) for img in imgs]
        annotations = pd.read_csv(ANNOTATIONS_PATH)

        size = len(image_paths)
        resize_ratios = [None] * size
        targets_list = [None] * size

        ids = [ray_resize_images.remote(image_paths, max_image_size, annotations, i) for i in range(num_workers)]

        nb_job_left = size - num_workers

        for _ in trange(size, desc='Resizing images...'):
            ready, ids = ray.wait(ids, num_returns=1)
            resize_ratio, idx, box_list, targets = ray.get(ready)[0]
            resize_ratios[idx] = resize_ratio
            targets_list[idx] = targets

            if nb_job_left > 0:
                ids.extend([ray_resize_images.remote(image_paths, max_image_size, annotations, size - nb_job_left)])
                nb_job_left -= 1

        ray.shutdown()

        average_ratio = sum(resize_ratios) / len(resize_ratios)
        print(f'\nAverage resize ratio: {average_ratio:.2%}')
        print(f'Maximum resize ratio: {max(resize_ratios):.2%}')
        print(f'Minimum resize ratio: {min(resize_ratios):.2%}')

        json.dump(targets_list, open('data/annotations/targets_resized.json', 'w'), ensure_ascii=False)
        print('\nResized images have been saved to:\t\tdata/resized/')
        print('Resized targets have been saved to:\t\tdata/annotations/targets_resized.json')


@ray.remote
def ray_resize_images(image_paths: List[str], max_image_size: int, annotations: pd.DataFrame,
                      idx: int, show_bounding_boxes: bool = False) -> Tuple[float, int, np.array, dict]:
    f = image_paths[idx].rsplit('/', 1)[-1].split(".")
    func = lambda x: x.split(".")[0]

    box_array = annotations.loc[annotations["filename"].apply(func) == f[0]][["xmin", "ymin", "xmax", "ymax"]].values
    label_array = annotations.loc[annotations["filename"].apply(func) == f[0]][["label"]].values

    label_list = []

    for label in label_array:
        label_list.append(CLASS_DICT[str(label[0])])

    num_boxes = len(box_array)

    img = Image.open(image_paths[idx]).convert("RGB")
    original_size = img.size

    img2 = Image.new('RGB', (max_image_size, max_image_size), (255, 255, 255))

    resize_ratio = (img2.size[0] * img2.size[1]) / (original_size[0] * original_size[1])

    if max_image_size < original_size[0] or max_image_size < original_size[1]:
        img.thumbnail((max_image_size, max_image_size),
                      resample=Image.BILINEAR,
                      reducing_gap=2)

        downsize_ratio = img.size[0] / original_size[0]
    else:
        downsize_ratio = 1

    x_offset = int((max_image_size - img.size[0]) / 2)
    y_offset = int((max_image_size - img.size[1]) / 2)
    img2.paste(img, (x_offset, y_offset, x_offset + img.size[0], y_offset + img.size[1]))

    if show_bounding_boxes:
        draw = ImageDraw.Draw(img2)

    for i in range(num_boxes):
        for j in range(4):
            box_array[i][j] = int(box_array[i][j] * downsize_ratio)

            if j == 0 or j == 2:
                box_array[i][j] += x_offset
            else:
                box_array[i][j] += y_offset

        if show_bounding_boxes:
            draw.rectangle([(box_array[i][0], box_array[i][1]), (box_array[i][2], box_array[i][3])],
                           outline="red", width=5)

    img2.save(f'data/resized/{image_paths[idx].split("/")[-1]}')

    area = [int(a) for a in list((box_array[:, 3] - box_array[:, 1]) * (box_array[:, 2] - box_array[:, 0]))]

    targets = {"boxes": list(box_array.tolist()),
               "labels": label_list,
               "image_id": idx,
               "area": area,
               "iscrowd": [0] * num_boxes}

    return resize_ratio, idx, box_array, targets


@ray.remote
def ray_get_rgb(image_paths: List[str], idx: int) -> Tuple[float, float, float, int]:
    image = Image.open(image_paths[idx]).convert("RGB")

    r = np.dstack(np.array(image)[:, :, 0])
    g = np.dstack(np.array(image)[:, :, 1])
    b = np.dstack(np.array(image)[:, :, 2])

    return r, g, b, idx
