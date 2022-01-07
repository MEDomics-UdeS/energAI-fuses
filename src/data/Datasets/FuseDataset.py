"""
File:
    src/data/FuseDataset.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Custom fuse dataset class.
"""

import torch
import os
from tqdm import trange
import json
from PIL import Image
import ray
from typing import Tuple, List
from copy import deepcopy

from src.data.Datasets.CustomDataset import CustomDataset, ray_load_images
from src.utils.constants import RESIZED_LEARNING_PATH


class FuseDataset(CustomDataset):
    """
    Custom fuse dataset class
    """
    def __init__(self,
                 images_path: str,
                 images_filenames: str,
                 targets: str,
                 num_workers: int,
                 phase: str) -> None: #,
                 # google_images: bool = True,
                 # load_to_ram: bool = True) -> None:
        """
        Class constructor

        :param images_path: str, path to the images
        :param targets_path: str, path to the targets json file
        :param num_workers: int, number of workers for multiprocessing
        """
        # Check if images_path has been specified
        # if images_path is not None:

        # Get all survey images paths and ignore the .gitkeep file
        # images = [img for img in sorted(os.listdir(images_path)) if img.startswith('.') is False]
        #
        # if not google_images:
        #     google_imgs = [image for image in images if image.startswith('G')]
        #     google_indices = [images.index(google_image) for google_image in google_imgs]
        #     images = [e for i, e in enumerate(images) if i not in google_indices]
        #
        # Save the image paths as an object attribute
        # self._image_paths = [os.path.join(images_path, img) for img in images]
        self._image_paths = [images_path + image_path for image_path in images_filenames]
        
        # Check if targets_path has been specified
        if phase == "Inference":
            # Only used for the GUI application
            if targets is not None:
                # If we have targets bbox, the order of inference images should be kept
                self._image_paths = sorted(self._image_paths)

                # Load the targets json into the targets attribute in the object
                self._targets = json.load(open(targets))
            
                # Convert the targets to tensors
                for target in self._targets:
                    for key, value in target.items():
                        target[key] = torch.as_tensor(value, dtype=torch.int64) 
        else:
            # Required for SplittingManager maybe
            self._targets = deepcopy(targets)

        # Get the dataset size
        size = len(self._image_paths)

        # if load_to_ram:
        # Initialize ray
        ray.init(include_dashboard=False)

        # Declare empty list to save all images in RAM
        self._images = [None] * size

        # Get ray workers IDs for varying size of dataset and num_workers
        if size < num_workers:
            ids = [ray_load_images.remote(
                self._image_paths, i) for i in range(size)]
        else:
            ids = [ray_load_images.remote(
                self._image_paths, i) for i in range(num_workers)]

        # Calculate initial number of jobs left
        nb_job_left = size - num_workers

        # Ray multiprocessing loop
        for _ in trange(size, desc=f'Loading {phase} Images to RAM', leave=False):
            # Get ready status and IDs of ray workers
            ready, ids = ray.wait(ids, num_returns=1)

            # Get current image and index
            image, idx = ray.get(ready)[0]

            # Save current image to the images list
            self._images[idx] = image

            # Check if there are jobs left
            if nb_job_left > 0:
                # Assign workers to the remaining tasks
                ids.extend([ray_load_images.remote(self._image_paths, size - nb_job_left)])

                # Decrease number of jobs left
                nb_job_left -= 1

        # Shutdown ray
        ray.shutdown()
        # else:
        #     # Specify blank image_paths and images lists
        #     self._image_paths = []
        #     self._images = []

        # Check if targets_path has been specified
        # if targets_path is not None:
        #     # Load the targets json into the targets attribute in the object
        #     self._targets = json.load(open(targets_path))
        #
        #     if not google_images:
        #         self._targets = [e for i, e in enumerate(self._targets) if i not in google_indices]
        #
        #     # Convert the targets to tensors
        #     for target in self._targets:
        #         for key, value in target.items():
        #             target[key] = torch.as_tensor(value, dtype=torch.int64)
        # else:
        #     # Declare empty targets list
        #     self._targets = []

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:
        """
        Class __getitem__ method, called when object[index] is used

        :param index: int, actual index to get
        :return: tuple, transformed current image and current targets
        """
        # When working with small batches
        if self._targets:        
            return self.transforms(self._images[index]), self._targets[index]
        else:
            return self.transforms(self._images[index]), {}

    @property
    def targets(self):
        return self._targets

    # def extract_data(self, index_list: List[int]) -> Tuple[List[str], List[Image.Image], List[dict]]:
    #     """
    #     Extract data from the object
    #
    #     :param index_list: list, indices to extract
    #     :return: tuple, extracted elements
    #     """
    #     # Sort and reverse the index list
    #     index_list = sorted(index_list, reverse=True)
    #
    #     # Declare empty lists for the extracted elements
    #     image_paths = []
    #     images = []
    #     targets = []
    #
    #     # Loop through the index list
    #     for index in index_list:
    #         # Pop the elements from the object and append to the extracted elements' lists
    #         image_paths.append(self._image_paths.pop(index))
    #         images.append(self._images.pop(index))
    #         targets.append(self._targets.pop(index))
    #
    #     # Return the extracted elements
    #     return image_paths, images, targets
    #
    # def add_data(self, image_paths: List[str], images: List[Image.Image], targets: List[dict]) -> None:
    #     """
    #     Add data to the object
    #
    #     :param image_paths: list, strings of image paths
    #     :param images: list, PIL Images
    #     :param targets: list, targets dictionaries
    #     """
    #     # Add the data in arguments to the object attributes
    #     self._image_paths.extend(image_paths)
    #     self._images.extend(images)
    #     self._targets.extend(targets)
