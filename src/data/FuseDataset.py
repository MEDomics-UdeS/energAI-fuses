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
from torch.utils.data import Dataset
import os
from tqdm import trange
import json
from PIL import Image
import ray
from typing import Tuple, List


class FuseDataset(Dataset):
    """
    Custom fuse dataset class
    """
    def __init__(self,
                 images_path: str = None,
                 targets_path: str = None,
                 num_workers: int = None,
                 google_images: bool = True) -> None:
        """
        Class constructor

        :param images_path: str, path to the images
        :param targets_path: str, path to the targets json file
        :param num_workers: int, number of workers for multiprocessing
        """
        # Check if images_path has been specified
        if images_path is not None:
            # Initialize ray
            ray.init(include_dashboard=False)

            # Get all survey images paths and ignore the .gitkeep file
            images = [img for img in sorted(os.listdir(images_path)) if img.startswith('S') or img.startswith('G')]

            if not google_images:
                google_imgs = [image for image in images if image.startswith('G')]
                google_indices = [images.index(google_image) for google_image in google_imgs]
                images = [e for i, e in enumerate(images) if i not in google_indices]

            # Save the image paths as an object attribute
            self.__image_paths = [os.path.join(images_path, img) for img in images]

            # Get the dataset size
            size = len(self.__image_paths)

            # Declare empty list to save all images in RAM
            self.__images = [None] * size

            # Get ray workers IDs
            ids = [ray_load_images.remote(self.__image_paths, i) for i in range(num_workers)]

            # Calculate initial number of jobs left
            nb_job_left = size - num_workers

            # Ray multiprocessing loop
            for _ in trange(size, desc='Loading images to RAM', leave=False):
                # Get ready status and IDs of ray workers
                ready, ids = ray.wait(ids, num_returns=1)

                # Get current image and index
                image, idx = ray.get(ready)[0]

                # Save current image to the images list
                self.__images[idx] = image

                # Check if there are jobs left
                if nb_job_left > 0:
                    # Assign workers to the remaining tasks
                    ids.extend([ray_load_images.remote(self.__image_paths, size - nb_job_left)])

                    # Decrease number of jobs left
                    nb_job_left -= 1

            # Shutdown ray
            ray.shutdown()
        else:
            # Specify blank image_paths and images lists
            self.__image_paths = []
            self.__images = []

        # Check if targets_path has been specified
        if targets_path is not None:
            # Load the targets json into the targets attribute in the object
            self.__targets = json.load(open(targets_path))

            if not google_images:
                self.__targets = [e for i, e in enumerate(self.__targets) if i not in google_indices]

            # Convert the targets to tensors
            for target in self.__targets:
                for key, value in target.items():
                    target[key] = torch.as_tensor(value, dtype=torch.int64)
        else:
            # Declare empty targets list
            self.__targets = []

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:
        """
        Class __getitem__ method, called when object[index] is used

        :param index: int, actual index to get
        :return: tuple, transformed current image and current targets
        """
        return self.transforms(self.__images[index]), self.__targets[index]

    def __len__(self) -> int:
        """
        Class __len__ method, called when len(object) is used

        :return: int, number of images in the dataset
        """
        return len(self.__images)

    @property
    def images(self):
        return self.__images

    @property
    def image_paths(self):
        return self.__image_paths

    @property
    def targets(self):
        return self.__targets

    def load_image(self, index: int) -> Image:
        """
        Load an image as a PIL Image object
        :param index: int, image index
        :return: PIL Image
        """
        image_path = self.__image_paths[index]
        img = Image.open(image_path)
        return img

    def extract_data(self, index_list: List[int]) -> Tuple[List[str], List[Image.Image], List[dict]]:
        """
        Extract data from the object

        :param index_list: list, indices to extract
        :return: tuple, extracted elements
        """
        # Sort and reverse the index list
        index_list = sorted(index_list, reverse=True)

        # Declare empty lists for the extracted elements
        image_paths = []
        images = []
        targets = []

        # Loop through the index list
        for index in index_list:
            # Pop the elements from the object and append to the extracted elements' lists
            image_paths.append(self.__image_paths.pop(index))
            images.append(self.__images.pop(index))
            targets.append(self.__targets.pop(index))

        # Return the extracted elements
        return image_paths, images, targets

    def add_data(self, image_paths: List[str], images: List[Image.Image], targets: List[dict]) -> None:
        """
        Add data to the object

        :param image_paths: list, strings of image paths
        :param images: list, PIL Images
        :param targets: list, targets dictionaries
        """
        # Add the data in arguments to the object attributes
        self.__image_paths.extend(image_paths)
        self.__images.extend(images)
        self.__targets.extend(targets)


@ray.remote
def ray_load_images(image_paths: List[str], index: int) -> Tuple[Image.Image, int]:
    """
    Ray remote function to parallelize the loading of PIL Images to RAM

    :param image_paths: list, strings of image paths
    :param index: int, current index
    :return: tuple, PIL Image and current index
    """
    return Image.open(image_paths[index]), index
