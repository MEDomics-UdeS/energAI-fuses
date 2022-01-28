"""
File:
    src/data/Datasets/FuseDataset.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Custom fuse dataset class.
"""

import torch
from tqdm import trange
import json
import ray
from typing import Tuple, List
from copy import deepcopy

from src.data.Datasets.CustomDataset import CustomDataset, ray_load_images


class FuseDataset(CustomDataset):
    """Custom fuse dataset class"""
    def __init__(self,
                 images_path: str,
                 images_filenames: List[str],
                 targets: str,
                 num_workers: int,
                 phase: str) -> None:
        """Class constructor

        Args:
            images_path(str): path to the images
            images_filenames(List[str]): list of individual file names for all images
            targets(str): targets file name
            num_workers(int): number of workers for multiprocessing
            phase(str): phase, 'Training' or 'Inference'

        """

        # Save the image paths as an object attribute
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
                self._targets = []
        else:
            # Required for SplittingManager maybe
            self._targets = deepcopy(targets)

        # Get the dataset size
        size = len(self._image_paths)

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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:
        """Class __getitem__ method, called when object[index] is used

        Args:
            index(int): actual index to get

        Returns:
            Tuple[torch.Tensor,dict]: transformed current image and current targets

        """
        # When working with small batches
        if self._targets:        
            return self.transforms(self._images[index]), self._targets[index]
        else:
            return self.transforms(self._images[index]), {}

    @property
    def targets(self):
        """ """
        return self._targets
