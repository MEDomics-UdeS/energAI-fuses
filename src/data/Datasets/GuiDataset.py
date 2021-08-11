import torch
import os

from tqdm import trange
from src.utils.constants import GUI_RESIZED_PATH
from src.data.Datasets.CustomDataset import ray_load_images

from PIL import Image
import ray
from typing import Tuple, List
from src.data.Datasets.CustomDataset import CustomDataset
import json

class GuiDataset(CustomDataset):
    """
    Custom GUI dataset class
    """
    def __init__(self,
                 images_path: str = None,
                 targets_path: str = None,
                 num_workers: int = None) -> None:
        """
        Class constructor

        :param images_path: str, path to the images
        :param num_workers: int, number of workers for multiprocessing
        """
        if images_path is not None:
            # Initialize ray
            ray.init(include_dashboard=False)

            # Get all the images paths 
            images = [img for img in sorted(os.listdir(GUI_RESIZED_PATH)) if img.startswith('.') is False]

            # Save the image paths as an object attribute
            self._image_paths = [os.path.join(GUI_RESIZED_PATH, img) for img in images]

            # Get the dataset size
            size = len(self._image_paths)

            # Declare empty list to save all images in RAM
            self._images = [None] * size

            # Get ray workers IDs for varying size of dataset and num_workers
            if size < num_workers:
                ids = [ray_load_images.remote(self._image_paths, i) for i in range(size)]
            else:
                ids = [ray_load_images.remote(self._image_paths, i) for i in range(num_workers)]

            # Calculate initial number of jobs left
            nb_job_left = size - num_workers

            # Ray multiprocessing loop
            for _ in trange(size, desc='Loading images to RAM', leave=False):
                # Get ready status and IDs of ray workers
                ready, ids = ray.wait(ids, num_returns=1)

                # Get current image and index
                image, idx = ray.get(ready)[0]

                # Save current image to the images list
                self._images[idx] = image

                # Check if there are jobs left
                if nb_job_left > 0:
                    # Assign workers to the remaining tasks
                    ids.extend([ray_load_images.remote(
                        self._image_paths, size - nb_job_left)])

                    # Decrease number of jobs left
                    nb_job_left -= 1

            # Shutdown ray
            ray.shutdown()
        else:
            # Specify blank image_paths and images lists
            self._image_paths = []
            self._images = []
        
        # Check if targets_path has been specified
        if targets_path is not None:
            # Load the targets json into the targets attribute in the object
            self._targets = json.load(open(targets_path))

            # Convert the targets to tensors
            for target in self._targets:
                for key, value in target.items():
                    target[key] = torch.as_tensor(value, dtype=torch.int64)
        else:
            # Declare empty targets list
            self._targets = []


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
            image_paths.append(self._image_paths.pop(index))
            images.append(self._images.pop(index))
            targets.append(self._targets.pop(index))

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
        self._image_paths.extend(image_paths)
        self._images.extend(images)
        self._targets.extend(targets)
