import torch
from torch.utils.data import Dataset

import os
import numpy as np
from tqdm import trange
from torchvision import transforms
from src.utils.constants import MEAN, STD
from src.data.DatasetManager import ray_get_rgb
from src.data.FuseDataset import ray_load_images

from PIL import Image
import ray
from typing import Tuple, List, Optional


class GuiDataset(Dataset):
    """
    Custom GUI dataset class
    """

    def __init__(self,
                 images_path: str = None,
                 num_workers: int = None,
                 norm: str= 'none') -> None:

        """
        Class constructor

        :param images_path: str, path to the images
        :param num_workers: int, number of workers for multiprocessing
        """
        # Check if images_path has been specified

        if norm == 'precalculated':
            # Use precalculated mean and standard deviation
            mean, std = MEAN, STD
        elif norm == 'calculated':
            # Recalculate mean and standard deviation
            mean, std = self.__calculate_mean_std(num_workers)
        elif norm == 'none':
            mean, std = None, None

        self.transforms = self.__transforms_base(mean, std)
        
        if images_path is not None:
            # Initialize ray
            ray.init(include_dashboard=False)

            # Get all survey images paths and ignore the .gitkeep file
            images = [img for img in sorted(os.listdir(
                images_path)) if img.startswith('S') or img.startswith('G')]

            # Save the image paths as an object attribute
            self.__image_paths = [os.path.join(
                images_path, img) for img in images]

            # Get the dataset size
            size = len(self.__image_paths)

            # Declare empty list to save all images in RAM
            self.__images = [None] * size

            # Get ray workers IDs for varying size of dataset
            if num_workers > size:
                ids = [ray_load_images.remote(self.__image_paths, i)
                       for i in range(size)]
            else:
                ids = [ray_load_images.remote(self.__image_paths, i)
                       for i in range(num_workers)]

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
                    ids.extend([ray_load_images.remote(
                        self.__image_paths, size - nb_job_left)])

                    # Decrease number of jobs left
                    nb_job_left -= 1

            # Shutdown ray
            ray.shutdown()
        else:
            # Specify blank image_paths and images lists
            self.__image_paths = []
            self.__images = []


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:
        """
        Class __getitem__ method, called when object[index] is used

        :param index: int, actual index to get
        :return: tuple, transformed current image and current targets
        """

        return self.transforms(self.__images[index]), {}

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

        # Loop through the index list
        for index in index_list:
            # Pop the elements from the object and append to the extracted elements' lists
            image_paths.append(self.__image_paths.pop(index))
            images.append(self.__images.pop(index))

        # Return the extracted elements
        return image_paths, images

    def add_data(self, image_paths: List[str], images: List[Image.Image]) -> None:
        """
        Add data to the object

        :param image_paths: list, strings of image paths
        :param images: list, PIL Images
        :param targets: list, targets dictionaries
        """
        # Add the data in arguments to the object attributes
        self.__image_paths.extend(image_paths)
        self.__images.extend(images)


    @staticmethod
    def __transforms_base(mean: Optional[Tuple[float, float, float]],
                        std: Optional[Tuple[float, float, float]]) -> transforms.Compose:
        """
            Method to construct the validation and testing datasets transforms

            :param mean: tuple of 3 floats, containing the mean (R, G, B) values
            :param std: tuple of 3 floats, containing the standard deviation of (R, G, B) values
            :return: transforms.Compose, custom composed transforms list
            """
        transforms_list = [
            # Convert to tensor
                transforms.ToTensor(),
            ]

        # Normalize by mean, std
        if mean is not None:
                transforms_list.append(transforms.Normalize(mean, std))

        # Return a composed transforms list
        return transforms.Compose(transforms_list)

    def __calculate_mean_std(self, num_workers: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Method to calculate the mean and standard deviation for each channel (R, G, B) for each image

        :param num_workers: int, number of workers for multiprocessing
        :return: tuple, containing the mean and std deviation values for each channel (R, G, B)
        """
        # Initialize ray
        ray.init(include_dashboard=False)

        # Get ray workers IDs
        ids = [ray_get_rgb.remote(self.__dataset_train.image_paths, i)
               for i in range(num_workers)]

        # Get dataset size
        size = len(self.__dataset_train.image_paths)

        # Declare lists to store R, G, B values
        r = [None] * size
        g = [None] * size
        b = [None] * size

        # Calculate initial number of jobs left
        nb_job_left = size - num_workers

        # Multiprocessing loop
        for _ in trange(size, desc='Getting RGB values of each pixel...'):
            # Handle ray workers ready states and IDs
            ready, ids = ray.wait(ids, num_returns=1)

            # Get R, G, B values and index
            r_val, g_val, b_val, idx = ray.get(ready)[0]

            # Saving values to the values lists
            r[idx] = r_val
            g[idx] = g_val
            b[idx] = b_val

            # Check if there are jobs left
            if nb_job_left > 0:
                # Assign workers to the remaining tasks
                ids.extend(
                    [ray_get_rgb.remote(self.__dataset_train.image_paths, size - nb_job_left)])

                # Decreasing the number of jobs left
                nb_job_left -= 1

        # Shutting down ray
        ray.shutdown()

        # Converting the R, G, B lists to numpy arrays
        r = np.array(r)
        g = np.array(g)
        b = np.array(b)

        # Calculating the mean values per channel
        print('Calculating RGB means...')
        mean = (np.mean(r) / 255, np.mean(g) / 255, np.mean(b) / 255)

        # Calculating the standard deviation values per channel
        print('Calculating RGB standard deviations...')
        std = (np.std(r) / 255, np.std(g) / 255, np.std(b) / 255)

        # Returning the mean and standard deviation per channel
        return mean, std
