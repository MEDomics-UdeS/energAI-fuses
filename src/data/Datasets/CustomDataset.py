"""
File:
    src/data/Datasets/CustomDataset.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Generate custom dataset
"""

from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset

from PIL import Image
from typing import List, Tuple
import ray


class CustomDataset(ABC, Dataset):
    """ """
    def __len__(self) -> int:
        """A method to get the number of images in a dataset

        Returns:
          int: The number of images in the dataset

        """
        return len(self._images)

    def load_image(self, index: int) -> Image:
        """Load an image as a PIL Image object

        Args:
          index(int): image index

        Returns:
          Image: PIL Image

        """
        image_path = self._image_paths[index]
        img = Image.open(image_path)
        return img

    @property
    def images(self):
        """ """
        return self._images

    @property
    def image_paths(self):
        """ """
        return self._image_paths

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:
        pass


@ray.remote
def ray_load_images(image_paths: List[str], index: int) -> Tuple[Image.Image, int]:
    """Ray remote function to parallelize the loading of PIL Images to RAM

    Args:
        index(int): current index

    Returns:
        Tuple[Image.Image, int]: PIL Image and current index

    """
    return Image.open(image_paths[index]), index
