
from src.data.Datasets.CustomDataset import CustomDataset
from PIL import Image
from typing import Tuple, List
import torch


class CocoDataset(CustomDataset):
    """ """
    
    def __init__(self,
                 image,
                 target,
                 path) -> None:
        """Class constructor

        Args:
            image: 
            target: 
            path: 

        """
        self._images = [image]
        self._targets = [target]
        self._image_paths = [path]
    
    @property
    def targets(self):
        """ """
        return self._targets

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:
        """Class __getitem__ method, called when object[index] is used

        Args:
            index(int): actual index to get

        Returns:
            Tuple[torch.Tensor,dict]: transformed current image and current targets

        """
        return self._images[index], self._targets[index]

    def extract_data(self, index_list: List[int]) -> Tuple[List[str], List[Image.Image], List[dict]]:
        """Extract data from the object

        Args:
            index_list(List[int]): indices to extract

        Returns:
            Tuple[List[str],List[Image.Image],List[dict]]: extracted elements

        """
        # Sort and reverse the index list
        index_list = sorted(index_list, reverse=True)

        # Declare empty lists for the extracted elements
        image_paths = []
        images = []

        # Loop through the index list
        for index in index_list:
            # Pop the elements from the object and append to the extracted elements' lists
            image_paths.append(self._image_paths.pop(index))
            images.append(self._images.pop(index))

        # Return the extracted elements
        return image_paths, images

    def add_data(self, image_paths: List[str], images: List[Image.Image]) -> None:
        """Add data to the object

        Args:
            image_paths(List[str]): strings of image paths
            images(List[Image.Image]): PIL Images

        """
        # Add the data in arguments to the object attributes
        self._image_paths.extend(image_paths)
        self._images.extend(images)
