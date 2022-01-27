"""
File:
    src/data/DatasetManagers/CustomDatasetManager.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Contains the CustomDatasetManager, parent DatasetManager for all children classes.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
from torchvision import transforms
import numpy as np
import ray
from PIL import Image, ImageDraw
import pandas as pd

from src.utils.constants import CLASS_DICT
from src.utils.helper_functions import cp_split


class CustomDatasetManager(ABC):
    """Parent class for all DatasetManager child classes."""
    @staticmethod
    def _transforms_base(mean: Optional[Tuple[float, float, float]],
                         std: Optional[Tuple[float, float, float]]) -> transforms.Compose:
        """Method to construct the validation and testing datasets transforms

        Args:
            mean(Optional[Tuple[float, float, float]]): tuple of 3 floats, containing the mean (R, G, B) values
            std(Optional[Tuple[float, float, float]]): tuple of 3 floats, containing the standard deviation of (R, G, B) values

        Returns:
            transforms.Compose: custom composed transforms list

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

    @staticmethod
    @abstractmethod
    def _resize_images(image_size: int, num_workers: int) -> None:
        """

        Args:
            image_size(int): 
            num_workers(int): 

        """
        pass


@ray.remote
def ray_resize_images(image_paths: List[str],
                      destination_path: str,
                      image_size: int,
                      annotations_csv: str,
                      idx: int,
                      show_bounding_boxes: bool = False) -> Tuple[float, int, np.array, dict]:
    """Ray remote function to parallelize the resizing of images

    Args:
        image_paths(List[str]): contains image paths
        destination_path(str): destination file path
        image_size(int): image size to resize all images to (height & width)
        annotations_csv(str): file path to csv file containing bounding box annotations
        idx(int): current index
        show_bounding_boxes(bool, optional): if True, ground truth bounding boxes are drawn on the resized images
                                             (used to test if bounding boxes are properly resized) (Default value = False)

    Returns:
        Tuple[float,int,np.array,dict]: resize_ratio, idx, box_array, targets to continue ray paralleling

    """
    # Convert the annotations csv file to a pandas DataFrame
    if annotations_csv:
        annotations = pd.read_csv(annotations_csv)
    
    # Get the current image name, without the file path and without the file extension
    f = cp_split(image_paths[idx])[-1].split(".")[0]

    # Create the box and label array if annotations are provided
    if annotations_csv:
        # Get bounding boxes coordinates for the current image
        box_array = annotations.loc[annotations["filename"] == f][[
            "xmin", "ymin", "xmax", "ymax"]].values

        # Get class labels (str) for the current image
        label_array = annotations.loc[annotations["filename"] == f][[
            "label"]].values

        # Declare empty label list
        label_list = []

        # Get class labels (index) for the current image
        for label in label_array:
            label_list.append(CLASS_DICT[str(label[0])])

        # Get number of bounding boxes
        num_boxes = len(box_array)

    # Open the current image
    img = Image.open(image_paths[idx])

    # Get the current image size
    original_size = img.size

    # Create a new blank white image of size (image_size, image_size)
    img2 = Image.new('RGB', (image_size, image_size), (255, 255, 255))

    # Calculate the resize ratio
    resize_ratio = (img2.size[0] * img2.size[1]) / \
        (original_size[0] * original_size[1])

    # Check if the original size is larger than the maximum image size
    if image_size < original_size[0] or image_size < original_size[1]:
        # Downsize the image using the thumbnail method
        img.thumbnail((image_size, image_size),
                      resample=Image.BILINEAR,
                      reducing_gap=2)

        # Calculate the downsize ratio
        downsize_ratio = img.size[0] / original_size[0]
    else:
        downsize_ratio = 1

    # Calculate the x and y offsets at which the downsized image needs to be pasted (to center it)
    x_offset = int((image_size - img.size[0]) / 2)
    y_offset = int((image_size - img.size[1]) / 2)

    # Paste the downsized original image in the new (image_size, image_size) image
    img2.paste(img, (x_offset, y_offset, x_offset +
               img.size[0], y_offset + img.size[1]))

    # Declare an ImageDraw object if the show_bounding_boxes argument is set to True
    if show_bounding_boxes:
        draw = ImageDraw.Draw(img2)

    # Save the resized image
    img2.save(f'{destination_path}{image_paths[idx].split("/")[-1]}')

    if annotations_csv:
        # Loop through each bounding box
        for i in range(num_boxes):
            # Loop through each of the 4 coordinates (x_min, y_min, x_max, y_max)
            for j in range(4):
                # Apply a downsize ratio to the bounding boxes
                box_array[i][j] = int(box_array[i][j] * downsize_ratio)

                # Apply an offset to the bounding boxes
                if j == 0 or j == 2:
                    box_array[i][j] += x_offset
                else:
                    box_array[i][j] += y_offset

            # Draw the current ground truth bounding box if the show_bounding_boxes argument is set to True
            if show_bounding_boxes:
                draw.rectangle([(box_array[i][0], box_array[i][1]), (box_array[i][2], box_array[i][3])],
                               outline="red", width=5)

        # Calculate the area of each bounding box
        area = [int(a) for a in list((box_array[:, 3] - box_array[:, 1])
                * (box_array[:, 2] - box_array[:, 0]))]

        # Save ground truth targets to a dictionary
        targets = {"boxes": list(box_array.tolist()),
                   "labels": label_list,
                   "image_id": idx,
                   "area": area,
                   "iscrowd": [0] * num_boxes}

        # Return the objects required for ray parallelization
        return resize_ratio, idx, box_array, targets
    
    # Return the objects required for ray parallelization
    return resize_ratio, idx


@ray.remote
def ray_get_rgb(image_paths: List[str],
                idx: int) -> Tuple[np.array, np.array, np.array, int]:
    """Ray remote function to parallelize the extraction of R, G, B values from images

    Args:
        image_paths(List[str]): contains the image paths
        idx(int): current index

    Returns:
        Tuple[np.array,np.array,np.array,int]: (R, G, B) values numpy arrays for the current image and the current index

    """
    # Open the current image
    image = Image.open(image_paths[idx])

    # Get the values of each pixel in the R, G, B channels
    r = np.dstack(np.array(image)[:, :, 0])
    g = np.dstack(np.array(image)[:, :, 1])
    b = np.dstack(np.array(image)[:, :, 2])

    # Return the r, g, b values numpy arrays for the current image and the current index
    return r, g, b, idx
