"""
File:
    src/data/DatasetManagers/LearningDatasetManager.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Generate datasets for training, validation and testing.
"""

import json
import os
import numpy as np
import ray
from torchvision import transforms
from tqdm import trange
from typing import Optional, Tuple

from src.data.DatasetManagers.CustomDatasetManager import CustomDatasetManager, ray_resize_images, ray_get_rgb
from src.data.SplittingManager import SplittingManager
from src.utils.constants import *
from src.data.Datasets.FuseDataset import FuseDataset


class LearningDatasetManager(CustomDatasetManager):
    """Dataset Manager class, handles the creation of the training, validation and testing datasets."""
    def __init__(self,
                 num_workers: int,
                 data_aug: float,
                 validation_size: float,
                 test_size: float,
                 norm: str,
                 google_images: bool,
                 seed: int,
                 splitting_manager: SplittingManager,
                 current_fold: int) -> None:
        """Class constructor

        Args:
            num_workers(int): number of workers for multiprocessing
            data_aug(float): intensity of data augmentation transforms
            validation_size(float): size of validation dataset as a subset of the entire dataset
            test_size(float): size of test dataset as a subset of the entire dataset
            norm(str): RGB pixel normalization parameter, either 'precalculated', 'calculated' or 'none'
            google_images(bool): choose whether to include images from google
            seed(int): initialization seed
            splitting_manager(SplittingManager): splitting manager for k-cross validation + train, valid, test splitting
            current_fold(int): current k-fold cross validation fold

        """
        self._google_images = google_images
        self._seed = seed

        # Declare training, validation and testing datasets
        if test_size < 1:
            self._dataset_train = FuseDataset(images_path=splitting_manager.images_path,
                                              images_filenames=splitting_manager.image_paths_train[current_fold],
                                              targets=splitting_manager.targets_train[current_fold],
                                              num_workers=num_workers,
                                              phase='Training')
        else:
            self._dataset_train = []

        if validation_size > 0:
            self._dataset_valid = FuseDataset(images_path=splitting_manager.images_path,
                                              images_filenames=splitting_manager.image_paths_valid[current_fold],
                                              targets=splitting_manager.targets_valid[current_fold],
                                              num_workers=num_workers,
                                              phase='Validation')
        else:
            self._dataset_valid = []

        if test_size > 0:
            self._dataset_test = FuseDataset(images_path=splitting_manager.images_path,
                                             images_filenames=splitting_manager.image_paths_test,
                                             targets=splitting_manager.targets_test,
                                             num_workers=num_workers,
                                             phase='Testing')
        else:
            self._dataset_test = []

        if norm == 'precalculated':
            # Use precalculated mean and standard deviation
            mean, std = MEAN, STD
        elif norm == 'calculated':
            # Recalculate mean and standard deviation
            mean, std = self._calculate_mean_std(num_workers)
        elif norm == 'none':
            mean, std = None, None

        # Apply transforms to the training, validation and testing datasets
        if (test_size + validation_size) < 1:
            self._dataset_train.transforms = self.__transforms_train(mean, std, data_aug)

        if validation_size > 0:
            self._dataset_valid.transforms = self._transforms_base(mean, std)

        if test_size > 0:
            self._dataset_test.transforms = self._transforms_base(mean, std)

    @property
    def dataset_train(self):
        """ """
        return self._dataset_train

    @property
    def dataset_valid(self):
        """ """
        return self._dataset_valid

    @property
    def dataset_test(self):
        """ """
        return self._dataset_test

    @staticmethod
    def __transforms_train(mean: Optional[Tuple[float, float, float]],
                           std: Optional[Tuple[float, float, float]],
                           data_aug: float) -> transforms.Compose:
        """Method to construct the training dataset transforms

        Args:
            mean(Optional[Tuple[float, float, float]]): tuple of 3 floats, containing the mean (R, G, B) values
            std(Optional[Tuple[float, float, float]]): tuple of 3 floats, containing the standard deviation of (R, G, B) values
            data_aug(float): intensity of data augmentation transforms

        Returns:
            transforms.Compose: custom composed transforms list

        """
        transforms_list = [
            # Apply ColorJitter data augmentation transform
            transforms.ColorJitter(brightness=data_aug,
                                   contrast=data_aug,
                                   saturation=data_aug,
                                   hue=0),

            # Convert to tensor
            transforms.ToTensor(),
        ]

        # Normalize by mean, std
        if mean is not None:
            transforms_list.append(transforms.Normalize(mean, std))

        # Return a composed transforms list
        return transforms.Compose(transforms_list)

    def _calculate_mean_std(self,
                            num_workers: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Method to calculate the mean and standard deviation for each channel (R, G, B) for each image

        Args:
            num_workers(int): number of workers for multiprocessing

        Returns:
            Tuple[Tuple[float,float,float],Tuple[float,float,float]]: tuple, containing the mean and std deviation values for each channel (R, G, B)

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

    @staticmethod
    def _resize_images(image_size: int, num_workers: int) -> None:
        """Method to resize all images in the data/raw folder and save them to the data/resized folder

        Args:
            image_size(int): maximum image size in pixels (will be used for height & width)
            num_workers(int): number of workers for multiprocessing

        """
        # Initialize ray
        ray.init(include_dashboard=False)

        # Get list of image and exclude the hidden .gitkeep file
        imgs = [img for img in sorted(os.listdir(RAW_LEARNING_PATH)) if img.startswith('.') is False]

        # Create image paths
        image_paths = [os.path.join(RAW_LEARNING_PATH, img) for img in imgs]

        # Get dataset size
        size = len(image_paths)

        # Declare empty lists to save the resize ratios and targets
        resize_ratios = [None] * size
        targets_list = [None] * size

        # Get ray workers IDs
        ids = [ray_resize_images.remote(image_paths, RESIZED_LEARNING_PATH, image_size, ANNOTATIONS_PATH, i) for i in range(num_workers)]

        # Calculate initial number of jobs left
        nb_job_left = size - num_workers

        # Multiprocessing loop
        for _ in trange(size, desc='Resizing images...'):
            # Handle ray workers ready states and IDs
            ready, ids = ray.wait(ids, num_returns=1)

            # Get resize ratios, indices, box_list and targets
            resize_ratio, idx, box_list, targets = ray.get(ready)[0]

            # Save the resize ratios and targets to lists
            resize_ratios[idx] = resize_ratio
            targets_list[idx] = targets

            # Check if there are jobs left
            if nb_job_left > 0:
                # Assign workers to the remaining tasks
                ids.extend([ray_resize_images.remote(image_paths, RESIZED_LEARNING_PATH, image_size, ANNOTATIONS_PATH, size - nb_job_left)])

                # Decreasing the number of jobs left
                nb_job_left -= 1

        # Shutting down ray
        ray.shutdown()

        # Calculating and displaying the average, maximum and minimum resizing ratios
        average_ratio = sum(resize_ratios) / len(resize_ratios)
        print(f'\nAverage resize ratio: {average_ratio:.2%}')
        print(f'Maximum resize ratio: {max(resize_ratios):.2%}')
        print(f'Minimum resize ratio: {min(resize_ratios):.2%}')

        # Saving the targets to a json file
        json.dump(targets_list, open(TARGETS_LEARNING_PATH, 'w'), ensure_ascii=False)

        # Displaying where files have been saved to
        print(f'\nResized images have been saved to:\t\t{RESIZED_LEARNING_PATH}')
        print(f'Resized targets have been saved to:\t\t{TARGETS_LEARNING_PATH}')
