"""
File:
    src/data/DatasetManager.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Generate datasets for training, validation and testing.
"""

import json
import os
import sys
import zipfile

import numpy as np
import ray
import requests
from PIL import Image
from torchvision import transforms
# from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm, trange
from typing import Tuple, Optional
from src.data.DatasetManagers.CustomDatasetManager import CustomDatasetManager, ray_resize_images, ray_get_rgb
from src.data.SplittingManager import SplittingManager
from src.utils.constants import *
from src.data.Datasets.FuseDataset import FuseDataset


class LearningDatasetManager(CustomDatasetManager):
    """
    Dataset Manager class, handles the creation of the training, validation and testing datasets.
    """
    def __init__(self,
                 images_path: str,
                 targets_path: str,
                 image_size: int,
                 num_workers: int,
                 data_aug: float,
                 validation_size: float,
                 test_size: float,
                 norm: str,
                 google_images: bool,
                 seed: int,
                 splitting_manager: SplittingManager,
                 current_fold: int) -> None:
        """
        Class constructor

        :param images_path: str, path to images files
        :param targets_path: str, path to targets file (ground truths)
        :param image_size: int, maximum image size in pixels, images are resized to (image_size, image_size)
        :param num_workers: int, number of workers for multiprocessing
        :param data_aug: float, intensity of data augmentation transforms
        :param validation_size: float, size of validation dataset as a subset of the entire dataset
        :param test_size: float, size of test dataset as a subset of the entire dataset
        :param mean_std: bool, if True, mean and std values for RGB channel normalization will be calculated
                               if False, mean and std precalculated values will be used
        """
        self._google_images = google_images
        self._seed = seed

        # Check if any image exists in the data/resized folder
        if any(file.endswith(f'.{IMAGE_EXT}') for file in os.listdir(RESIZED_LEARNING_PATH)):
            # Get the first found image's size
            for file in os.listdir(RESIZED_LEARNING_PATH):
                if file.endswith(f'.{IMAGE_EXT}'):
                    img_size = Image.open(f'{RESIZED_LEARNING_PATH}{file}').size
                    break

            # Check if the first image's size is not equal to the image_size parameter
            if img_size != (image_size, image_size):
                print(f'Max image size argument is {(image_size, image_size)} '
                      f'but a resized image of {img_size} was found')
                print(f'All images will be resized to {(image_size, image_size)}')

                # Resize all images
                self._resize_images(image_size, num_workers)
        else:
            # Check if any image exists in the data/raw folder
            # if any(file.endswith(f'.{IMAGE_EXT}') for file in os.listdir(RAW_LEARNING_PATH)):
            if os.path.isdir(RAW_LEARNING_PATH):
                # Resize all images
                self._resize_images(image_size, num_workers)
            else:
                # Ask the user if the data should be downloaded
                if input('Raw data folder contains no images. '
                         'Do you want to download them? (~ 3 GB) (y/n): ') == 'y':
                    # Download the data
                    self.__fetch_data(IMAGES_ID, ANNOTATIONS_ID)

                    # Resize all images
                    self._resize_images(image_size, num_workers)
                else:
                    # Exit the program
                    sys.exit(1)

        # Declare training, validation and testing datasets
        self._dataset_train = FuseDataset(image_paths=splitting_manager.image_paths_train[current_fold],
                                          targets=splitting_manager.targets_train[current_fold],
                                          num_workers=num_workers,
                                          phase='Training')
        self._dataset_valid = FuseDataset(image_paths=splitting_manager.image_paths_valid[current_fold],
                                          targets=splitting_manager.targets_valid[current_fold],
                                          num_workers=num_workers,
                                          phase='Validation')
        self._dataset_test = FuseDataset(image_paths=splitting_manager.image_paths_test,
                                         targets=splitting_manager.targets_test,
                                         num_workers=num_workers,
                                         phase='Testing')

        # # Get total dataset size
        # total_size = sum(image_path.rsplit('/')[-1].startswith('S') for image_path in self._dataset_train.image_paths)
        #
        # # Split the training set into training + validation
        # self._dataset_train, self._dataset_valid = self.__split_dataset(self._dataset_train, self._dataset_valid,
        #                                                                 validation_size, total_size)
        #
        # # Split the training set into training + testing
        # self._dataset_train, self._dataset_test = self.__split_dataset(self._dataset_train, self._dataset_test,
        #                                                                test_size, total_size)

        if norm == 'precalculated':
            # Use precalculated mean and standard deviation
            mean, std = MEAN, STD
        elif norm == 'calculated':
            # Recalculate mean and standard deviation
            mean, std = self._calculate_mean_std(num_workers)
        elif norm == 'none':
            mean, std = None, None

        # Apply transforms to the training, validation and testing datasets
        self._dataset_train.transforms = self.__transforms_train(mean, std, data_aug)
        self._dataset_valid.transforms = self._transforms_base(mean, std)
        self._dataset_test.transforms = self._transforms_base(mean, std)

    @property
    def dataset_train(self):
        return self._dataset_train

    @property
    def dataset_valid(self):
        return self._dataset_valid

    @property
    def dataset_test(self):
        return self._dataset_test


    @staticmethod
    def __transforms_train(mean: Optional[Tuple[float, float, float]],
                           std: Optional[Tuple[float, float, float]],
                           data_aug: float) -> transforms.Compose:
        """
        Method to construct the training dataset transforms

        :param mean: tuple of 3 floats, containing the mean (R, G, B) values
        :param std: tuple of 3 floats, containing the standard deviation of (R, G, B) values
        :param data_aug: float, intensity of data augmentation transforms
        :return: transforms.Compose, custom composed transforms list
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

    # def __split_dataset(self, dataset_in: FuseDataset, dataset_out: FuseDataset, split_size: float, total_size: int) \
    #         -> Tuple[FuseDataset, FuseDataset]:
    #     """
    #     Split a dataset into two sub-datasets, used to create the validation and testing dataset splits
    #
    #     :param dataset_in: FuseDataset, input dataset from which to extract data
    #     :param dataset_out: FuseDataset, output dataset into which we insert data
    #     :param split_size: float, size of the split
    #     :param total_size: int, total size of the original dataset
    #     :return: tuple of two FuseDataset objects, which are the dataset_in and dataset_out after splitting
    #     """
    #
    #     split_size = split_size / (1 - HOLDOUT_SIZE)
    #
    #     if 0 < split_size < 1:
    #         if self._google_images:
    #             google_image_paths = [image_path for image_path in dataset_in.image_paths
    #                                   if image_path.rsplit('/')[-1].startswith('G')]
    #
    #             google_indices = [dataset_in.image_paths.index(google_image_path)
    #                               for google_image_path in google_image_paths]
    #
    #             google_image_paths, google_images, google_targets = dataset_in.extract_data(index_list=google_indices)
    #
    #         most_freq_labels = [max(set(target['labels'].tolist()), key=target['labels'].tolist().count)
    #                             for target in dataset_in.targets]
    #
    #         strat_split = StratifiedShuffleSplit(n_splits=1,
    #                                              test_size=split_size * total_size / len(dataset_in),
    #                                              random_state=self._seed)
    #
    #         # Iterate through generator object once and break
    #         for _, indices in strat_split.split([0] * len(dataset_in), most_freq_labels):
    #             break
    #
    #         # Extract the image paths, images and targets from dataset_in
    #         image_paths, images, targets = dataset_in.extract_data(index_list=indices)
    #
    #         # Insert the extracted image paths, images and targets into dataset_out
    #         dataset_out.add_data(image_paths, images, targets)
    #
    #         if self._google_images:
    #             dataset_in.add_data(google_image_paths, google_images, google_targets)
    #
    #     if split_size == 1:
    #         return dataset_out, dataset_in
    #     else:
    #         # Return the split_size input and output datasets
    #         return dataset_in, dataset_out

    def _calculate_mean_std(self, num_workers: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
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

    @staticmethod
    def _resize_images(image_size: int, num_workers: int) -> None:
        """
        Method to resize all images in the data/raw folder and save them to the data/resized folder

        :param image_size: int, maximum image size in pixels (will be used for height & width)
        :param num_workers: int, number of workers for multiprocessing
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
