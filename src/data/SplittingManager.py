"""
File:
    src/data/SplittingManager.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Contains the SplittingManager class to split dataset into training, validation, testing and k-fold sets.
"""

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import os
import json
import torch
import sys
import zipfile
import requests
from tqdm import tqdm, trange
from typing import List
from PIL import Image
import ray

from src.data.DatasetManagers.CustomDatasetManager import ray_resize_images
from src.utils.constants import *


class SplittingManager:
    """Splitting manager class"""
    def __init__(self,
                 dataset: str,
                 validation_size: float,
                 test_size: float,
                 k_cross_valid: int,
                 seed: int,
                 google_images: bool,
                 image_size: int,
                 num_workers: int) -> None:
        """Class constructor

        Args:
            dataset(str): dataset string, either 'learning' or 'holdout'
            validation_size(float): validation set size in float [0, 1]
            test_size(float): test set size in float [0, 1]
            k_cross_valid(int): number of folds for k-fold cross-validation
            seed(int): splitting seed
            google_images(bool): choose whether to include google images or not
            image_size(int): image size for image resizing purposes
            num_workers(int): number of workers for multiprocessing

        """

        self.__images_path = RESIZED_LEARNING_PATH if dataset == 'learning' else RESIZED_HOLDOUT_PATH
        self.__raw_images_path = RAW_LEARNING_PATH if dataset == 'learning' else RAW_HOLDOUT_PATH
        self.__targets_path = TARGETS_LEARNING_PATH if dataset == 'learning' else TARGETS_HOLDOUT_PATH

        self.__validation_size = validation_size
        self.__test_size = test_size
        self.__k_cross_valid = k_cross_valid
        self.__seed = seed
        self.__google_images = google_images

        # Check if any image exists in the data/resized folder
        if any(file.endswith(f'.{IMAGE_EXT}') for file in os.listdir(self.__images_path)):
            # Get the first found image's size
            for file in os.listdir(self.__images_path):
                if file.endswith(f'.{IMAGE_EXT}'):
                    img_size = Image.open(f'{self.__images_path}{file}').size
                    break

            # Check if the first image's size is not equal to the image_size parameter
            if img_size != (image_size, image_size):
                print(f'Max image size argument is {(image_size, image_size)} '
                      f'but a resized image of {img_size} was found')
                print(f'All images will be resized to {(image_size, image_size)}')

                # Resize all images
                self._resize_images(image_size, num_workers)
        else:
            if os.path.isdir(self.__raw_images_path):
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

        images = [img for img in sorted(os.listdir(self.__raw_images_path)) if img.startswith('.') is False]

        if not google_images:
            google_imgs = [image for image in images if image.startswith('G')]
            google_indices = [images.index(google_image) for google_image in google_imgs]
            images = [e for i, e in enumerate(images) if i not in google_indices]

        self.__image_paths = [img for img in images]

        self.__targets = json.load(open(self.__targets_path))

        if not google_images:
            self.__targets = [e for i, e in enumerate(self.__targets) if i not in google_indices]

        # Convert the targets to tensors
        for target in self.__targets:
            for key, value in target.items():
                target[key] = torch.as_tensor(value, dtype=torch.int64)

        self.__split_dataset()

        # For backwards results reproducibility, valid and test sets are reverse ordered
        for i in range(k_cross_valid):
            self.__image_paths_valid[i].reverse()
            self.__targets_valid[i].reverse()

        self.__image_paths_test.reverse()
        self.__targets_test.reverse()

    @property
    def images_path(self) -> str:
        """Returns the image paths"""

        return self.__images_path

    @property
    def image_paths_train(self) -> list:
        """Returns the training image paths"""

        return self.__image_paths_train

    @property
    def targets_train(self) -> list:
        """Returns the training labels"""

        return self.__targets_train

    @property
    def image_paths_valid(self) -> list:
        """Returns the validation image paths"""

        return self.__image_paths_valid

    @property
    def targets_valid(self) -> list:
        """Returns the validation labels"""

        return self.__targets_valid

    @property
    def image_paths_test(self) -> list:
        """Returs the image paths"""

        return self.__image_paths_test

    @property
    def targets_test(self) -> list:
        """Returns the test labels"""

        return self.__targets_test

    def __split_dataset(self) -> None:
        """Splitting the datasets"""

        image_paths = self.__image_paths
        targets = self.__targets

        total_size = sum(image_path.rsplit('/')[-1].startswith('S') for image_path in image_paths)

        if 0 < self.__validation_size < 1:
            self.__validation_size = self.__validation_size / (1 - HOLDOUT_SIZE)

        if 0 < self.__test_size < 1:
            self.__test_size = self.__test_size / (1 - HOLDOUT_SIZE)

        if self.__google_images:
            google_image_paths = [image_path for image_path in image_paths
                                  if image_path.rsplit('/')[-1].startswith('G')]

            google_indices = [image_paths.index(google_image_path)
                              for google_image_path in google_image_paths]

            google_targets = self.__filter_list(targets, google_indices, True)

            image_paths = self.__filter_list(image_paths, google_indices, False)
            targets = self.__filter_list(targets, google_indices, False)

        most_freq_labels = self.__get_most_frequent_labels(targets)

        if self.__k_cross_valid > 1:
            if self.__test_size > 0:
                strat_split_test = StratifiedShuffleSplit(n_splits=1,
                                                          test_size=self.__test_size,
                                                          random_state=self.__seed)

                for _, indices_test in strat_split_test.split([0] * len(image_paths), most_freq_labels):
                    break

                indices_test = list(indices_test)

                self.__image_paths_test = self.__filter_list(image_paths, indices_test, True)
                self.__targets_test = self.__filter_list(targets, indices_test, True)

                image_paths = self.__filter_list(image_paths, indices_test, False)
                targets = self.__filter_list(targets, indices_test, False)

                most_freq_labels = self.__get_most_frequent_labels(targets)
            else:
                self.__image_paths_test = []
                self.__targets_test = []

            strat_split_valid = StratifiedKFold(n_splits=self.__k_cross_valid,
                                                random_state=self.__seed, shuffle=True)

            indices_valid = []

            for _, indices in strat_split_valid.split([0] * len(image_paths), most_freq_labels):
                indices_valid.append(list(indices))

            image_paths_train = []
            targets_train = []

            image_paths_valid = []
            targets_valid = []

            for i in range(self.__k_cross_valid):
                image_paths_train.append(self.__filter_list(image_paths, indices_valid[i], False))
                targets_train.append(self.__filter_list(targets, indices_valid[i], False))

                if self.__google_images:
                    image_paths_train[i] = image_paths_train[i] + google_image_paths
                    targets_train[i] = targets_train[i] + google_targets

                image_paths_valid.append(self.__filter_list(image_paths, indices_valid[i], True))
                targets_valid.append(self.__filter_list(targets, indices_valid[i], True))

            self.__image_paths_train = image_paths_train
            self.__targets_train = targets_train

            self.__image_paths_valid = image_paths_valid
            self.__targets_valid = targets_valid
        else:
            if self.__validation_size == 0 and self.__test_size == 0:
                self.__image_paths_train = [image_paths + google_image_paths]
                self.__targets_train = [targets + google_targets]

                self.__image_paths_valid = [[]]
                self.__targets_valid = [[]]

                self.__image_paths_test = []
                self.__targets_test = []
            else:
                if self.__validation_size > 0:
                    strat_split_valid = StratifiedShuffleSplit(n_splits=1,
                                                               test_size=self.__validation_size,
                                                               random_state=self.__seed)

                    for _, indices_valid in strat_split_valid.split([0] * len(image_paths), most_freq_labels):
                        break

                    indices_valid = list(indices_valid)

                    self.__image_paths_valid = [self.__filter_list(image_paths, indices_valid, True)]
                    self.__targets_valid = [self.__filter_list(targets, indices_valid, True)]

                    image_paths = self.__filter_list(image_paths, indices_valid, False)
                    targets = self.__filter_list(targets, indices_valid, False)

                    most_freq_labels = self.__get_most_frequent_labels(targets)
                else:
                    self.__image_paths_valid = [[]]
                    self.__targets_valid = [[]]

                if self.__test_size == 1:
                    self.__image_paths_train = []
                    self.__targets_train = []

                    self.__image_paths_test = image_paths
                    self.__targets_test = targets
                elif self.__test_size == 0:
                    self.__image_paths_train = [image_paths + google_image_paths]
                    self.__targets_train = [targets + google_targets]

                    self.__image_paths_test = []
                    self.__targets_test = []
                else:
                    strat_split_test = StratifiedShuffleSplit(n_splits=1,
                                                              test_size=self.__test_size * total_size / len(image_paths),
                                                              random_state=self.__seed)

                    for _, indices_test in strat_split_test.split([0] * len(image_paths), most_freq_labels):
                        break

                    indices_test = list(indices_test)

                    self.__image_paths_train = [self.__filter_list(image_paths, indices_test, False) + google_image_paths]
                    self.__targets_train = [self.__filter_list(targets, indices_test, False) + google_targets]

                    self.__image_paths_test = self.__filter_list(image_paths, indices_test, True)
                    self.__targets_test = self.__filter_list(targets, indices_test, True)

    @staticmethod
    def __filter_list(my_list: List[str], indices: List[int], logic: bool) -> List[str]:
        """Method to filter a list based on a number of indices, either in direct or inverse logic

        Args:
            my_list(List[str]): input list
            indices(List[int]): list of indices
            logic(bool): choose whether to filter the list based on if the indices are in or not in the list

        Returns:
            list: filtered list

        """
        if logic:
            return [my_list[i] for i in range(len(my_list)) if i in indices]
        else:
            return [my_list[i] for i in range(len(my_list)) if i not in indices]

    @staticmethod
    def __get_most_frequent_labels(targets: list) -> list:
        """Function to find the most frequent labels in the ground truth file

        Args:
            targets(list): input targets list

        Returns:
            list: list of most frequent labels

        """
        return [max(set(target['labels'].tolist()), key=target['labels'].tolist().count)
                for target in targets]

    @staticmethod
    def __download_file_from_google_drive(file_id: str,
                                          dest: str,
                                          chunk_size: int = 32768) -> None:
        """Method to download a file from Google Drive
        
        Inspired from :
        https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

        Args:
            file_id(str): Google Drive file ID hash to download
            dest(str): filepath + filename to save the contents to
            chunk_size(int, optional): chunk size in bytes (Default value = 32768)

        """
        # Declare URL, session, response and token objects
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={'id': file_id}, stream=True)
        token = None

        # Get token from the response cookie 'download_warning'
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value

        # If token obtained, get params and response
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        # If the response is OK, then download the file chunk by chunk
        if response.ok:
            with open(dest, "wb") as f:
                for chunk in tqdm(response.iter_content(chunk_size), desc='Downloading'):
                    if chunk:
                        f.write(chunk)
        else:
            raise Exception(f'Error {response.status_code}: {response.reason}')

    def __fetch_data(self,
                     images_id: str,
                     annotations_id: str) -> None:
        """Method to fetch the images and annotations from Google Drive

        Args:
            images_id(str): Google Drive file ID hash for the images zip file
            annotations_id(str): Google Drive file ID hash for the annotations file

        """
        # Create the file path for the images zip file
        images_zip = os.path.join(RAW_PATH, 'images.zip')

        # Download images zip file
        print('\nDownloading images to:\t\t', RAW_PATH)
        self.__download_file_from_google_drive(images_id, images_zip)

        # Unzip the images zip file
        print('Done!\nUnzipping images...')

        with zipfile.ZipFile(images_zip, 'r') as zip_ref:
            zip_ref.extractall(RAW_PATH)

        # Delete the images zip file
        os.remove(images_zip)

        # Download the annotations file
        print('Done!\n\nDownloading annotations to:\t', ANNOTATIONS_PATH)
        self.__download_file_from_google_drive(annotations_id, ANNOTATIONS_PATH)
        print('Done!')

    def _resize_images(self,
                       image_size: int,
                       num_workers: int) -> None:
        """Method to resize all images in the data/raw folder and save them to the data/resized folder

        Args:
            image_size(int): int, maximum image size in pixels (will be used for height & width)
            num_workers(int): int, number of workers for multiprocessing

        """
        # Initialize ray
        ray.init(include_dashboard=False)

        # Get list of image and exclude the hidden .gitkeep file
        imgs = [img for img in sorted(os.listdir(self.__raw_images_path)) if img.startswith('.') is False]

        # Create image paths
        image_paths = [os.path.join(self.__raw_images_path, img) for img in imgs]

        # Get dataset size
        size = len(image_paths)

        # Declare empty lists to save the resize ratios and targets
        resize_ratios = [None] * size
        targets_list = [None] * size

        # Get ray workers IDs
        ids = [ray_resize_images.remote(image_paths, self.__images_path, image_size, ANNOTATIONS_PATH, i) for i in range(num_workers)]

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
                ids.extend([ray_resize_images.remote(image_paths, self.__images_path, image_size, ANNOTATIONS_PATH, size - nb_job_left)])

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
        json.dump(targets_list, open(self.__targets_path, 'w'), ensure_ascii=False)

        # Displaying where files have been saved to
        print(f'\nResized images have been saved to:\t\t{self.__images_path}')
        print(f'Resized targets have been saved to:\t\t{self.__targets_path}')
