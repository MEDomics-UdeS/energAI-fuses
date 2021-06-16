"""
File:
    src/data/DatasetManager.py

Authors:
    - Simon Giard-Leroux
    - Shreyas Sunil Kulkarni

Description:
    Generate datasets for training, validation and testing.
"""

import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import ray
import requests
from PIL import Image, ImageDraw
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm, trange
from typing import Tuple, List, Optional

from src.utils.constants import *
from src.data.FuseDataset import FuseDataset


class DatasetManager:
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
                 seed: int) -> None:
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
        self.__google_images = google_images
        self.__seed = seed

        # Check if any image exists in the data/resized folder
        if any(file.endswith(f'.{IMAGE_EXT}') for file in os.listdir(RESIZED_PATH)):
            # Get the first found image's size
            for file in os.listdir(RESIZED_PATH):
                if file.endswith(f'.{IMAGE_EXT}'):
                    img_size = Image.open(f'{RESIZED_PATH}{file}').size
                    break

            # Check if the first image's size is not equal to the image_size parameter
            if img_size != (image_size, image_size):
                print(f'Max image size argument is {(image_size, image_size)} '
                      f'but a resized image of {img_size} was found')
                print(f'All images will be resized to {(image_size, image_size)}')

                # Resize all images
                self.__resize_images(image_size, num_workers)
        else:
            # Check if any image exists in the data/raw folder
            if any(file.endswith(f'.{IMAGE_EXT}') for file in os.listdir(RAW_PATH)):
                # Resize all images
                self.__resize_images(image_size, num_workers)
            else:
                # Ask the user if the data should be downloaded
                if input('Raw data folder contains no images. '
                         'Do you want to download them? (~ 3 GB) (y/n): ') == 'y':
                    # Download the data
                    self.__fetch_data(IMAGES_ID, ANNOTATIONS_ID)

                    # Resize all images
                    self.__resize_images(image_size, num_workers)
                else:
                    # Exit the program
                    sys.exit(1)

        # Declare training, validation and testing datasets
        self.__dataset_train = FuseDataset(images_path, targets_path, num_workers, google_images)
        self.__dataset_valid = FuseDataset()
        self.__dataset_test = FuseDataset()

        # Get total dataset size
        total_size = sum(image_path.rsplit('/')[-1].startswith('S') for image_path in self.__dataset_train.image_paths)

        # Split the training set into training + validation
        self.__dataset_train, self.__dataset_valid = self.__split_dataset(self.__dataset_train, self.__dataset_valid,
                                                                          validation_size, total_size)

        # Split the training set into training + testing
        self.__dataset_train, self.__dataset_test = self.__split_dataset(self.__dataset_train, self.__dataset_test,
                                                                         test_size, total_size)

        if norm == 'precalculated':
            # Use precalculated mean and standard deviation
            mean, std = MEAN, STD
        elif norm == 'calculated':
            # Recalculate mean and standard deviation
            mean, std = self.__calculate_mean_std(num_workers)
        elif norm == 'none':
            mean, std = None, None

        # Apply transforms to the training, validation and testing datasets
        self.__dataset_train.transforms = self.__transforms_train(mean, std, data_aug)
        self.__dataset_valid.transforms = self.__transforms_base(mean, std)
        self.__dataset_test.transforms = self.__transforms_base(mean, std)

    @property
    def dataset_train(self):
        return self.__dataset_train

    @property
    def dataset_valid(self):
        return self.__dataset_valid

    @property
    def dataset_test(self):
        return self.__dataset_test

    @staticmethod
    def __download_file_from_google_drive(file_id: str, dest: str, chunk_size: int = 32768) -> None:
        """
        Method to download a file from Google Drive

        Inspired from :
        https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

        :param file_id: str, Google Drive file ID hash to download
        :param dest: str, filepath + filename to save the contents to
        :param chunk_size: int, chunk size in bytes
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

    def __fetch_data(self, images_id: str, annotations_id: str) -> None:
        """
        Method to fetch the images and annotations from Google Drive

        :param images_id: str, Google Drive file ID hash for the images zip file
        :param annotations_id: str, Google Drive file ID hash for the annotations file
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

    def __split_dataset(self, dataset_in: FuseDataset, dataset_out: FuseDataset, split_size: float, total_size: int) \
            -> Tuple[FuseDataset, FuseDataset]:
        """
        Split a dataset into two sub-datasets, used to create the validation and testing dataset splits

        :param dataset_in: FuseDataset, input dataset from which to extract data
        :param dataset_out: FuseDataset, output dataset into which we insert data
        :param split_size: float, size of the split
        :param total_size: int, total size of the original dataset
        :return: tuple of two FuseDataset objects, which are the dataset_in and dataset_out after splitting
        """

        if 0 < split_size < 1:
            if self.__google_images:
                google_image_paths = [image_path for image_path in dataset_in.image_paths
                                      if image_path.rsplit('/')[-1].startswith('G')]

                google_indices = [dataset_in.image_paths.index(google_image_path)
                                  for google_image_path in google_image_paths]

                google_image_paths, google_images, google_targets = dataset_in.extract_data(index_list=google_indices)

            most_freq_labels = [max(set(target['labels'].tolist()), key=target['labels'].tolist().count)
                                for target in dataset_in.targets]

            strat_split = StratifiedShuffleSplit(n_splits=1,
                                                 test_size=split_size * total_size / len(dataset_in),
                                                 random_state=self.__seed)

            # Iterate through generator object once and break
            for _, indices in strat_split.split([0] * len(dataset_in), most_freq_labels):
                break

            # Extract the image paths, images and targets from dataset_in
            image_paths, images, targets = dataset_in.extract_data(index_list=indices)

            # Insert the extracted image paths, images and targets into dataset_out
            dataset_out.add_data(image_paths, images, targets)

            if self.__google_images:
                dataset_in.add_data(google_image_paths, google_images, google_targets)

        if split_size == 1:
            return dataset_out, dataset_in
        else:
            # Return the split_size input and output datasets
            return dataset_in, dataset_out

    def __calculate_mean_std(self, num_workers: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Method to calculate the mean and standard deviation for each channel (R, G, B) for each image

        :param num_workers: int, number of workers for multiprocessing
        :return: tuple, containing the mean and std deviation values for each channel (R, G, B)
        """
        # Initialize ray
        ray.init(include_dashboard=False)

        # Get ray workers IDs
        ids = [ray_get_rgb.remote(self.__dataset_train.image_paths, i) for i in range(num_workers)]

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
                ids.extend([ray_get_rgb.remote(self.__dataset_train.image_paths, size - nb_job_left)])

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
    def __resize_images(image_size: int, num_workers: int) -> None:
        """
        Method to resize all images in the data/raw folder and save them to the data/resized folder

        :param image_size: int, maximum image size in pixels (will be used for height & width)
        :param num_workers: int, number of workers for multiprocessing
        """
        # Initialize ray
        ray.init(include_dashboard=False)

        # Get list of image and exclude the hidden .gitkeep file
        imgs = [img for img in sorted(os.listdir(RAW_PATH)) if img.startswith('.') is False]

        # Create image paths
        image_paths = [os.path.join(RAW_PATH, img) for img in imgs]

        # Convert the annotations csv file to a pandas DataFrame
        annotations = pd.read_csv(ANNOTATIONS_PATH)

        # Get dataset size
        size = len(image_paths)

        # Declare empty lists to save the resize ratios and targets
        resize_ratios = [None] * size
        targets_list = [None] * size

        # Get ray workers IDs
        ids = [ray_resize_images.remote(image_paths, image_size, annotations, i) for i in range(num_workers)]

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
                ids.extend([ray_resize_images.remote(image_paths, image_size, annotations, size - nb_job_left)])

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
        json.dump(targets_list, open(TARGETS_PATH, 'w'), ensure_ascii=False)

        # Displaying where files have been saved to
        print(f'\nResized images have been saved to:\t\t{RESIZED_PATH}')
        print(f'Resized targets have been saved to:\t\t{TARGETS_PATH}')


@ray.remote
def ray_resize_images(image_paths: List[str], image_size: int, annotations: pd.DataFrame,
                      idx: int, show_bounding_boxes: bool = False) -> Tuple[float, int, np.array, dict]:
    """
    Ray remote function to parallelize the resizing of images

    :param image_paths: list, contains image paths
    :param image_size: int, image size to resize all images to (height & width)
    :param annotations: pandas DataFrame, contains the coordinates of each ground truth bounding box for each image
    :param idx: int, current index
    :param show_bounding_boxes: bool, if True, ground truth bounding boxes are drawn on the resized images
                                (used to test if bounding boxes are properly resized)
    :return: resize_ratio, idx, box_array, targets to continue ray paralleling
    """
    # Get the current image name, without the file path and without the file extension
    f = image_paths[idx].rsplit('/', 1)[-1].split(".")[0]

    # Get bounding boxes coordinates for the current image
    box_array = annotations.loc[annotations["filename"] == f][["xmin", "ymin", "xmax", "ymax"]].values

    # Get class labels (str) for the current image
    label_array = annotations.loc[annotations["filename"] == f][["label"]].values

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
    resize_ratio = (img2.size[0] * img2.size[1]) / (original_size[0] * original_size[1])

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
    img2.paste(img, (x_offset, y_offset, x_offset + img.size[0], y_offset + img.size[1]))

    # Declare an ImageDraw object if the show_bounding_boxes argument is set to True
    if show_bounding_boxes:
        draw = ImageDraw.Draw(img2)

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

    # Save the resized image
    img2.save(f'data/resized/{image_paths[idx].split("/")[-1]}')

    # Calculate the area of each bounding box
    area = [int(a) for a in list((box_array[:, 3] - box_array[:, 1]) * (box_array[:, 2] - box_array[:, 0]))]

    # Save ground truth targets to a dictionary
    targets = {"boxes": list(box_array.tolist()),
               "labels": label_list,
               "image_id": idx,
               "area": area,
               "iscrowd": [0] * num_boxes}

    # Return the objects required for ray parallelization
    return resize_ratio, idx, box_array, targets


@ray.remote
def ray_get_rgb(image_paths: List[str], idx: int) -> Tuple[np.array, np.array, np.array, int]:
    """
    Ray remote function to parallelize the extraction of R, G, B values from images

    :param image_paths: list, contains the image paths
    :param idx: int, current index
    :return: tuple, r, g, b values numpy arrays for the current image and the current index
    """
    # Open the current image
    image = Image.open(image_paths[idx])

    # Get the values of each pixel in the R, G, B channels
    r = np.dstack(np.array(image)[:, :, 0])
    g = np.dstack(np.array(image)[:, :, 1])
    b = np.dstack(np.array(image)[:, :, 2])

    # Return the r, g, b values numpy arrays for the current image and the current index
    return r, g, b, idx
