"""
File:
    src/data/DataLoaderManager.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Generate data loaders for batch management and multiprocessing during training, validation and testing.
"""

from src.data.DatasetManagers.LearningDatasetManager import LearningDatasetManager
from src.data.DataLoaderManagers.CustomDataLoaderManager import CustomDataLoaderManager


class LearningDataLoaderManager(CustomDataLoaderManager):
    """
    Data Loader Manager class, handles the creation of the training, validation and testing data loaders.
    """
    def __init__(self,
                 dataset_manager: LearningDatasetManager,
                 batch_size: int,
                 gradient_accumulation: int,
                 num_workers: int,
                 deterministic: bool) -> None:
        """
        Class constructor.

        :param dataset_manager: DatasetManager class, contains the training, validation and testing datasets
        :param batch_size: int, mini-batch size for data loaders
        :param gradient_accumulation: int, gradient accumulation size
        :param num_workers: int, number of workers for multiprocessing
        :param deterministic: bool, if True, then :
                                    - worker_init_fn will be specified for the data loaders
                                    - data won't be shuffled in the data loaders
                                    if False, then:
                                    - worker_init_fn will not be specified for the data loaders
                                    - data will be shuffled in the data loaders
        """
        self._num_workers = num_workers
        self._deterministic = deterministic

        # Calculate the effective batch size with regards to the gradient accumulation size
        self._batch_size_ga = int(batch_size / gradient_accumulation)

        # If the training dataset is not empty, declare the training data loader
        if len(dataset_manager.dataset_train) > 0:
            self._data_loader_train = self._get_data_loader(dataset_manager.dataset_train)
        else:
            self._data_loader_train = []

        # If the validation dataset is not empty, declare the validation data loader
        if len(dataset_manager.dataset_valid) > 0:
            self._data_loader_valid = self._get_data_loader(dataset_manager.dataset_valid)
        else:
            self._data_loader_valid = []

        # If the testing dataset is not empty, declare the testing data loader
        if len(dataset_manager.dataset_test) > 0:
            self._data_loader_test = self._get_data_loader(dataset_manager.dataset_test)
        else:
            self._data_loader_test = []

    @property
    def data_loader_train(self):
        return self._data_loader_train

    @property
    def data_loader_valid(self):
        return self._data_loader_valid

    @property
    def data_loader_test(self):
        return self._data_loader_test
