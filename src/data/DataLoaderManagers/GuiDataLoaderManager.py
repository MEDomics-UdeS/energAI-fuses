"""
File:
    src/data/DataLoaderManagers/GuiDataLoaderManager.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Contains the GuiDataLoaderManager, DataLoaderManager for GUI inference tool.
"""

from src.data.DatasetManagers.GuiDatasetManager import GuiDatasetManager
from src.data.DataLoaderManagers.CustomDataLoaderManager import CustomDataLoaderManager


class GuiDataLoaderManager(CustomDataLoaderManager):
    """
    Data Loader Manager class, handles the creation of the training, validation and testing data loaders.
    """
    def __init__(self,
                 dataset: GuiDatasetManager,
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
        if len(dataset.dataset) > 0:
            self._data_loader_test = self._get_data_loader(dataset.dataset)

    @property
    def data_loader_test(self):
        return self._data_loader_test
