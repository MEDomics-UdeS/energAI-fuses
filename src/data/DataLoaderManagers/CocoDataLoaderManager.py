"""
File:
    src/data/DataLoaderManagers/CocoDataLoaderManager.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Contains the CocoDataLoaderManager class, DataLoaderManager for COCO evaluation.
"""

from src.data.DatasetManagers.CocoDatasetManager import CocoDatasetManager
from src.data.DataLoaderManagers.CustomDataLoaderManager import CustomDataLoaderManager


class CocoDataLoaderManager(CustomDataLoaderManager):
    """
    Data Loader Manager class, handles the creation of the training, validation and testing data loaders.
    """
    def __init__(self,
                 dataset: CocoDatasetManager,
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

        self._data_loaders = []
        
        if len(dataset) > 0:
            for ds in dataset.datasets:
                self._data_loaders.append(self._get_data_loader(ds))

    @property
    def data_loaders(self):
        return self._data_loaders
