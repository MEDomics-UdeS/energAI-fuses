"""
File:
    src/data/DataLoaderManager.py

Authors:
    - Simon Giard-Leroux
    - Shreyas Sunil Kulkarni

Description:
    Generate data loaders for batch management and multiprocessing during training, validation and testing.
"""

from src.data.DatasetManager import DatasetManager
from src.utils.reproducibility import seed_worker
from torch.utils.data import DataLoader


class DataLoaderManager:
    """
    Data Loader Manager class, handles the creation of the training, validation and testing data loaders.
    """
    def __init__(self,
                 dataset_manager: DatasetManager,
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
        # Calculate the effective batch size with regards to the gradient accumulation size
        batch_size_ga = int(batch_size / gradient_accumulation)

        # If the training dataset is not empty, declare the training data loader
        if len(dataset_manager.dataset_train) > 0:
            self.data_loader_train = DataLoader(dataset_manager.dataset_train,
                                                batch_size=batch_size_ga,
                                                shuffle=not deterministic,
                                                num_workers=num_workers,
                                                collate_fn=self.__collate_fn,
                                                worker_init_fn=seed_worker if deterministic else None)

        # If the validation dataset is not empty, declare the validation data loader
        if len(dataset_manager.dataset_valid) > 0:
            self.data_loader_valid = DataLoader(dataset_manager.dataset_valid,
                                                batch_size=batch_size_ga,
                                                shuffle=not deterministic,
                                                num_workers=num_workers,
                                                collate_fn=self.__collate_fn,
                                                worker_init_fn=seed_worker if deterministic else None)

        # If the testing dataset is not empty, declare the testing data loader
        if len(dataset_manager.dataset_test) > 0:
            self.data_loader_test = DataLoader(dataset_manager.dataset_test,
                                               batch_size=batch_size_ga,
                                               shuffle=not deterministic,
                                               num_workers=num_workers,
                                               collate_fn=self.__collate_fn,
                                               worker_init_fn=seed_worker if deterministic else None)

    @staticmethod
    def __collate_fn(batch: list) -> tuple:
        """
        Custom batching collation function.

        :param batch: list, containing the current batch
        :return: tuple
        """
        return tuple(zip(*batch))
