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

from src.gui.GuiDataset import GuiDataset
from src.utils.reproducibility import seed_worker
from torch.utils.data import DataLoader


class GuiDataLoader:
    """
    Data Loader Manager class, handles the creation of the training, validation and testing data loaders.
    """
    def __init__(self,
                 dataset: GuiDataset,
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
        self.__num_workers = num_workers
        self.__deterministic = deterministic

        # Calculate the effective batch size with regards to the gradient accumulation size
        self.__batch_size_ga = int(batch_size / gradient_accumulation)

        # If the training dataset is not empty, declare the training data loader
        if len(dataset) > 0:
            self.__data_loader = self.__get_data_loader(dataset)

    @property
    def data_loader(self):
        return self.__data_loader


    def __get_data_loader(self, dataset: GuiDataset) -> DataLoader:
        return DataLoader(dataset,
                          batch_size=self.__batch_size_ga,
                          shuffle=not self.__deterministic,
                          num_workers=self.__num_workers,
                          collate_fn=self.__collate_fn,
                          worker_init_fn=seed_worker if self.__deterministic else None)

    @staticmethod
    def __collate_fn(batch: list) -> tuple:
        """
        Custom batching collation function.

        :param batch: list, containing the current batch
        :return: tuple
        """
        return tuple(zip(*batch))
