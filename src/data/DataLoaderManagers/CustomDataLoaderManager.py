"""
File:
    src/data/DataLoaderManagers/CustomDataLoaderManager.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Contains the CustomDataLoaderManager class, parent class for all other DataLoaderManager classes.
"""

from abc import ABC
from src.data.Datasets.CustomDataset import CustomDataset
from src.utils.reproducibility import seed_worker
from torch.utils.data import DataLoader


class CustomDataLoaderManager(ABC):
    """Parent Data Loader Manager class, handles the creation of the training, validation and testing data loaders."""
    def _get_data_loader(self,
                         dataset: CustomDataset) -> DataLoader:
        """

        Args:
            dataset(CustomDataset): 

        Returns:

        """
        return DataLoader(dataset,
                          batch_size=self._batch_size_ga,
                          shuffle=not self._deterministic,
                          num_workers=self._num_workers,
                          collate_fn=self._collate_fn,
                          worker_init_fn=seed_worker if self._deterministic else None)

    @staticmethod
    def _collate_fn(batch: list) -> tuple:
        """Custom batching collation function.

        Args:
            batch(list): list containing the current batch

        Returns:
            tuple: tuple

        """
        return tuple(zip(*batch))
