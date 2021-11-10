"""
File:
    src/data/DatasetManagers/CocoDatasetManager.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Contains the CocoDatasetManager, DatasetManager for COCO evaluation.
"""

from src.data.Datasets.CocoDataset import CocoDataset


class CocoDatasetManager:
    """Dataset Manager class, handles the creation of the training, validation and testing datasets."""
    def __init__(self,
                 ds) -> None:
        """

        Args:
            ds: 

        """
        self.__datasets = []

        for i in range(len(ds)):
            self.__datasets.append(CocoDataset(image=ds[i][0], target=ds[i][1], path=ds.image_paths[i]))

    @property
    def datasets(self):
        """ """
        return self.__datasets
    
    def __len__(self):
        """ """
        return len(self.__datasets)
