from torch.utils.data import DataLoader
from src.data.Datasets.FuseDataset import FuseDataset
from src.data.Datasets.CocoDataset import CocoDataset


class CocoDatasetManager:
    
    def __init__(self, ds) -> None:
        
        self.__datasets = []

        for i in range(len(ds)):
            self.__datasets.append(CocoDataset(image=ds[i][0], target=ds[i][1], path=ds.image_paths[i]))

    @property
    def datasets(self):
        return self.__datasets
    
    def __len__(self):
        return len(self.__datasets)
