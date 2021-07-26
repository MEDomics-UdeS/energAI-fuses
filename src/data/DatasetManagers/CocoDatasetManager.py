from torch.utils.data import DataLoader
from src.data.Datasets.CocoDataset import CocoDataset


class CocoDatasetManager:
    
    def __init__(self, data_loader: DataLoader) -> None:
        
        self.__datasets = []
        
        ds = data_loader.dataset
        
        for i in range(len(ds)):        
            self.__datasets.append(CocoDataset(ds.images[i], ds.targets[i], ds.image_paths[i]))

    @property
    def datasets(self):
        return self.__datasets
    
    def __len__(self):
        return len(self.__datasets)
