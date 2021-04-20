from src.data.DatasetManager import DatasetManager
from src.utils.seed import seed_worker
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
                 shuffle: bool = False) -> None:

        batch_size_ga = int(batch_size / gradient_accumulation)

        self.data_loader_train = DataLoader(dataset_manager.dataset_train,
                                            batch_size=batch_size_ga,
                                            shuffle=shuffle,
                                            num_workers=num_workers,
                                            collate_fn=self.collate_fn,
                                            worker_init_fn=seed_worker)

        self.data_loader_valid = DataLoader(dataset_manager.dataset_valid,
                                            batch_size=batch_size_ga,
                                            shuffle=shuffle,
                                            num_workers=num_workers,
                                            collate_fn=self.collate_fn,
                                            worker_init_fn=seed_worker)

        self.data_loader_test = DataLoader(dataset_manager.dataset_test,
                                           batch_size=batch_size_ga,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           collate_fn=self.collate_fn,
                                           worker_init_fn=seed_worker)

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
