from src.data.DatasetManager import DatasetManager
from torch.utils.data import DataLoader
from src.utils.seed import seed_worker


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

        self.train_data_loader = DataLoader(dataset_manager.train_dataset,
                                            batch_size=batch_size_ga,
                                            shuffle=shuffle,
                                            num_workers=num_workers,
                                            collate_fn=self.collate_fn,
                                            worker_init_fn=seed_worker)

        self.valid_data_loader = DataLoader(dataset_manager.valid_dataset,
                                            batch_size=batch_size_ga,
                                            shuffle=shuffle,
                                            num_workers=num_workers,
                                            collate_fn=self.collate_fn,
                                            worker_init_fn=seed_worker)

        self.test_data_loader = DataLoader(dataset_manager.test_dataset,
                                           batch_size=batch_size_ga,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           collate_fn=self.collate_fn,
                                           worker_init_fn=seed_worker)

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
