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

        batch_size_ga = int(batch_size / gradient_accumulation)

        self.data_loader_train = DataLoader(dataset_manager.dataset_train,
                                            batch_size=batch_size_ga,
                                            shuffle=not deterministic,
                                            num_workers=num_workers,
                                            collate_fn=self.collate_fn,
                                            worker_init_fn=seed_worker if deterministic else None)

        self.data_loader_valid = DataLoader(dataset_manager.dataset_valid,
                                            batch_size=batch_size_ga,
                                            shuffle=not deterministic,
                                            num_workers=num_workers,
                                            collate_fn=self.collate_fn,
                                            worker_init_fn=seed_worker if deterministic else None)

        self.data_loader_test = DataLoader(dataset_manager.dataset_test,
                                           batch_size=batch_size_ga,
                                           shuffle=not deterministic,
                                           num_workers=num_workers,
                                           collate_fn=self.collate_fn,
                                           worker_init_fn=seed_worker if deterministic else None)

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
