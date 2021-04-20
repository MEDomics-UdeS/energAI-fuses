import numpy as np
import torchvision.transforms as transforms
import ray
from PIL import Image
from tqdm import trange

from src.data.FuseDataset import FuseDataset
from constants import MEAN, STD


class DatasetManager:
    """
    Dataset Manager class, handles the creation of the training, validation and testing datasets.
    """
    def __init__(self,
                 images_path,
                 annotations_path,
                 num_workers,
                 data_aug,
                 validation_size,
                 test_size,
                 mean_std) -> None:

        # Declare training, validation and testing datasets
        self.train_dataset = FuseDataset(images_path, annotations_path, num_workers)
        self.valid_dataset = FuseDataset()
        self.test_dataset = FuseDataset()

        total_size = len(self.train_dataset)

        self.train_dataset, self.valid_dataset = self.split_dataset(self.train_dataset, self.valid_dataset,
                                                                    validation_size, total_size)
        self.train_dataset, self.test_dataset = self.split_dataset(self.train_dataset, self.test_dataset,
                                                                   test_size, total_size)

        if mean_std:
            mean, std = self.calculate_mean_std(num_workers)
        else:
            mean, std = MEAN, STD

        self.train_dataset.transforms = self.train_transform(mean, std, data_aug)
        self.valid_dataset.transforms = self.base_transform(mean, std)
        self.test_dataset.transforms = self.base_transform(mean, std)

    @staticmethod
    def train_transform(mean, std, data_aug_value):
        transforms_list = [
            transforms.ColorJitter(brightness=data_aug_value,
                                   contrast=data_aug_value,
                                   saturation=data_aug_value,
                                   hue=data_aug_value),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        return transforms.Compose(transforms_list)

    @staticmethod
    def base_transform(mean, std):
        transforms_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        return transforms.Compose(transforms_list)

    @staticmethod
    def split_dataset(dataset_in, dataset_out, split, total_size):
        dataset_size = len(dataset_in)
        indices = list(range(dataset_size))
        split_idx = int(np.floor(split * total_size))
        np.random.shuffle(indices)
        indices = indices[0:split_idx]

        image_paths, images, targets = dataset_in.extract_data(idx_list=indices)
        dataset_out.add_data(image_paths, images, targets)

        return dataset_in, dataset_out

    def calculate_mean_std(self, num_workers):
        # Calculate dataset mean & std for normalization
        ray.init(include_dashboard=False)

        image_paths = self.train_dataset.image_paths

        ids = [ray_get_rgb.remote(image_paths, i) for i in range(num_workers)]
        size = len(image_paths)
        r = [None] * size
        g = [None] * size
        b = [None] * size
        nb_job_left = size - num_workers

        for _ in trange(size, desc='Getting RGB values of each pixel...'):
            ready, ids = ray.wait(ids, num_returns=1)
            r_val, g_val, b_val, idx = ray.get(ready)[0]
            r[idx] = r_val
            g[idx] = g_val
            b[idx] = b_val

            if nb_job_left > 0:
                ids.extend([ray_get_rgb.remote(image_paths, size - nb_job_left)])
                nb_job_left -= 1

        ray.shutdown()

        r = np.array(r)
        g = np.array(g)
        b = np.array(b)

        print('Calculating RGB means...')
        mean = (np.mean(r) / 255, np.mean(g) / 255, np.mean(b) / 255)

        print('Calculating RGB standard deviations...')
        std = (np.std(r) / 255, np.std(g) / 255, np.std(b) / 255)

        return mean, std


@ray.remote
def ray_get_rgb(image_paths, idx):
    image = Image.open(image_paths[idx]).convert("RGB")

    r = np.dstack(np.array(image)[:, :, 0])
    g = np.dstack(np.array(image)[:, :, 1])
    b = np.dstack(np.array(image)[:, :, 2])

    return r, g, b, idx
