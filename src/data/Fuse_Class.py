import torch
import torch.utils.data
import os
from tqdm import trange
import json
from PIL import Image
import ray


class FuseDataset(torch.utils.data.Dataset):
    def __init__(self, root=None, targets_path=None, num_workers=None):
        if root is not None:
            ray.init(include_dashboard=False)

            images = [img for img in sorted(os.listdir(root)) if img.startswith('.') is False]
            self.image_paths = [os.path.join(root, img) for img in images]
            size = len(self.image_paths)
            self.images = [None] * size

            ids = [ray_load_images.remote(self.image_paths, i) for i in range(num_workers)]

            nb_job_left = size - num_workers

            for _ in trange(size, desc='Loading images to RAM...'):
                ready, ids = ray.wait(ids, num_returns=1)
                image, idx = ray.get(ready)[0]
                self.images[idx] = image

                if nb_job_left > 0:
                    ids.extend([ray_load_images.remote(self.image_paths, size - nb_job_left)])
                    nb_job_left -= 1

            ray.shutdown()
        else:
            self.image_paths = []
            self.images = []

        if targets_path is not None:
            self.targets = json.load(open(targets_path))

            for target in self.targets:
                for key, value in target.items():
                    target[key] = torch.as_tensor(value, dtype=torch.int64)
        else:
            self.targets = []

    def __getitem__(self, idx):
        return self.transforms(self.images[idx]), self.targets[idx]

    def __len__(self):
        return len(self.images)

    def extract_data(self, idx_list):
        idx_list = sorted(idx_list, reverse=True)
        images = []
        targets = []

        for idx in idx_list:
            images.append(self.images.pop(idx))
            targets.append(self.targets.pop(idx))

        return images, targets

    def add_data(self, images, targets):
        self.images.extend(images)
        self.targets.extend(targets)


@ray.remote
def ray_load_images(image_paths, idx):
    return Image.open(image_paths[idx]).convert("RGB"), idx
