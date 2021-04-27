import torch
import torch.utils.data
import os
from tqdm import trange
import json
from PIL import Image
import ray
from typing import Tuple, List


class FuseDataset(torch.utils.data.Dataset):
    def __init__(self, root: str = None, targets_path: str = None, num_workers: int = None) -> None:
        if root is not None:
            ray.init(include_dashboard=False)

            images = [img for img in sorted(os.listdir(root)) if img.startswith('.') is False]
            self.image_paths = [os.path.join(root, img) for img in images]
            size = len(self.image_paths)
            self.images = [None] * size

            ids = [ray_load_images.remote(self.image_paths, i) for i in range(num_workers)]

            nb_job_left = size - num_workers

            for _ in trange(size, desc='Loading images to RAM', leave=False):
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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:
        return self.transforms(self.images[index]), self.targets[index]

    def __len__(self) -> int:
        return len(self.images)

    def load_image(self, index: int) -> Image:
        image_path = self.image_paths[index]
        img = Image.open(image_path)
        return img

    def extract_data(self, index_list: List[int]) -> Tuple[List[str], List[Image.Image], List[dict]]:
        index_list = sorted(index_list, reverse=True)
        image_paths = []
        images = []
        targets = []

        for index in index_list:
            image_paths.append(self.image_paths.pop(index))
            images.append(self.images.pop(index))
            targets.append(self.targets.pop(index))

        return image_paths, images, targets

    def add_data(self, image_paths: List[str], images: List[Image.Image], targets: List[dict]) -> None:
        self.image_paths.extend(image_paths)
        self.images.extend(images)
        self.targets.extend(targets)


@ray.remote
def ray_load_images(image_paths: List[str], index: int) -> Tuple[Image.Image, int]:
    return Image.open(image_paths[index]).convert("RGB"), index
