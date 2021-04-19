import torch
import torch.utils.data
import os
import pandas as pd
from fuse_config import *
from tqdm import tqdm
import json
from PIL import Image


class FuseDataset(torch.utils.data.Dataset):
    def __init__(self, root=None, targets_path=None):
        if root is not None:
            images = [img for img in sorted(os.listdir(root)) if img.startswith('.') is False]
            image_paths = [os.path.join(root, img) for img in images]
            self.images = [Image.open(image_path).convert("RGB")
                           for image_path in tqdm(image_paths, desc='Loading images to RAM...')]
        else:
            self.images = []

        if targets_path is not None:
            self.targets = json.load(open(targets_path))

            for target in self.targets:
                for key, value in target.items():
                    target[key] = torch.as_tensor(value, dtype=torch.int64)
        else:
            self.targets = []

        # self.targets = torch.as_tensor(json_list, dtype=self.dtype)

        # df = pd.read_csv(self.annotations_path)
        #
        # func = lambda x: x.split(".")[0]
        #
        # self.targets = []
        #
        # for i, img in enumerate(tqdm(self.imgs)):
        #     f = img.split(".")
        #     box_list = df.loc[df["filename"].apply(func) == f[0]][["xmin", "ymin", "xmax", "ymax"]].values
        #     label_array = df.loc[df["filename"].apply(func) == f[0]][["label"]].values
        #
        #     label_list = []
        #
        #     for label in label_array:
        #         label_list.append(class_dictionary[str(label[0])])
        #
        #     image_id = i #torch.as_tensor(i, dtype=torch.int16)
        #     boxes = list(box_list.tolist()) #torch.as_tensor(box_list, dtype=torch.float32)
        #     num_objs = len(boxes)
        #
        #     labels = label_list # torch.as_tensor(label_list, dtype=torch.long)
        #
        #     area = list((box_list[:, 3] - box_list[:, 1]) * (box_list[:, 2] - box_list[:, 0]))
        #     area = [int(a) for a in area]
        #     # area = torch.as_tensor(area, dtype=torch.float32)
        #     iscrowd = [0] * num_objs # torch.zeros((num_objs,), dtype=torch.int64)
        #
        #     # names = [image_path.split("/")[-1] for image_path in self.image_paths]
        #     # for name in names:
        #     #     name = [ord(char) for char in name]
        #     # name = torch.tensor(name)
        #     self.targets.append({"boxes": boxes,
        #                          "labels": labels,
        #                          "image_id": image_id,
        #                          "area": area,
        #                          "iscrowd": iscrowd})
        #                          #"name": name})
        #
        # json.dump(self.targets, open(f'myjson.json', 'w'), ensure_ascii=False)
        # print('hi')

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
