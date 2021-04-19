import torch
import torch.utils.data
import os
import pandas as pd
from fuse_config import class_dictionary
from tqdm import tqdm
import json
from PIL import Image


class FuseDataset(torch.utils.data.Dataset):
    def __init__(self, root, targets_path, transforms):
        self.transforms = transforms

        if root is not None:
            images = [img for img in sorted(os.listdir(root)) if img.startswith('.') is False]
            self.image_paths = [os.path.join(root, img) for img in images]
        else:
            self.image_paths = []

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
        return self.transforms(Image.open(self.image_paths[idx]).convert("RGB")), self.targets[idx]

    def __len__(self):
        return len(self.image_paths)

    def extract_data(self, idx_list):
        idx_list = sorted(idx_list, reverse=True)
        image_paths = []
        targets = []

        for idx in idx_list:
            image_paths.append(self.image_paths.pop(idx))
            targets.append(self.targets.pop(idx))

        return image_paths, targets

    def add_data(self, image_paths, targets):
        self.image_paths.extend(image_paths)
        self.targets.extend(targets)
