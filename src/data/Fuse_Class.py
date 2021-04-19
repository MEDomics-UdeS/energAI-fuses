import torch
import torch.utils.data
import os
from PIL import Image
import cv2
import pandas as pd
from fuse_config import class_dictionary

"""
The Dataset class
used to build the dataset
- init => initializes the path, gets the list of images, and also gets the annotation file
- get_item => retrieves the image,label, and bounding boxes and converts it to a tensor - this value is returned during training/testing
- len - length of the image
optional
- get_image => retrieves the image_path 
- draw_img => displays the image
"""


class FuseDataset(torch.utils.data.Dataset):

    def __init__(self, root, data_file, transforms=None):
        self.root = root
        self.transforms = transforms
        imgs = sorted(os.listdir(self.root))
        self.imgs = [img for img in imgs if img.startswith('.') is False]
        self.path_to_data_file = data_file

        self.image_paths = [os.path.join(root, img) for img in imgs]
        img_path = os.path.join(self.root, self.imgs)
        # img = Image.open(img_path).convert("RGB")
        # save_img = img
        df = pd.read_csv(data_file)
        f = img_path.split(".")
        func = lambda x: x.split(".")[0]
        box_list = df.loc[df["filename"].apply(func) == f[0]][["xmin", "ymin", "xmax", "ymax"]].values
        label_array = df.loc[df["filename"].apply(func) == f[0]][["label"]].values
        label_list = []
        for l in label_array:
            label_list.append(class_dictionary[str(l[0])])

        image_id = torch.tensor(0)
        boxes = torch.as_tensor(box_list, dtype=torch.float32)
        num_objs = len(box_list)

        labels = torch.as_tensor(label_list, dtype=torch.long)

        # draw = ImageDraw.Draw(img)
        # for i in range(num_objs):
        #   draw.rectangle([(boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3])],outline ="red",width=5)

        # display(img)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        self.targets = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}
        name = img_path.split("/")[-1]
        name = [ord(c) for c in name]
        self.targets["name"] = torch.tensor(name)

    def __getitem__(self, idx):
        if self.transforms is not None:
            return self.transforms(self.imgs[idx]), self.targets[idx]
        return self.imgs[idx], self.targets[idx]

    # def get_image(self, idx):
    #     img_path = os.path.join(self.root, "images", self.imgs[idx])
    #     img = cv2.imread(img_path, 0)
    #     # img_new = Image.fromarray(img)
    #     return img_path

    # def draw_img(self, idx):
    #     img_path = os.path.join(self.root, "images", self.imgs[idx])
    #     img = Image.open(img_path).convert("RGB")
    #     img.show()

    def __len__(self):
        return len(self.imgs)

    def extract_data(self, idx_list):
        idx_list = sorted(idx_list, reverse=True)
        imgs = []
        targets = []

        for idx in idx_list:
            imgs.append(self.imgs.pop(idx))
            targets.append(self.targets.pop(idx))

        return imgs, targets

    def add_data(self, imgs, targets):
        self.imgs.extend(imgs)
        self.targets.extend(targets)
