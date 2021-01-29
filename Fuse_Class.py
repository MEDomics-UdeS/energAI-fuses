import torch
import torch.utils.data
import torchvision
import os
from PIL import Image, ImageDraw
import cv2
from pycocotools.coco import COCO
import pandas as pd
import numpy as np
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
  
    def __init__(self, root, data_file, max_image_size, resize, transforms=None):
      self.root = root
      self.transforms = transforms
      self.imgs = sorted(os.listdir(os.path.join(root, "images")))
      self.path_to_data_file = data_file
      self.max_image_size = max_image_size
      self.resize = resize

    def __getitem__(self, idx):
      img_path = os.path.join(self.root, "images", self.imgs[idx])
      img = Image.open(img_path).convert("RGB")

      box_list,label_list = parse_annotations(self.path_to_data_file, self.imgs[idx])

      num_objs = len(box_list)

      # ==========================================================
      # SGL 2021-01-22
      # Image Resize
      # ==========================================================
      if self.resize:
        if img.size[0] > self.max_image_size or img.size[1] > self.max_image_size:
          original_size = img.size
          img.thumbnail((self.max_image_size, self.max_image_size),
                        resample=Image.BILINEAR,
                        reducing_gap=2)
          resize_ratio = img.size[0] / original_size[0]

          # draw = ImageDraw.Draw(img)

          for i in range(num_objs):
            for j in range(4):
              box_list[i][j] = int(box_list[i][j] * resize_ratio)

            # draw.rectangle([(box_list[i][0], box_list[i][1]), (box_list[i][2], box_list[i][3])], outline ="red", width=5)

          # img.show()

      # ==========================================================

      image_id = torch.tensor([idx])
      boxes = torch.as_tensor(box_list, dtype=torch.float32)
      labels = torch.as_tensor(label_list, dtype=torch.long)

      area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
      area = torch.as_tensor(area, dtype=torch.float16) #float32
      iscrowd = torch.zeros((num_objs,), dtype=torch.int8) #int64
      target  = {}
      target ["boxes"] = boxes
      target ["labels"] = labels
      target ["image_id"] = image_id
      target ["area"] = area
      target ["iscrowd"] = iscrowd
      name = img_path.split("/")[-1]
      name = [ord(c) for c in name]
      target ["name"] = torch.tensor(name)

      if self.transforms is not None:
        img = self.transforms(img)

      return img, target

    # def get_image(self,idx):
    #   img_path = os.path.join(self.root, "images", self.imgs[idx])
    #   img = cv2.imread(img_path,0)
    #   img_new = Image.fromarray(img)
    #   return img_path
    #
    # def draw_img(self,idx):
    #   img_path = os.path.join(self.root, "images", self.imgs[idx])
    #   img = Image.open(img_path).convert("RGB")
    #   img.show()

    def __len__(self):
      return len(self.imgs)


"""
converts the text_labels to a numerical value
returns the bounding boxes to the dataset class
"""
def parse_annotations(path_to_data_file, filename):
  df = pd.read_csv(path_to_data_file)
  f = filename.split(".")
  func = lambda x: x.split(".")[0]
  boxes_array = df.loc[df["filename"].apply(func) == f[0]][["xmin", "ymin","xmax", "ymax"]].values
  label_array = df.loc[df["filename"].apply(func) == f[0]][["label"]].values
  label = []
  for l in label_array:
    label.append(class_dictionary[str(l[0])])
  return boxes_array,label
