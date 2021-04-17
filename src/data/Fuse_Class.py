import torch
import torch.utils.data
import os
from PIL import Image, ImageDraw
import pandas as pd
import ray
import numpy as np

from fuse_config import class_dictionary, NUM_WORKERS_RAY

"""
The Dataset class
used to build the dataset
- init => initializes the path, gets the list of images, and also gets the annotation file
- get_item => retrieves the image,label, and bounding boxes and converts it to a tensor - this value is returned during 
  training/testing
- len - length of the image

optional
- get_image => retrieves the image_path 
- draw_img => displays the image
"""


class FuseDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_file, max_image_size, transforms=None, num_workers=NUM_WORKERS_RAY, save=True):
        self.transforms = transforms
        image_paths = sorted(os.listdir(root)) if root else []
        image_paths = [path for path in image_paths if path.startswith('.') is False]

        annotations = pd.read_csv(data_file)

        size = len(image_paths)

        self.targets = [None] * size
        self.imgs = [None] * size
        resize_ratios = [None] * size
        box_lists = [None] * size

        if size > 0:
            image_paths = [os.path.join(root, path) for path in image_paths]

            ids = [parallelize.remote(image_paths, max_image_size, annotations, i, save)
                   for i in range(num_workers)]

            nb_job_left = size - num_workers

            for _ in range(size):
                ready, not_ready = ray.wait(ids, num_returns=1)
                ids = not_ready
                result = ray.get(ready)[0]
                img, resize_ratio, idx, box_list, target = result

                self.imgs[idx] = img
                self.targets[idx] = target
                resize_ratios[idx] = resize_ratio
                box_lists[idx] = box_list

                if nb_job_left > 0:
                    idx = size - nb_job_left

                    ids.extend([parallelize.remote(image_paths, max_image_size, annotations, idx, save)])
                    nb_job_left -= 1

            average_ratio = sum(resize_ratios) / len(resize_ratios)
            print(f'Average resize ratio : {average_ratio:.2%}')
            print(f'Maximum resize ratio : {max(resize_ratios):.2%}')
            print(f'Minimum resize ratio : {min(resize_ratios):.2%}')

            if save:
                idx = 0
                for i in range(len(box_lists)):
                    for j in range(len(box_lists[i])):
                        annotations.loc[idx, 'xmin'] = box_lists[i][j][0]
                        annotations.loc[idx, 'ymin'] = box_lists[i][j][1]
                        annotations.loc[idx, 'xmax'] = box_lists[i][j][2]
                        annotations.loc[idx, 'ymax'] = box_lists[i][j][3]
                        idx += 1

                annotations.to_csv('/data/annotations/annotation_processed.csv')
                print('Resized images and annotations have been saved here : /data/annotations/')

    def __getitem__(self, idx):
        if self.transforms is not None:
            return self.transforms(self.imgs[idx]), self.targets[idx]
        return self.imgs[idx], self.targets[idx]

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


@ray.remote
def parallelize(image_paths, max_image_size, annotations, idx, save):
    f = image_paths[idx].rsplit('/', 1)[-1].split(".")
    func = lambda x: x.split(".")[0]
    box_list = annotations.loc[annotations["filename"].apply(func) == f[0]][["xmin", "ymin", "xmax", "ymax"]].values

    label_array = annotations.loc[annotations["filename"].apply(func) == f[0]][["label"]].values
    label_list = []

    for label in label_array:
        label_list.append(class_dictionary[str(label[0])])

    num_objs = len(box_list)

    name_original = image_paths[idx].split("/")[-1]

    img = Image.open(image_paths[idx]).convert("RGB")
    original_size = img.size

    show_bounding_boxes = False

    img2 = Image.new('RGB', (max_image_size, max_image_size), (255, 255, 255))

    resize_ratio = (img2.size[0] * img2.size[1]) / (original_size[0] * original_size[1])

    if max_image_size < original_size[0] or max_image_size < original_size[1]:
        img.thumbnail((max_image_size, max_image_size),
                      resample=Image.BILINEAR,
                      reducing_gap=2)

        downsize_ratio = img.size[0] / original_size[0]
    else:
        downsize_ratio = 1

    x_offset = int((max_image_size - img.size[0]) / 2)
    y_offset = int((max_image_size - img.size[1]) / 2)
    img2.paste(img, (x_offset, y_offset, x_offset + img.size[0], y_offset + img.size[1]))

    if show_bounding_boxes:
        draw = ImageDraw.Draw(img2)

    for i in range(num_objs):
        for j in range(4):
            box_list[i][j] = int(box_list[i][j] * downsize_ratio)

            if j == 0 or j == 2:
                box_list[i][j] += x_offset
            else:
                box_list[i][j] += y_offset

        if show_bounding_boxes:
            draw.rectangle([(box_list[i][0], box_list[i][1]), (box_list[i][2], box_list[i][3])],
                           outline="red", width=5)

    if save:
        img2.save(f'/data/processed/{name_original}')

    image_id = torch.tensor([idx])
    boxes = torch.as_tensor(np.array(box_list), dtype=torch.float32)
    labels = torch.as_tensor(label_list, dtype=torch.long)

    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    area = torch.as_tensor(area, dtype=torch.float16)
    iscrowd = torch.zeros((num_objs,), dtype=torch.int8)
    targets = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

    name = [ord(c) for c in name_original]
    targets["name"] = torch.tensor(name)

    return img2, resize_ratio, idx, box_list, targets
