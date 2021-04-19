from fuse_config import CLASS_DICT
import ray
import os
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import trange
import json


def resize_images(max_image_size, num_workers,
                  root='data/raw/',
                  annotations_path='data/annotations/annotations_raw.csv'):
    ray.init(include_dashboard=False)

    imgs = [img for img in sorted(os.listdir(root)) if img.startswith('.') is False]
    image_paths = [os.path.join(root, img) for img in imgs]
    annotations = pd.read_csv(annotations_path)

    size = len(image_paths)
    resize_ratios = [None] * size
    targets_list = [None] * size

    ids = [ray_resize_images.remote(image_paths, max_image_size, annotations, i) for i in range(num_workers)]

    nb_job_left = size - num_workers

    for _ in trange(size, desc='Resizing images...'):
        ready, ids = ray.wait(ids, num_returns=1)
        resize_ratio, idx, box_list, targets = ray.get(ready)[0]
        resize_ratios[idx] = resize_ratio
        targets_list[idx] = targets

        if nb_job_left > 0:
            ids.extend([ray_resize_images.remote(image_paths, max_image_size, annotations, size - nb_job_left)])
            nb_job_left -= 1

    ray.shutdown()

    average_ratio = sum(resize_ratios) / len(resize_ratios)
    print(f'Average resize ratio : {average_ratio:.2%}')
    print(f'Maximum resize ratio : {max(resize_ratios):.2%}')
    print(f'Minimum resize ratio : {min(resize_ratios):.2%}')

    json.dump(targets_list, open('data/annotations/targets_resized.json', 'w'), ensure_ascii=False)
    print('Resized images have been saved to:\t\tdata/resized/')
    print('Resized targets have been saved to:\t\tdata/annotations/targets_resized.json')


@ray.remote
def ray_resize_images(image_paths, max_image_size, annotations, idx, show_bounding_boxes=False):
    f = image_paths[idx].rsplit('/', 1)[-1].split(".")
    func = lambda x: x.split(".")[0]

    box_array = annotations.loc[annotations["filename"].apply(func) == f[0]][["xmin", "ymin", "xmax", "ymax"]].values
    label_array = annotations.loc[annotations["filename"].apply(func) == f[0]][["label"]].values

    label_list = []

    for label in label_array:
        label_list.append(CLASS_DICT[str(label[0])])

    num_boxes = len(box_array)

    img = Image.open(image_paths[idx]).convert("RGB")
    original_size = img.size

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

    for i in range(num_boxes):
        for j in range(4):
            box_array[i][j] = int(box_array[i][j] * downsize_ratio)

            if j == 0 or j == 2:
                box_array[i][j] += x_offset
            else:
                box_array[i][j] += y_offset

        if show_bounding_boxes:
            draw.rectangle([(box_array[i][0], box_array[i][1]), (box_array[i][2], box_array[i][3])],
                           outline="red", width=5)

    img2.save(f'data/resized/{image_paths[idx].split("/")[-1]}')

    area = [int(a) for a in list((box_array[:, 3] - box_array[:, 1]) * (box_array[:, 2] - box_array[:, 0]))]

    targets = {"boxes": list(box_array.tolist()),
               "labels": label_list,
               "image_id": idx,
               "area": area,
               "iscrowd": [0] * num_boxes}

    return resize_ratio, idx, box_array, targets
