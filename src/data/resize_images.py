from fuse_config import CLASS_DICT
import ray
import os
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import trange


def resize_images(max_image_size, num_workers,
                  root='data/raw/',
                  annotations_path='data/annotations/annotations_raw.csv'):
    ray.init(include_dashboard=False)

    imgs = sorted(os.listdir(root))
    imgs = [img for img in imgs if img.startswith('.') is False]
    image_paths = [os.path.join(root, img) for img in imgs]
    annotations = pd.read_csv(annotations_path)
    size = len(image_paths)
    resize_ratios = [None] * size
    box_lists = [None] * size

    ids = [parallelize.remote(image_paths, max_image_size, annotations, i)
           for i in range(num_workers)]

    nb_job_left = size - num_workers

    for _ in trange(size, desc='Resizing images...'):
        ready, not_ready = ray.wait(ids, num_returns=1)
        ids = not_ready
        result = ray.get(ready)[0]
        img, resize_ratio, idx, box_list = result#, target = result

        # self.imgs[idx] = img
        # self.targets[idx] = target
        resize_ratios[idx] = resize_ratio
        box_lists[idx] = box_list

        if nb_job_left > 0:
            idx = size - nb_job_left

            ids.extend([parallelize.remote(image_paths, max_image_size, annotations, idx)])
            nb_job_left -= 1

    ray.shutdown()

    idx = 0

    for i in range(len(box_lists)):
        for j in range(len(box_lists[i])):
            annotations.loc[idx, 'xmin'] = box_lists[i][j][0]
            annotations.loc[idx, 'ymin'] = box_lists[i][j][1]
            annotations.loc[idx, 'xmax'] = box_lists[i][j][2]
            annotations.loc[idx, 'ymax'] = box_lists[i][j][3]
            idx += 1

    annotations.to_csv('data/annotations/annotations_resized.csv', index=False)
    print('Resized images have been saved to:\t\tdata/resized/')
    print('Resized annotations have been saved to:\tdata/annotations/annotations_resized.csv')

    average_ratio = sum(resize_ratios) / len(resize_ratios)
    print(f'Average resize ratio : {average_ratio:.2%}')
    print(f'Maximum resize ratio : {max(resize_ratios):.2%}')
    print(f'Minimum resize ratio : {min(resize_ratios):.2%}')


@ray.remote
def parallelize(image_paths, max_image_size, annotations, idx, show_bounding_boxes=False):
    f = image_paths[idx].rsplit('/', 1)[-1].split(".")
    func = lambda x: x.split(".")[0]
    box_list = annotations.loc[annotations["filename"].apply(func) == f[0]][["xmin", "ymin", "xmax", "ymax"]].values

    label_array = annotations.loc[annotations["filename"].apply(func) == f[0]][["label"]].values
    label_list = []

    for label in label_array:
        label_list.append(class_dictionary[str(label[0])])

    num_objs = len(box_list)

    name_original = image_paths[idx].split("/")[-1]

    #if max_image_size > 0:
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

    img2.save(f'data/resized/{name_original}')
    # else:
    #     resize_ratio = 1
    #
    # image_id = torch.tensor([idx])
    # boxes = torch.as_tensor(np.array(box_list), dtype=torch.float32)
    # labels = torch.as_tensor(label_list, dtype=torch.long)
    #
    # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    # area = torch.as_tensor(area, dtype=torch.float16)
    # iscrowd = torch.zeros((num_objs,), dtype=torch.int8)
    # targets = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}
    #
    # name = [ord(c) for c in name_original]
    # targets["name"] = torch.tensor(name)
    #
    return img2, resize_ratio, idx, box_list#, targets
