"""
File:
    src/visualization/inference.py

Authors:
    - Simon Giard-Leroux
    - Shreyas Sunil Kulkarni

Description:
    Inference functions to test a saved model and show model predictions and ground truths boxes on the images
"""

import torch
from PIL import ImageDraw, ImageFont
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Tuple

from src.utils.constants import CLASS_DICT, FONT_PATH
from src.utils.helper_functions import filter_by_nms, filter_by_score


def save_test_images(model_file_name: str, data_loader: DataLoader, iou_threshold: float, save_path: str) -> None:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    pbar = tqdm(total=len(data_loader), leave=False, desc='Inference Test')

    model = torch.load(f'models/{model_file_name}')
    model.eval()

    font = ImageFont.truetype(FONT_PATH, 12)

    # Deactivate the autograd engine
    with torch.no_grad():
        for batch_no, (images, targets) in enumerate(data_loader):
            indices = range(data_loader.batch_size * batch_no,
                            min(data_loader.batch_size * (batch_no + 1), len(data_loader.dataset)))

            images = torch.stack(images).to(device)

            preds = model(images)
            preds = filter_by_nms(preds, iou_threshold)
            preds = filter_by_score(preds, iou_threshold)

            images = [data_loader.dataset.load_image(index) for index in indices]

            for index, image, target, pred in zip(indices, images, targets, preds):
                draw = ImageDraw.Draw(image)
                draw_boxes(draw, target, 'green', 3, font, (255, 255, 0, 0))
                draw_boxes(draw, pred, 'red', 3, font, (255, 255, 255, 0))

                image.save(f'{save_path}'
                           f'{data_loader.dataset.image_paths[index].rsplit("/", 1)[-1].split(".", 1)[0]}.png')

            pbar.update()

    pbar.close()


def draw_boxes(draw: ImageDraw.ImageDraw, box_dict: dict, outline_color: str, outline_width: int,
               font: ImageFont.ImageFont, font_color: Tuple[int, int, int, int]) -> None:
    boxes = box_dict['boxes'].tolist()
    labels = [list(CLASS_DICT.keys())[list(CLASS_DICT.values()).index(label)]
              for label in box_dict['labels'].tolist()]

    if 'scores' in box_dict:
        scores = box_dict['scores'].tolist()
    else:
        scores = [1] * len(labels)

    for box, label, score in zip(boxes, labels, scores):
        draw.rectangle(box, outline=outline_color, width=outline_width)
        draw.text((box[0], box[1]), text=f'{label} {score:.4f}', font=font, fill=font_color)
