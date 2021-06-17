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
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Tuple

from src.utils.constants import CLASS_DICT, FONT_PATH, MODELS_PATH, RESIZED_PATH, RAW_PATH, IMAGE_EXT
from src.utils.helper_functions import filter_by_nms, filter_by_score


def save_test_images(model_file_name: str,
                     data_loader: DataLoader,
                     iou_threshold: float,
                     score_threshold: float,
                     save_path: str,
                     image_size: int) -> None:
    """
    Main inference testing function to save images with predicted and ground truth bounding boxes

    :param score_threshold:
    :param model_file_name: str, model file name to load
    :param data_loader: DataLoader, data loader object
    :param iou_threshold: float, intersection-over-union threshold for predicted bounding boxes filtering
    :param save_path: str, save path for the inference test images
    """
    image_paths_raw = [image_path.replace(RESIZED_PATH, RAW_PATH) for image_path in data_loader.dataset.image_paths]

    # Declare device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Declare progress bar
    pbar = tqdm(total=len(data_loader), leave=False, desc='Inference Test')

    # Load saved model
    model = torch.load(f'{MODELS_PATH}{model_file_name}')

    model.to(device)

    # Put the model into eval() mode for inference
    model.eval()

    # Declare the font object to write the confidence scores on the images
    font = ImageFont.truetype(FONT_PATH, 12)

    # Deactivate the autograd engine
    with torch.no_grad():
        # Loop through each batch in the data loader
        for batch_no, (images, targets) in enumerate(data_loader):
            # Get current batch indices
            indices = range(data_loader.batch_size * batch_no,
                            min(data_loader.batch_size * (batch_no + 1), len(data_loader.dataset)))

            # Send images to the device
            images = torch.stack(images).to(device)

            # Perform a forward pass and obtain bounding boxes predictions
            preds = model(images)

            # Filter predicted bounding boxes by using non-maximum suppression
            preds = filter_by_nms(preds, iou_threshold)

            # Filter predicted bounding boxes by using a confidence score threshold
            preds = filter_by_score(preds, score_threshold)

            # Load images in the current batch
            images_raw = [Image.open(image_paths_raw[index]) for index in indices]

            # Loop through each image in the current batch
            for index, image, image_raw, target, pred in zip(indices, images, images_raw, targets, preds):
                image_resized = image_raw.copy()

                # Check if the original size is larger than the maximum image size
                if image_size < image_raw.size[0] or image_size < image_raw.size[1]:
                    # Downsize the image using the thumbnail method
                    image_resized.thumbnail((image_size, image_size),
                                            resample=Image.BILINEAR,
                                            reducing_gap=2)

                    # Calculate the downsize ratio
                    downsize_ratio = image_resized.size[0] / image_raw.size[0]
                else:
                    downsize_ratio = 1

                # Calculate the x and y offsets at which the downsized image needs to be pasted (to center it)
                x_offset = int((image_size - image_resized.size[0]) / 2)
                y_offset = int((image_size - image_resized.size[1]) / 2)

                target = resize_box_coord(target, downsize_ratio, x_offset, y_offset)
                pred = resize_box_coord(pred, downsize_ratio, x_offset, y_offset)

                # Declare an ImageDraw object
                draw = ImageDraw.Draw(image_raw)

                # Draw ground truth bounding boxes
                draw_boxes(draw, target, 'green', 3, font, (255, 255, 0, 0))

                # Draw predicted bounding boxes
                draw_boxes(draw, pred, 'red', 3, font, (255, 255, 255, 0))

                # Save the image
                image_raw.save(f'{save_path}'
                               f'{data_loader.dataset.image_paths[index].rsplit("/", 1)[-1].split(".", 1)[0]}'
                               f'.{IMAGE_EXT}')

            # Update the progress bar
            pbar.update()

    # Close the progress bar
    pbar.close()


def draw_boxes(draw: ImageDraw.ImageDraw, box_dict: dict, outline_color: str, outline_width: int,
               font: ImageFont.ImageFont, font_color: Tuple[int, int, int, int]) -> None:
    """
    Function to draw bounding boxes on an image

    :param draw: ImageDraw, ImageDraw PIL object
    :param box_dict: dict, dictionary containing bounding boxes
    :param outline_color: str, bounding box outline color in plain english
    :param outline_width: int, bounding box outline width in pixels
    :param font: ImageFont, ImageFont PIL object to specify which font to use to write the bounding boxes confidence
                 scores on the images
    :param font_color: tuple, four integers in the range (0, 255) to indicate font color in RGBA format
    """
    # Get list of boxes
    boxes = box_dict['boxes'].tolist()

    # Convert labels from integer indices values to class strings
    labels = [list(CLASS_DICT.keys())[list(CLASS_DICT.values()).index(label)]
              for label in box_dict['labels'].tolist()]

    # Check if confidence scores are present in the dictionary
    if 'scores' in box_dict:
        # Get list of confidence scores
        scores = box_dict['scores'].tolist()
    else:
        # Declare list of confidence scores as 100% confidence score for each box
        scores = [1] * len(labels)

    # Loop through each bounding box
    for box, label, score in zip(boxes, labels, scores):
        # Draw the bounding box
        draw.rectangle(box, outline=outline_color, width=outline_width)

        # Write the confidence score
        draw.text((box[0], box[1]), text=f'{label} {score:.4f}', font=font, fill=font_color)


def resize_box_coord(box_dict: dict, downsize_ratio: float, x_offset: float, y_offset: float) -> dict:
    # Loop through each bounding box
    for i in range(len(box_dict['boxes'])):
        # Loop through each of the 4 coordinates (x_min, y_min, x_max, y_max)
        for j in range(4):
            # Apply an offset to the bounding boxes
            if j == 0 or j == 2:
                box_dict['boxes'][i][j] -= x_offset
            else:
                box_dict['boxes'][i][j] -= y_offset

            # Apply a downsize ratio to the bounding boxes
            box_dict['boxes'][i][j] = int(box_dict['boxes'][i][j] / downsize_ratio)

    return box_dict
