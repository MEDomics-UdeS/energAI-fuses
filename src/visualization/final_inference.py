"""
File:
    src/visualization/inference.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Inference functions to test a saved model and show model predictions and ground truths boxes on the images
"""

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.models.models import load_model

from src.utils.constants import CLASS_DICT, FONT_PATH, GUI_RESIZED_PATH, INFERENCE_PATH, IMAGE_EXT
from src.utils.helper_functions import filter_by_nms, filter_by_score, format_detr_outputs
import os


@torch.no_grad()
def save_test_images(model_file_name: str,
                     data_loader: DataLoader,
                     iou_threshold: float,
                     score_threshold: float,
                     save_path: str,
                     img_path: str,
                     image_size: int,
                     device_type: str) -> None:
    """
    Main inference testing function to save images with predicted and ground truth bounding boxes

    :param score_threshold:
    :param model_file_name: str, model file name to load
    :param data_loader: DataLoader, data loader object
    :param iou_threshold: float, intersection-over-union threshold for predicted bounding boxes filtering
    :param save_path: str, save path for the inference test images
    """
    save_state = torch.load(model_file_name)

    image_paths_raw = [image_path.replace(
        GUI_RESIZED_PATH, f'{img_path}/') for image_path in data_loader.dataset.image_paths]

    # Declare device
    if device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device(device_type)
    else:
        if device_type != 'cpu':
            print('\n', '=' * 75)
            print(f"Couldn't assign device to type {device_type}, defaulting to CPU.")
        device = torch.device('cpu')
    
    # Declare progress bar
    pbar = tqdm(total=len(data_loader), leave=False, desc='Inference Test')

    model_name = save_state["args_dict"]["model"]
    model = load_model(model_name=model_name, pretrained=False,
                       num_classes=len(CLASS_DICT), progress=True)
    model.load_state_dict(save_state["model"])

    model.to(device)

    # Put the model into eval() mode for inference
    model.eval()

    # Removes every files from the inference directory if it's not empty
    for file in os.listdir(INFERENCE_PATH):
        if file.startswith('.') is False:
            os.remove(f'{INFERENCE_PATH}{file}')

    # Loop through each batch in the data loader
    for batch_no, (images, _) in enumerate(data_loader):
        # Get current batch indices
        indices = range(data_loader.batch_size * batch_no,
                        min(data_loader.batch_size * (batch_no + 1), len(data_loader.dataset)))

        # Send images to the device
        images = torch.stack(images).to(device)

        # Perform a forward pass and obtain bounding boxes predictions
        preds = model(images)

        if model_name == 'detr':
            target_sizes = torch.stack(
                [torch.tensor([image_size, image_size]) for _ in range(data_loader.batch_size)], dim=0)
            preds = format_detr_outputs(preds, target_sizes, device)

        # Filter predicted bounding boxes by using non-maximum suppression
        preds = filter_by_nms(preds, iou_threshold)

        # Filter predicted bounding boxes by using a confidence score threshold
        preds = filter_by_score(preds, score_threshold)

        # Load images in the current batch
        images_raw = [Image.open(image_paths_raw[index]) for index in indices]

        # Loop through each image in the current batch
        for index, image, image_raw, pred in zip(indices, images, images_raw, preds):
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

            pred = resize_box_coord(pred, downsize_ratio, x_offset, y_offset)

            # Declare an ImageDraw object
            draw = ImageDraw.Draw(image_raw)

            # Defines bbox outlines width and font for image drawing
            pred_annotations = scale_annotation_sizes(image_raw, pred["boxes"])

            # Drawing the annotations on the image
            draw_annotations(draw, pred, pred_annotations)

            # Save the image
            image_raw.save(f'{save_path}'
                           f'{data_loader.dataset.image_paths[index].rsplit("/", 1)[-1].split(".", 1)[0]}'
                           f'.{IMAGE_EXT}')

        # Update the progress bar
        pbar.update()

    # Close the progress bar
    pbar.close()


def draw_annotations(draw: ImageDraw.ImageDraw, pred_box_dict: dict, pred_annotations: list):
    """
    Function to draw bounding boxes on an image

    :param draw: ImageDraw, ImageDraw PIL object
    :param pred_box_dict: dict, dictionary containing predicted bounding boxes
    :param target_box_dict: dict, dictionary containing ground truth bounding boxes
    :param pred_annotations: list, contains the width and font of every individual predicted bounding boxes
    :param target_annotations: list, contains the width and font of every individual ground truth bounding boxes
    """
    # Get list of boxes
    pred_boxes = pred_box_dict['boxes'].tolist()

    # Convert labels from integer indices values to class strings
    pred_labels = [list(CLASS_DICT.keys())[list(CLASS_DICT.values()).index(label)]
                   for label in pred_box_dict['labels'].tolist()]

    # Get list of confidence scores
    pred_scores = pred_box_dict['scores'].tolist()

    # Drawing predicted bounding boxes on the image
    for pred_box, (box_width, font) in zip(pred_boxes, pred_annotations):
        draw.rectangle(pred_box, outline="red", width=box_width)
        
    # Drawing predicted labels over the bounding boxes on the image
    for pred_box, pred_label, pred_score, (box_width, font)\
            in zip(pred_boxes, pred_labels, pred_scores, pred_annotations):
        draw.text(
            (pred_box[0], pred_box[1] + font.size), text=f'{pred_label} {pred_score:.4f}', font=font, fill=(255, 255, 255, 0))


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
            box_dict['boxes'][i][j] = int(
                box_dict['boxes'][i][j] / downsize_ratio)

    return box_dict


def scale_annotation_sizes(img: Image, pred_boxes: list, box_scaler: float = 0.006, text_scaler: float = 0.015) -> list:
    """
    Function to scale the annotations drawn on an image during inference
    """
    MAX_BBOX_SIZE = 32
    MAX_FONT_SIZE = 64
    img_area = img.size[0] * img.size[1]

    pred_annotations = []

    for box in pred_boxes:
        box_area = (box[0] - box[2]) * (box[1] - box[3])

        box_width = min(
            int(img_area / box_area * box_scaler) + 8, MAX_BBOX_SIZE)
        font_size = min(int(max(img.size) * text_scaler) + 6, MAX_FONT_SIZE)

        pred_annotations.append(
            (box_width, ImageFont.truetype(FONT_PATH, font_size)))


    # Declare the font object to write the confidence scores on the images
    return pred_annotations
