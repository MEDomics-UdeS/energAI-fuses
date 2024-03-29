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
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Tuple, Any

from src.coco.coco_utils import get_coco_api_from_dataset
from src.coco.coco_eval import CocoEvaluator
from src.utils.constants import *
from src.utils.helper_functions import cp_split, filter_by_nms, filter_by_score, format_detr_outputs
from src.models.models import load_model


@torch.no_grad()
def save_test_images(model_file_name: str,
                     data_loader: DataLoader,
                     with_gui: bool,
                     iou_threshold: float,
                     score_threshold: float,
                     save_path: str,
                     img_path: str,
                     image_size: int,
                     device_type: str) -> None:
    """Main inference testing function to save images with predicted and ground truth bounding boxes

    Args:
        model_file_name(str): model file name to load
        data_loader(DataLoader): data loader object
        with_gui(bool): choose whether to use the GUI for inference
        iou_threshold(float): intersection-over-union threshold for predicted bounding boxes filtering
        score_threshold(float): objectness score for predicted bounding boxes filtering
        save_path(str): save path for the inference test images
        img_path(str): image path
        image_size(int): image resizing size
        device_type(str): device type ('cpu' or 'cuda')

    """
    # Load the save state on the cpu for compatibility on non-CUDA systems
    save_state = torch.load(model_file_name, map_location=torch.device('cpu'))

    if with_gui:
        image_paths_raw = [image_path.replace(GUI_RESIZED_PATH, f'{img_path}/') for image_path in data_loader.dataset.image_paths]
        
        # Declare device
        if device_type == 'cuda' and torch.cuda.is_available():
            device = torch.device(device_type)
        else:
            if device_type != 'cpu':
                print('\n', '=' * 75)
                print(f"Couldn't assign device to type {device_type}, defaulting to CPU.")
            device = torch.device('cpu')
    else:
        image_paths_raw = [image_path.replace(RESIZED_PATH, RAW_PATH) for image_path in data_loader.dataset.image_paths]

        # Declare device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Declare progress bar
    pbar = tqdm(total=len(data_loader), leave=False, desc='Inference Test')

    model_name = save_state["args_dict"]["model"]
    model = load_model(model_name=model_name, pretrained=False, num_classes=len(CLASS_DICT), progress=True)
    model.load_state_dict(save_state["model"])

    model.to(device)

    # Put the model into eval() mode for inference
    model.eval()
    
    # Removes every files from the inference directory if it's not empty
    for file in os.listdir(INFERENCE_PATH):
        if file.startswith('.') is False:
            os.remove(f'{INFERENCE_PATH}{file}')

    # Loop through each batch in the data loader
    for batch_no, (images, targets) in enumerate(data_loader):
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

            if target:
                target = resize_box_coord(target, downsize_ratio, x_offset, y_offset)
            pred = resize_box_coord(pred, downsize_ratio, x_offset, y_offset)

            # Declare an ImageDraw object
            draw = ImageDraw.Draw(image_raw)

            # Defines bbox outlines width and font for image drawing
            pred_annotations, target_annotations = scale_annotation_sizes(image_raw, pred, target)

            # Drawing the annotations on the image
            draw_annotations(draw, pred, target, pred_annotations, target_annotations)

            # Save the image
            image_raw.save(f'{save_path}'
                           f'{cp_split(data_loader.dataset.image_paths[index])[-1].split(".")[0]}'
                           f'.{IMAGE_EXT}')

        # Update the progress bar
        pbar.update()

    # Close the progress bar
    pbar.close()
    
    
def draw_annotations(draw: ImageDraw.ImageDraw,
                     pred_box_dict: dict,
                     target_box_dict: dict,
                     pred_annotations: list,
                     target_annotations: list) -> None:
    """Function to draw bounding boxes on an image

    Args:
        draw(ImageDraw.ImageDraw): ImageDraw PIL object
        pred_box_dict(dict): dictionary containing predicted bounding boxes
        target_box_dict(dict): dictionary containing ground truth bounding boxes
        pred_annotations(list): contains the width and font of every individual predicted bounding boxes
        target_annotations(list):  contains the width and font of every individual ground truth bounding boxes

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
        draw.rectangle(pred_box, outline=COLOR_PALETTE["yellow"], width=box_width)

    if target_box_dict:
        target_boxes = target_box_dict['boxes'].tolist()
        target_labels = [list(CLASS_DICT.keys())[list(CLASS_DICT.values()).index(label)]
                         for label in target_box_dict['labels'].tolist()]
        # Declare list of confidence scores as 100% confidence score for each box
        target_scores = [1] * len(target_labels)
        
        # Drawing predicted bounding boxes on the image
        for target_box, (box_width, font) in zip(target_boxes, target_annotations):
            draw.rectangle(
                target_box, outline=COLOR_PALETTE["green"], width=box_width)

        # Drawing predicted labels over the bounding boxes on the image
        for target_box, target_label, target_score, (box_width, font)\
                in zip(target_boxes, target_labels, target_scores, target_annotations):
            draw.text(
                (target_box[0], target_box[1] + font.size), text=f'{target_label}', font=font, fill=COLOR_PALETTE["green"], stroke_width=int(font.size / 10), stroke_fill=COLOR_PALETTE["bg"])

    # Drawing predicted labels over the bounding boxes on the image
    for pred_box, pred_label, pred_score, (box_width, font)\
            in zip(pred_boxes, pred_labels, pred_scores, pred_annotations):
        draw.text(
            (pred_box[0], pred_box[1]), text=f'{pred_label} {pred_score:.4f}', font=font, fill=COLOR_PALETTE["yellow"], stroke_width=int(font.size / 10), stroke_fill=COLOR_PALETTE["bg"])


def resize_box_coord(box_dict: dict,
                     downsize_ratio: float,
                     x_offset: float,
                     y_offset: float) -> dict:
    """Function to resize bounding box coordinates

    Args:
        box_dict(dict): bounding box coordinates dictionary
        downsize_ratio(float): downsize ratio floating point value [0, 1]
        x_offset(float): x offset for bounding boxes
        y_offset(float): y offset for bounding boxes

    Returns: resized bounding box coordinates dictionary

    """
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


def scale_annotation_sizes(img: Image,
                           pred: dict,
                           target: dict) -> Tuple[list, list]:
    """Function to scale the annotations drawn on an image during inference
    
    Bounding boxes are scaled with a power function in relation to the area of the box over the area of the picture.
    Font sizes are scaled with a power function in relation to the area of the box alone

    Args:
        img(Image): PIL image
        pred(dict): predictions dictionary
        target(dict): targets dictionary

    Returns: two lists of scaled annotations

    """

    pred_annotations = []
    target_annotations = []

    for box in pred["boxes"]:
        box_area = (box[0] - box[2]) * (box[1] - box[3])

        box_width = scale_box_width(box_area)

        font_size = int((1.2 * box_area) ** 0.3)

        pred_annotations.append(
            (box_width, ImageFont.truetype(FONT_PATH, font_size)))
        
    try:
        for box in target["boxes"]:
            box_area = (box[0] - box[2]) * (box[1] - box[3])

            box_width = scale_box_width(box_area)

            font_size = int((1.2 * box_area) ** 0.3)

            target_annotations.append(
                (box_width, ImageFont.truetype(FONT_PATH, font_size)))
    except KeyError:
        pass

    # Declare the font object to write the confidence scores on the images
    return pred_annotations, target_annotations


def scale_box_width(area):
    # Scaling formula from : https://math.stackexchange.com/questions/43698/range-scaling-problem
    # Scales linearly values from range [A, B] in [C, D]

    A, B, C, D = 1000, 2_000_000, 5, 32

    # Forcing extreme values in the expected range
    if area < A:
        area = A
    if area > B:
        area = B

    term1 = C * (1 - (area - A)/(B - A))
    term2 = D * ((area - A)/(B - A))

    return int(term1 + term2)


@torch.no_grad()
def coco_evaluate(model: Any,
                  data_loader: DataLoader,
                  desc: str,
                  device: torch.device,
                  image_size: int,
                  model_name: str) -> dict:
    """Function to evaluate COCO AP results for inference

    Args:
        model(Any): model object
        data_loader(DataLoader): data loader object
        desc(str): description string for tqdm progress bar
        device(torch.device): PyTorch device 'cpu' or 'cuda'
        image_size(int): image resizing size
        model_name(str): model name

    Returns: dictionary of COCO AP results

    """
    pbar = tqdm(total=len(data_loader), leave=False, desc=desc)

    coco = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco, ['bbox'])

    model.eval()

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)

        outputs = model(images)

        if model_name == 'detr':
            target_sizes = torch.stack(
                [torch.tensor([image_size, image_size]) for _ in targets], dim=0)
            outputs = format_detr_outputs(outputs, target_sizes, device)

        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

        results = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(results)

        pbar.update()

    pbar.close()

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return dict(zip(COCO_PARAMS_LIST, coco_evaluator.coco_eval['bbox'].stats.tolist()))
