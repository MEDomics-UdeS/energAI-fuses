import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
import cv2
from pycocotools.coco import COCO
from torchvision import transforms
import pandas as pd
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pytesseract
import numpy as np
from PIL import ImageDraw, Image, ImageFont
# from engine import evaluate
import re
from itertools import chain

from fuse_config import EPOCHS
from fuse_config import BATCH_SIZE
from fuse_config import NO_OF_CLASSES
from fuse_config import TRAIN_DATAPATH
from fuse_config import train_test_split_index
from fuse_config import class_dictionary
from fuse_config import LEARNING_RATE

from Fuse_Class import FuseDataset

from helper_functions import get_transform
from helper_functions import collate_fn
from helper_functions import get_model_instance_segmentation
from helper_functions import iou


if __name__ == "__main__":
    train_dataset = FuseDataset(
        root=TRAIN_DATAPATH, data_file=TRAIN_DATAPATH + "annotation.csv", transforms=get_transform())
    test_dataset = FuseDataset(
        root=TRAIN_DATAPATH, data_file=TRAIN_DATAPATH + "annotation.csv", transforms=get_transform())
    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(train_dataset)).tolist()

    train_dataset = torch.utils.data.Subset(
        train_dataset, indices[:-train_test_split_index])
    test_dataset = torch.utils.data.Subset(
        test_dataset, indices[-train_test_split_index:])

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1,
        collate_fn=collate_fn)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1,
        collate_fn=collate_fn)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("We have: {} examples, {} are training and {} testing".format(
        len(indices), len(train_dataset), len(test_dataset)))

    num_classes = NO_OF_CLASSES+1
    num_epochs = 1
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    len_dataloader = len(train_data_loader)

    for epoch in range(num_epochs):
        model.train()
        i = 0
        for imgs, annotations in train_data_loader:
            i += 1
            try:
                imgs = list(img.to(device) for img in imgs)
                annotations = [{k: v.to(device) for k, v in t.items()}
                               for t in annotations]

                loss_dict = model(imgs, annotations)
                if torch.isnan(sum(loss for loss in loss_dict.values())):
                    raise Exception("NAN")
            except Exception as e:
                print(e)
                print(annotations)
                continue
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            # print(annotations[0]['boxes'])
            print(
                f'[Epoch: {epoch}] Iteration: {i}/{len_dataloader}, Loss: {losses}')
            # evaluate(model, test_data_loader, device=device)

    torch.save(model.state_dict(), TRAIN_DATAPATH+"models/test5")

    loaded_model = get_model_instance_segmentation(num_classes=NO_OF_CLASSES+1)
    loaded_model.load_state_dict(torch.load(TRAIN_DATAPATH+"models/test5"))

    # print("Ground Truth \t\t Label \t\t\t BoxIndex \t\t IOU Score")
    print('{:^30} {:^30} {:^30} {:^30} {:^30} {:^30} {:^30}'.format(
        'Image', 'Ground Truth', 'Label', 'LabelOCR', 'Box Index', 'Confidence Score', 'IOU Score'))
    ocr_counter = 0
    label_counter = 0
    total = 0
    ocr_fail = 0
    for i in range(len(test_dataset)):
        img, _ = test_dataset[i]
        label_boxes = np.array(test_dataset[i][1]["boxes"])
        loaded_model.eval()

        with torch.no_grad():
            prediction = loaded_model([img])
            dict1 = []
            for elem in range(len(label_boxes)):
                label = test_dataset[i][1]["labels"][elem].cpu().numpy()
                label = list(class_dictionary.keys())[
                    list(class_dictionary.values()).index(label)]
                dict1.append({"label": label, "boxes": label_boxes[elem]})
            for element in range(len(prediction[0]["boxes"])):
                boxes = prediction[0]["boxes"][element].cpu().numpy()
                score = np.round(
                    prediction[0]["scores"][element].cpu().numpy(), decimals=4)
                label = prediction[0]["labels"][element].cpu().numpy()
                label = list(class_dictionary.keys())[
                    list(class_dictionary.values()).index(label)]
            try:
                label_via_ocr = "NO OCR"
                if score > 0.7:
                    # label_via_ocr = label_ocr(_['name'],boxes,label)
                    label_via_ocr = "NO OCR"
                    label_match = iou(label, boxes, dict1, score, _['name'])
                if label == label_via_ocr:
                    ocr_counter += 1
                if label_via_ocr == "ocr fail":
                    ocr_fail += 1
                    label_counter += label_match
                    total += 1
                else:
                    pass
            except Exception as e:
                print(e)

            print("-"*180)

    print("ACCURACY OF OCR:", ocr_counter/(total-ocr_fail))
    print("ACCURACY OF prediction:", label_counter/total)
