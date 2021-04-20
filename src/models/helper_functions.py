import operator
import os
import re
import time
import traceback
from itertools import chain
from tqdm import trange

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
import seaborn as sns
import torch
import torch.cuda.amp as amp
import torchvision
from PIL import Image, ImageDraw, ImageFont
from fuzzywuzzy import fuzz

from torchvision import transforms
from src.models.early_stopping import EarlyStopping
from constants import *

import ray


def train_model(epochs, accumulation_size, train_data_loader, device, mixed_precision,
                gradient_accumulation, filename, writer, early, validation, validation_dataset):
    model = get_model_instance_segmentation(len(CLASS_DICT) + 1)

    # move model to the right device
    model.to(device)
    if early:
        es = EarlyStopping(patience=early)
    # parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    len_dataloader = len(train_data_loader)
    to = []
    r = []
    a = []
    f = []
    if mixed_precision:
        scaler = amp.grad_scaler.GradScaler(enabled=mixed_precision)

    step = -1

    for epoch in range(epochs):
        model.train()
        i = 0
        losses = 0
        if gradient_accumulation:
            optimizer.zero_grad()

        for imgs, annotations in train_data_loader:
            i += 1
            step += 1
            # try:
            imgs = torch.stack(imgs).to(device)

            annotations = [{k: v.to(device) for k, v in t.items()}
                           for t in annotations]

            if gradient_accumulation and mixed_precision:
                with amp.autocast(enabled=mixed_precision):
                    loss_dict = model(imgs, annotations)
                    losses = sum(loss for loss in loss_dict.values())

                to.append(torch.cuda.get_device_properties(0).total_memory * 1e-9)
                r.append(torch.cuda.memory_reserved(0) * 1e-9)
                a.append(torch.cuda.memory_allocated(0) * 1e-9)
                f.append(torch.cuda.memory_reserved(0) * 1e-9 -
                         torch.cuda.memory_allocated(0) * 1e-9)

                scaler.scale(losses).backward()

                if (i + 1) % accumulation_size == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            elif gradient_accumulation and not mixed_precision:
                loss_dict = model(imgs, annotations)
                losses = sum(loss for loss in loss_dict.values())

                to.append(torch.cuda.get_device_properties(0).total_memory * 1e-9)
                r.append(torch.cuda.memory_reserved(0) * 1e-9)
                a.append(torch.cuda.memory_allocated(0) * 1e-9)
                f.append(torch.cuda.memory_reserved(0) * 1e-9 -
                         torch.cuda.memory_allocated(0) * 1e-9)

                losses.backward()

                if (i + 1) % accumulation_size == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                    optimizer.step()
                    optimizer.zero_grad()
            elif not gradient_accumulation and mixed_precision:
                with amp.autocast(enabled=mixed_precision):
                    loss_dict = model(imgs, annotations)
                    losses = sum(loss for loss in loss_dict.values())

                to.append(torch.cuda.get_device_properties(0).total_memory * 1e-9)
                r.append(torch.cuda.memory_reserved(0) * 1e-9)
                a.append(torch.cuda.memory_allocated(0) * 1e-9)
                f.append(torch.cuda.memory_reserved(0) * 1e-9 -
                         torch.cuda.memory_allocated(0) * 1e-9)

                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            elif not gradient_accumulation and not mixed_precision:
                loss_dict = model(imgs, annotations)

                to.append(torch.cuda.get_device_properties(0).total_memory * 1e-9)
                r.append(torch.cuda.memory_reserved(0) * 1e-9)
                a.append(torch.cuda.memory_allocated(0) * 1e-9)
                f.append(torch.cuda.memory_reserved(0) * 1e-9 -
                         torch.cuda.memory_allocated(0) * 1e-9)

                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            writer.add_scalar("Loss/train", losses, step)
            writer.add_scalar("Memory/reserved", r[-1], step)
            writer.add_scalar("Memory/allocated", a[-1], step)
            writer.add_scalar("Memory/free", f[-1], step)

            if i % 10 == 0:
                print(
                    f'[Epoch: {epoch}] Iteration: {i}/{len_dataloader}, Loss: {losses}')
            del imgs, annotations, loss_dict
            torch.cuda.empty_cache()

            # except Exception as e:
            #     print(e)
            #     print(annotations)
            #     to.append(torch.cuda.get_device_properties(0).total_memory * 1e-9)
            #     r.append(torch.cuda.memory_reserved(0) * 1e-9)
            #     a.append(torch.cuda.memory_allocated(0) * 1e-9)
            #     f.append(torch.cuda.memory_reserved(0) * 1e-9 -
            #              torch.cuda.memory_allocated(0) * 1e-9)
            #
            #     del imgs, annotations
            #     torch.cuda.empty_cache()
            #     to.append(torch.cuda.get_device_properties(0).total_memory * 1e-9)
            #     r.append(torch.cuda.memory_reserved(0) * 1e-9)
            #     a.append(torch.cuda.memory_allocated(0) * 1e-9)
            #     f.append(torch.cuda.memory_reserved(0) * 1e-9 -
            #              torch.cuda.memory_allocated(0) * 1e-9)
            #
            #     traceback.print_exc()
            #     continue

        if validation:
            if (epoch + 1) % validation == 0:
                torch.save(model.state_dict(), "models/" + filename)
                val_acc = validate_model(validation_dataset, filename)
                if early:
                    if es.step(torch.as_tensor(val_acc, dtype=torch.float16)):
                        print("Early Stopping")
                        break
        elif early:
            if es.step(losses):
                print("Early Stopping")
                break

    torch.save(model.state_dict(), "models/" + filename)


def validate_model(val_dataset, filename):
    start = time.time()
    loaded_model = get_model_instance_segmentation(num_classes=len(CLASS_DICT) + 1)
    loaded_model.load_state_dict(torch.load("models/" + filename))
    label_counter = 0
    total = 0
    for i in range(len(val_dataset)):
        if i % 20 == 0:
            print(i, end=" ")

        img, targets = val_dataset[i]
        label_boxes = targets["boxes"].detach().cpu().numpy()
        loaded_model.eval()
        with torch.no_grad():
            prediction = loaded_model([img])
            dict1 = []
            for elem in range(len(label_boxes)):
                label = targets["labels"][elem].detach().cpu().numpy()
                label = list(CLASS_DICT.keys())[
                    list(CLASS_DICT.values()).index(label)]
                dict1.append({"label": label, "boxes": label_boxes[elem]})
                total += 1
            for element in range(len(prediction[0]["boxes"])):
                boxes = prediction[0]["boxes"][element].cpu().numpy()
                score = np.round(
                    prediction[0]["scores"][element].cpu().numpy(), decimals=4)
                label = prediction[0]["labels"][element].cpu().numpy()
                label = list(CLASS_DICT.keys())[
                    list(CLASS_DICT.values()).index(label)]
                try:
                    label_match = 0
                    if score > 0.7:
                        label_match = iou(label, boxes, dict1, score, targets[
                            'name'], i)
                    else:
                        pass
                    label_counter += label_match
                except Exception as e:
                    print(e)
                    traceback.print_exc()

    print("\nValidation Accuracy:", label_counter / total)
    print("Time for Validation: ", round((time.time() - start) / 60, 2))
    return label_counter / total


def test_model(test_dataset, device, filename, writer):
    loaded_model = get_model_instance_segmentation(num_classes=len(CLASS_DICT) + 1)
    loaded_model.load_state_dict(torch.load("models/" + filename))

    # print("Ground Truth \t\t Label \t\t\t BoxIndex \t\t IOU Score")
    print('{:^3} {:^20} {:^20} {:^25} {:^20} {:^10} {:^10}'.format('Index',
                                                                   'Image', 'Ground Truth', 'Label', 'Box Index',
                                                                   'Confidence Score', 'IOU Score'))
    ocr_counter = 0
    label_counter = 0
    total = 0
    ocr_fail = 0
    for i in range(len(test_dataset)):
        img, targets = test_dataset[i]
        label_boxes = targets["boxes"].detach().cpu().numpy()
        loaded_model.eval()

        with torch.no_grad():
            prediction = loaded_model([img])
            dict1 = []
            for elem in range(len(label_boxes)):
                label = targets["labels"][elem].cpu().numpy()
                label = list(CLASS_DICT.keys())[
                    list(CLASS_DICT.values()).index(label)]
                dict1.append({"label": label, "boxes": label_boxes[elem]})
                total += 1
            for element in range(len(prediction[0]["boxes"])):
                boxes = prediction[0]["boxes"][element].cpu().numpy()
                score = np.round(
                    prediction[0]["scores"][element].cpu().numpy(), decimals=4)
                label = prediction[0]["labels"][element].cpu().numpy()
                label = list(CLASS_DICT.keys())[
                    list(CLASS_DICT.values()).index(label)]
                try:
                    label_match = 0
                    if score > 0.7:
                        # label_via_ocr = label_ocr(_['name'], boxes, label)
                        label_match = iou(label, boxes, dict1, score, targets[
                            'name'], i, "No OCR", True)

                        # if label == label_via_ocr:
                        #     ocr_counter += 1
                        # if label_via_ocr == "ocr fail":
                        #     ocr_fail += 1
                    else:
                        pass
                    # total += 1
                    label_counter += label_match
                    # print(label_counter,total,label_counter/total)
                except Exception as e:
                    print(e)

            print("-" * 150)

    try:
        print("ACCURACY OF OCR:", ocr_counter / (total - ocr_fail))
    except Exception:
        print("OCR FAILED")
    print(label_counter, total)
    print("ACCURACY OF prediction:", label_counter / total)
    writer.add_scalar("Accuracy/test", label_counter / total, 1)


def iou(label, box1, box2, score, name, index, label_ocr="No OCR", verbose=False):
    name = ''.join(chr(i) for i in name)
    # print('{:^30}'.format(name[-1]))
    iou_list = []
    iou_label = []
    label_index = -1
    for item in box2:
        try:
            x11, y11, x21, y21 = item["boxes"]
            x12, y12, x22, y22 = box1
            xi1 = max(x11, x12)
            yi1 = max(y11, y12)
            xi2 = min(x21, x22)
            yi2 = min(y21, y22)

            inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
            # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
            box1_area = (x21 - x11 + 1) * (y21 - y11 + 1)
            box2_area = (x22 - x12 + 1) * (y22 - y12 + 1)
            union_area = float(box1_area + box2_area - inter_area)
            # compute the IoU
            iou = inter_area / union_area
            iou_list.append(iou)
            label_index = iou_list.index(max(iou_list))
        except Exception as e:
            print("Error: ", e)
            print(iou_list)
            continue
    score1 = '%.2f' % score
    if verbose:
        try:
            print('{:^3} {:^20} {:^20} {:^25} {:^20} {:^10} {:^10}'.format(index,
                                                                           name, box2[label_index]["label"], label,
                                                                           label_index, score1,
                                                                           round(max(iou_list), 2)))
        except Exception:
            print('{:^3} {:^20} {:^20} {:^25} {:^20} {:^10} {:^10}'.format(index,
                                                                           name, "None", label, label_index, score1,
                                                                           "None"))
    if box2[label_index]["label"] == label:
        return float(score1)
    else:
        return 0





def label_ocr(img, box, label):
    name = ''.join(chr(i) for i in img)
    path = os.path.join("data/raw", name)
    image = cv2.imread(path)
    x1, y1, x2, y2 = box

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    # print(path)

    result_list = []
    try:
        crop_img_1 = image[x1:x2, y1:y2]
        gray_1 = cv2.cvtColor(crop_img_1, cv2.COLOR_BGR2GRAY)

        # blur_1 = cv2.GaussianBlur(gray_1, (3,3), 0)
        # thresh_1 = cv2.threshold(blur_1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # opening_1 = cv2.morphologyEx(thresh_1, cv2.MORPH_OPEN, kernel, iterations=1)
        # invert_1 = 255 - opening_1

        data_1 = pytesseract.image_to_string(
            gray_1, lang='eng', config='-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz --psm 11')
        dataList_1 = re.split(r'[,.\n ]', data_1)  # split the string
        result_list.append([(i.strip()) for i in dataList_1 if i != ''])
    except Exception:
        pass

    try:
        crop_img_2 = image[y1:y2, x1:x2]
        gray_2 = cv2.cvtColor(crop_img_1, cv2.COLOR_BGR2GRAY)

        # blur_2 = cv2.GaussianBlur(gray_2, (3,3), 0)
        # thresh_2 = cv2.threshold(blur_2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # opening_2 = cv2.morphologyEx(thresh_2, cv2.MORPH_OPEN, kernel, iterations=1)
        # invert_2 = 255 - opening_2

        data_2 = pytesseract.image_to_string(
            gray_2, lang='eng', config='-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz --psm 11')
        data_list_2 = re.split(r',|\.|\n| ', data_2)
        result_list.append([(i.strip()) for i in data_list_2 if i != ''])
    except Exception:
        pass

    ocr_dict_rank = {
        "Gould Shawmut A4J": 0,
        "Ferraz Shawmut AJT": 0,
        "English Electric C-J": 0,
        "Ferraz Shawmut CRS": 0,
        "GEC HRC I-J": 0,
        "Gould Shawmut TRSR": 0,
        "English Electric Form II": 0,
        "Bussmann LPJ": 0,
        "Gould Shawmut CJ": 0,
    }

    result_list = list(chain.from_iterable(result_list))
    # print("OCR Recognized List",result_list)
    count = 0
    out_count = 0
    # key = "Ferraz Shawmut CRS"
    # for i in range(100):
    #   ocr_dict_rank[key] += 1
    try:
        for list_item in result_list:
            for key, value in OCR_DICT.items():
                for v in value:
                    if fuzz.partial_ratio(v.lower(), list_item.lower()) > 90 and len(list_item) > 3:
                        ocr_dict_rank[key] = ocr_dict_rank.get(key, 0) + 1
                    elif fuzz.partial_ratio(v.lower(), list_item.lower()) > 75 and len(list_item) > 3:
                        ocr_dict_rank[key] = ocr_dict_rank.get(key, 0) + 0.5
                        # print("v: [{0}] item: [{1}] fuzz: [{2}] fuzz_reverse: [{3}]".format(v.lower(),
                        # list_item.lower(),fuzz.partial_ratio(v.lower(),list_item.lower()),fuzz.partial_ratio(
                        # list_item.lower(),v.lower()))) print(key)

    except Exception as e:
        print(e)

    sorted_d = dict(sorted(ocr_dict_rank.items(),
                           key=operator.itemgetter(1), reverse=True))
    if sorted_d[list(sorted_d.keys())[0]] > 1:
        print(sorted_d)
        return list(sorted_d.keys())[0]
    else:
        return "ocr fail"


def view_test_image(idx, test_dataset, filename):
    loaded_model = get_model_instance_segmentation(num_classes=len(CLASS_DICT) + 1)
    loaded_model.load_state_dict(torch.load("models/" + filename))
    img, targets = test_dataset[idx]
    img_name = ''.join(chr(i) for i in targets['name'])
    print(img_name)

    label_boxes = targets['boxes'].detach().cpu().numpy()
    # put the model in evaluation mode
    loaded_model.eval()
    with torch.no_grad():
        prediction = loaded_model([img])
    image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    draw = ImageDraw.Draw(image)
    # draw groundtruth
    for elem in range(len(label_boxes)):
        label = test_dataset[idx][1]["labels"][elem].cpu().numpy()
        label = list(CLASS_DICT.keys())[
            list(CLASS_DICT.values()).index(label)]
        draw.rectangle([(label_boxes[elem][0], label_boxes[elem][1]),
                        (label_boxes[elem][2], label_boxes[elem][3])], outline="green", width=3)
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 50)
        draw.text((label_boxes[elem][0], label_boxes[elem][1]),
                  text=label + " " + str(1), font=font, fill=(255, 255, 0, 0))
    for element in range(len(prediction[0]["boxes"])):
        boxes = prediction[0]["boxes"][element].cpu().numpy()
        score = np.round(prediction[0]["scores"]
                         [element].cpu().numpy(), decimals=4)
        label = prediction[0]["labels"][element].cpu().numpy()
        label = list(CLASS_DICT.keys())[
            list(CLASS_DICT.values()).index(label)]
        if score > 0.5:
            draw.rectangle(
                [(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline="red", width=5)
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 50)
            draw.text((boxes[0] - 1, boxes[3]), text=f"{label} {score}", font=font, fill=(255, 255, 255, 0))

    image = image.save("image_save_1/" + img_name)


