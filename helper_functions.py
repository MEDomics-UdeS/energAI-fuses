import operator
import os
import re
import traceback
from itertools import chain
import pandas as pd


import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import seaborn as sns
import torch
import torch.cuda.amp as amp
import torchvision
from fuzzywuzzy import fuzz
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm.auto import tqdm
from PIL import Image, ImageDraw, ImageFont

from fuse_config import (LEARNING_RATE, NO_OF_CLASSES,
                         TRAIN_DATAPATH, SAVE_PATH, class_dictionary)

"""
converts the image to a tensor
"""


def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def save_graph(t, r, a, f, filename):
    # print(t,r,a,f)
    x = np.arange(0, len(t))
    df = pd.DataFrame({"time": x, "t": t, "r": r, "a": a, "f": f})
    sns.set_style("darkgrid")
    sns.lineplot(x="time", y="t", data=df, label="total")
    sns.lineplot(x="time", y="r", data=df, label="reserved")
    sns.lineplot(x="time", y="a", data=df, label="allocated")
    sns.lineplot(x="time", y="f", data=df, label="free")
    title = filename.split("_")
    epoch_string = title[2][1:]
    batch_string = title[3][1:]
    downsample_string = title[4][1:]
    mp_string = title[5][2:]
    gradient_string = title[6][1:]
    plt.title("Epochs: "+epoch_string+" Batch Size: "+batch_string+" \nDownsample: " +
              downsample_string+" Mixed Prec: "+mp_string+" Gradient Acc: "+gradient_string)
    plt.savefig(SAVE_PATH+"/plots/"+filename+'.png')


def train_model(epochs, batch_size, train_data_loader, device, mixed_precision, gradient_accumulation, filename, verbose,writer):
    model = get_model_instance_segmentation(NO_OF_CLASSES+1)

    # move model to the right device
    model.to(device)

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
    for epoch in range(epochs+1):
        resize_ratios_all = []
        model.train()
        i = 0
        losses = 0
        if gradient_accumulation:
            optimizer.zero_grad()
        # pbar = tqdm(train_data_loader, desc=f'Epoch {epoch}',position=0, leave=True)

        for imgs, annotations, resize_ratios in train_data_loader:
            i += 1
            try:
                imgs = list(img.to(device) for img in imgs)

                annotations = [{k: v.to(device) for k, v in t.items()}
                               for t in annotations]

                resize_ratios_all.extend(list(resize_ratios))
                
                # to.append(torch.cuda.get_device_properties(0).total_memory*1e-9)
                # r.append(torch.cuda.memory_reserved(0)*1e-9)
                # a.append(torch.cuda.memory_allocated(0)*1e-9)
                # f.append(torch.cuda.memory_reserved(0)*1e-9 -
                #          torch.cuda.memory_allocated(0)*1e-9)

                if gradient_accumulation and mixed_precision:
                    with amp.autocast(enabled=mixed_precision):
                        loss_dict = model(imgs, annotations)
                        losses = sum(loss for loss in loss_dict.values())

                    to.append(torch.cuda.get_device_properties(0).total_memory*1e-9)
                    r.append(torch.cuda.memory_reserved(0)*1e-9)
                    a.append(torch.cuda.memory_allocated(0)*1e-9)
                    f.append(torch.cuda.memory_reserved(0)*1e-9 -
                         torch.cuda.memory_allocated(0)*1e-9)

                    writer.add_scalar("Loss/train", losses, epoch)
                    writer.add_scalar("Memory/reserved", r[-1], epoch)
                    writer.add_scalar("Memory/allocated", a[-1], epoch)
                    writer.add_scalar("Memory/free", f[-1], epoch)
                    
                         
                    scaler.scale(losses).backward()
                    if (i+1) % batch_size == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                elif gradient_accumulation and not mixed_precision:
                    loss_dict = model(imgs, annotations)
                    losses = sum(loss for loss in loss_dict.values())

                    to.append(torch.cuda.get_device_properties(0).total_memory*1e-9)
                    r.append(torch.cuda.memory_reserved(0)*1e-9)
                    a.append(torch.cuda.memory_allocated(0)*1e-9)
                    f.append(torch.cuda.memory_reserved(0)*1e-9 -
                         torch.cuda.memory_allocated(0)*1e-9)

                    writer.add_scalar("Loss/train", losses, epoch)
                    writer.add_scalar("Memory/reserved", r[-1], epoch)
                    writer.add_scalar("Memory/allocated", a[-1], epoch)
                    writer.add_scalar("Memory/free", f[-1], epoch)


                    losses.backward()
                    if (i+1) % batch_size == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                elif not gradient_accumulation and mixed_precision:
                    with amp.autocast(enabled=mixed_precision):
                        loss_dict = model(imgs, annotations)
                        losses = sum(loss for loss in loss_dict.values())

                    to.append(torch.cuda.get_device_properties(0).total_memory*1e-9)
                    r.append(torch.cuda.memory_reserved(0)*1e-9)
                    a.append(torch.cuda.memory_allocated(0)*1e-9)
                    f.append(torch.cuda.memory_reserved(0)*1e-9 -
                         torch.cuda.memory_allocated(0)*1e-9)
                    
                    writer.add_scalar("Loss/train", losses, epoch)
                    writer.add_scalar("Memory/reserved", r[-1], epoch)
                    writer.add_scalar("Memory/allocated", a[-1], epoch)
                    writer.add_scalar("Memory/free", f[-1], epoch)

                    scaler.scale(losses).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                elif not gradient_accumulation and not mixed_precision:
                    # to.append(torch.cuda.get_device_properties(0).total_memory*1e-9)
                    # r.append(torch.cuda.memory_reserved(0)*1e-9)
                    # a.append(torch.cuda.memory_allocated(0)*1e-9)
                    # f.append(torch.cuda.memory_reserved(0)*1e-9 -
                    #      torch.cuda.memory_allocated(0)*1e-9)

                    loss_dict = model(imgs, annotations)

                    to.append(torch.cuda.get_device_properties(0).total_memory*1e-9)
                    r.append(torch.cuda.memory_reserved(0)*1e-9)
                    a.append(torch.cuda.memory_allocated(0)*1e-9)
                    f.append(torch.cuda.memory_reserved(0)*1e-9 -
                         torch.cuda.memory_allocated(0)*1e-9)

                    losses = sum(loss for loss in loss_dict.values())

                    writer.add_scalar("Loss/train", losses, epoch)
                    writer.add_scalar("Memory/reserved", r[-1], epoch)
                    writer.add_scalar("Memory/allocated", a[-1], epoch)
                    writer.add_scalar("Memory/free", f[-1], epoch)

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                if i%10 == 0:
                    print(
                        f'[Epoch: {epoch}] Iteration: {i}/{len_dataloader}, Loss: {losses}')
                del imgs, annotations, loss_dict
                torch.cuda.empty_cache()

                # print(torch.cuda.memory_summary(device))

                # to.append(torch.cuda.get_device_properties(0).total_memory*1e-9)
                # r.append(torch.cuda.memory_reserved(0)*1e-9)
                # a.append(torch.cuda.memory_allocated(0)*1e-9)
                # f.append(torch.cuda.memory_reserved(0)*1e-9 -
                #          torch.cuda.memory_allocated(0)*1e-9)  # free inside reserved
                # # pbar.set_postfix({'loss': losses.item()})

            except Exception as e:
                print(e)
                print(annotations)
                to.append(torch.cuda.get_device_properties(0).total_memory*1e-9)
                r.append(torch.cuda.memory_reserved(0)*1e-9)
                a.append(torch.cuda.memory_allocated(0)*1e-9)
                f.append(torch.cuda.memory_reserved(0)*1e-9 -
                         torch.cuda.memory_allocated(0)*1e-9)

                del imgs, annotations
                torch.cuda.empty_cache()
                to.append(torch.cuda.get_device_properties(0).total_memory*1e-9)
                r.append(torch.cuda.memory_reserved(0)*1e-9)
                a.append(torch.cuda.memory_allocated(0)*1e-9)
                f.append(torch.cuda.memory_reserved(0)*1e-9 -
                         torch.cuda.memory_allocated(0)*1e-9)

                traceback.print_exc()
                # print(annotations)
                continue
        average_ratio = sum(resize_ratios_all) / len(resize_ratios_all)
        print(f'[Epoch: {epoch}] Average downsampling ratio : {average_ratio:.2%}')
        print(f'[Epoch: {epoch}] Maximum downsampling ratio : {max(resize_ratios_all):.2%}')
        print(f'[Epoch: {epoch}] Minimum downsampling ratio : {min(resize_ratios_all):.2%}')

    if verbose:
        save_graph(to, r, a, f, filename)
    torch.save(model.state_dict(), SAVE_PATH+"models/"+filename)


def test_model(test_dataset, device, filename,writer):
    loaded_model = get_model_instance_segmentation(num_classes=NO_OF_CLASSES+1)
    loaded_model.load_state_dict(torch.load(SAVE_PATH+"models/"+filename))

    # print("Ground Truth \t\t Label \t\t\t BoxIndex \t\t IOU Score")
    print('{:^3} {:^20} {:^20} {:^25} {:^20} {:^10} {:^10}'.format('Index',
        'Image', 'Ground Truth', 'Label', 'Box Index', 'Confidence Score', 'IOU Score'))
    ocr_counter = 0
    label_counter = 0
    total = 0
    ocr_fail = 0
    for i in range(len(test_dataset)):
        img, _, resize_ratio = test_dataset[i]
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
                total+=1
            for element in range(len(prediction[0]["boxes"])):
                boxes = prediction[0]["boxes"][element].cpu().numpy()
                score = np.round(
                    prediction[0]["scores"][element].cpu().numpy(), decimals=4)
                label = prediction[0]["labels"][element].cpu().numpy()
                label = list(class_dictionary.keys())[
                    list(class_dictionary.values()).index(label)]
                try:
                    label_match = 0
                    if score > 0.7:
                        # label_via_ocr = label_ocr(_['name'], boxes, label)
                        label_match = iou(label, boxes, dict1, score, _[
                                        'name'],i)
                        
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

            print("-"*150)

    try:
        print("ACCURACY OF OCR:", ocr_counter/(total-ocr_fail))
    except:
        print("OCR FAILED")
    print(label_counter,total)
    print("ACCURACY OF prediction:", label_counter/total)
    writer.add_scalar("Accuracy/test",label_counter/total,1)


def iou(label, box1, box2, score, name, index,label_ocr="No OCR"):
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
            print("Error: ",e)
            print(iou_list)
            continue
    score1 = '%.2f'%(score)
    try:
        print('{:^3} {:^20} {:^20} {:^25} {:^20} {:^10} {:^10}'.format(index,
            name, box2[label_index]["label"], label, label_index, score1 , round(max(iou_list),2)))
    except:
        print('{:^3} {:^20} {:^20} {:^25} {:^20} {:^10} {:^10}'.format(index,
            name, "None", label, label_index, score1 , "None"))
        return 0
    if box2[label_index]["label"] == label:
        return 1
    else:
        return 0


ocr_dict = {
    "Gould-Ferraz Shawmut A4J": ['Gould', 'Shawmut', 'Amp-trap', 'Class J', 'Current', 'Limiting', 'A4J###', '### Amps.', '### Amp.', '### Amp', '600 VAC or Less', 'HRC I-J', 'UND. LAB. INC.', 'LISTED FUSE', 'Gould Inc.', 'Newburyport, MA', 'Gould Shawmut', 'Toronto, Canada', '600VAC', '200kA I.R.', '300VDC 20kA I.R.', 'Ferraz Shawmut', 'Certified', 'Assy. in Mexico', 'Assy. in CANADA', '200kA IR 600V AC', '20kA IR 300V DC', '600 V.A.C. or Less', 'Interrupting Rating', '200,000 RMS. SYM. Amps.', 'Assembled in Mexico', 'IR 200kA', 'Electronics Inc.', 'U.S. Pat. Nos. 4,216,457:', '4,300,281 and 4,320,376', 'Newburyport, MA 01950', '600 V ~', '200k A I.R.', 'PATENTED', 'Ferraz', '200kA IR AC', '300VDC or Less', '100kA IR DC', 'LISTED', 'FUSE', 'SA'],
    "Ferraz Shawmut AJT": ['AMP-TRAP', '2000', 'DUAL ELEMENT', 'TIME DELAY', 'Smart', 'SpOt', 'AJT###', '###A', '600V AC', '500V DC', 'Ferraz', 'Shawmut', 'Any Red - Open', 'Amp-Trap 2000', 'Dual Element', 'Time Delay', 'Mersen', 'Ferraz Shawmut', 'Newburyport, MA 01950', 'Class J Fuse', '200,000A IR 600V AC', '100,000A IR 500V DC', 'Current Limiting', 'LISTED', 'FUSE', 'ISSUE NO.: ND57-62', 'Ferraz Shawmut Certified', '300,000A IR 600V AC', '600V AC/500V DC', 'Any Red = Open'],
    "English Electric C-J": ['ENGLISH', 'ELECTRIC', 'HRC', 'ENERGY', 'LIMITING FUSE', '### AMPS', '600 VOLTS A.C. OR LESS', 'CATALOG No. C###J', 'ENSURE CONTINUED', 'SAFE PROTECTION', 'REPLACE WITH', 'CATALOG No.', 'Made in England', 'CSA SPEC', 'CSA STD.', 'HRC1', 'C22.2-106', 'TESTED AT', '200,000 AMPS', 'EASTERN ELECTRIC', '5775 Ferrier Street.', 'Montreal PQ Canada'],
    "Ferraz Shawmut CRS": ['Ferraz', 'Shawmut', 'Type D', 'Time Delay', 'Temporise', 'CRS###amp', '600V A.C. or less', 'C.A. ou moins', '10kA IR', 'Ferraz Shawmut', 'Toronto, Canada', 'cefcon', 'LR14742', 'Mexico'],
    "Gould Shawmut AJT": ['AMP-TRAP', '2000', 'TIME DELAY', 'AJT###', '##A 600V AC', 'GOULD SHAWMUT', 'HRCI-J', 'UND. LAB. INC.', 'LISTED FUSE', 'U.S. PAT. NO. 4,344,058', 'Dual Element', 'Time Delay', '## Amp.', '600V AC', '500V DC', 'DUAL ELEMENT', 'Class J Fuse', '200,000A IR 600V AC', '100,000A IR 500V DC', 'Current Limiting', 'Gould Certified', '300,000A IR 600V AC', 'Gould Shawmut', '(508) 462-6662', 'Gould, Inc.', 'Newburyport, Mass., U.S.A.', 'Toronto, Ontario, Canada', 'Made in U.S.A.', 'U.S. Pat. 4,320,376', 'Nos. 4,300,281'],
    "GEC HRC I-J": ['GEC', 'HRC I-J', 'Rating', '### Amp', 'CSA', 'C22.2', 'No. 106-M1985', 'IR 200kA', '~ 60Hz', '600', 'VOLTS', 'Can. Pat. No. 148995', '188', 'Made in English', 'C###J', 'Cat No.', 'No. 106-M92', 'GEC ALSTHOM'],
    "Gould Shawmut TRSR": ['GOULD', 'Shawmut', 'Tri-onic', 'TRSR ###', 'Time Delay', 'Temporisé', 'HRCI-R', '###A', 'LR14742', '600V ~', '200k.A.I.R', 'Dual Element', '600 V AC', '600 V DC', '300V DC', '600V AC', 'Current Limiting', 'Class RK5 Fuse', 'UND. LAB. INC.', 'LISTED FUSE', '200.000A IR', '20.000A IR', 'Gould Shawmut', '198L', '(508) 462-6662', 'Action', 'Temporisée', 'HRC I', '600V A.C. or less', 'C.A. ou moins', 'TRS###R', '### Amps', '600 VAC or Less'],
    "English Electric Form II": ['THE CAT. No.', 'AND RATING', '(MARKED ON THIS CAP)', 'SHOULD BE QUOTED', 'WHEN RE-ORDERING', 'ENGLISH', 'ELECTRIC', 'TESTED AT', '200,000 Amps', 'FORM II', 'H.R.C. FUSE', 'SA', 'C.S.A.Spec.C22-2No.106', 'EASTERN ELECTRIC COMPANY LTD.', '600', 'VOLTS', 'or less', 'A.C. 60 cycle', 'EASTERN ELECTRIC FUSE PATENTED', 'CF###A', 'CC.###', 'CAT.NO.CC###.', 'Complies with', 'IEC 269-2', 'CSA STD', 'C22-2', 'No 106', 'Tested at', '200,000 Amps', '600V (or less)', 'AC 60HZ', '100,000 AMP RMS ASYM', 'C.S.A. APP. N°12203.', '600V. 60 CYCLE A.C.', 'FORM II.H.R.C.FUSE'],
    "Bussmann LPJ": ['BUSS', 'LOW-PEAK', 'DUAL-ELEMENT TIME-DELAY', 'FUSE', 'LPJ-###SP', '600 VAC OR LESS', 'CURRENT LIMITING', 'AMP', 'THIS FUSE MAY SUBSTITUTE FOR', 'A LISTED CLASS J FUSE', 'HRCI-J', 'IR 200kA AC', 'IR 100kA DC', 'TYPE D', 'UL LISTED', 'SPECIAL PURPOSE FUSE FP33-##', 'IR 300kA AC, IR 100kA DC', '600 VAC', '300 VDC', 'CLASS J', 'LISTED FUSE DL92-##', 'Bussmann LPJ', 'LOW-PEAK', 'ULTIMATE PROTECTION', 'CLASS J FUSE', '600Vac', 'AC IR 300kA', '300Vdc', 'DC IR 100kA', 'Self-certified DC rating', 'Cooper Bussmann, LLC', 'St. Louis, MO 63178', 'Assembled in Mexico', 'www.bussmann.com', 'Cooper Industries', 'Bussmann Division', 'St. Louis, MO', 'MADE IN U.S.A.', 'LISTED SPECIAL PURPOSE'],
    "Gould Shawmut CJ": ['GOULD', 'Shawmut', 'CJ ###', 'HRCI-J', '###A', 'LR14742', 'Class J', 'Cat. No.', '### Amps', 'Amp-trap', '600 V.A.C. or less', '200,000 Amps A.C.', 'Interrupting Rating', 'Current Limiting', '600 V.C.A. ou moins', '200,000 Amps C.A.', 'Intensité de Rupture', 'Limitant de Courant', '200,000 A.I.R.', 'Mfd. By/Fab. Par', 'Gould Shawmut', 'Toronto, Canada', 'Int. De Rupt.', 'Int. Rating', 'Gould Elec. Fuse Div.', 'HRC I', '200k A.I.R.', '600V ~']
}


def label_ocr(img, box, label):
    name = ''.join(chr(i) for i in img)
    path = os.path.join(TRAIN_DATAPATH, "images", name)
    image = cv2.imread(path)
    x1, y1, x2, y2 = box

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    # print(path)

    resultList = []
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
        dataList_1 = re.split(r',|\.|\n| ', data_1)  # split the string
        resultList.append([(i.strip()) for i in dataList_1 if i != ''])
    except Exception as e:
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
        dataList_2 = re.split(r',|\.|\n| ', data_2)
        resultList.append([(i.strip()) for i in dataList_2 if i != ''])
    except Exception as e:
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

    resultList = list(chain.from_iterable(resultList))
    # print("OCR Recognized List",resultList)
    count = 0
    out_count = 0
    # key = "Ferraz Shawmut CRS"
    # for i in range(100):
    #   ocr_dict_rank[key] += 1
    try:
        for list_item in resultList:
            for key, value in ocr_dict.items():
                for v in value:
                    if fuzz.partial_ratio(v.lower(), list_item.lower()) > 90 and len(list_item) > 3:
                        ocr_dict_rank[key] = ocr_dict_rank.get(key, 0) + 1
                    elif fuzz.partial_ratio(v.lower(), list_item.lower()) > 75 and len(list_item) > 3:
                        ocr_dict_rank[key] = ocr_dict_rank.get(key, 0) + 0.5
                        # print("v: [{0}] item: [{1}] fuzz: [{2}] fuzz_reverse: [{3}]".format(v.lower(),list_item.lower(),fuzz.partial_ratio(v.lower(),list_item.lower()),fuzz.partial_ratio(list_item.lower(),v.lower())))
                        # print(key)

    except Exception as e:
        print(e)

    sorted_d = dict(sorted(ocr_dict_rank.items(),
                           key=operator.itemgetter(1), reverse=True))
    if sorted_d[list(sorted_d.keys())[0]] > 1:
        print(sorted_d)
        return list(sorted_d.keys())[0]
    else:
        return "ocr fail"


def view_test_image(idx,test_dataset,filename):
    loaded_model = get_model_instance_segmentation(num_classes=NO_OF_CLASSES+1)
    loaded_model.load_state_dict(torch.load(SAVE_PATH+"models/"+filename))
    img, _ = test_dataset[idx]
    img_name = ''.join(chr(i) for i in _['name'])
    print(img_name)
    
    label_boxes = np.array(test_dataset[idx][1]["boxes"])
    # put the model in evaluation mode
    loaded_model.eval()
    with torch.no_grad():
        prediction = loaded_model([img])
    image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    draw = ImageDraw.Draw(image)
    # draw groundtruth
    for elem in range(len(label_boxes)):
        label = test_dataset[idx][1]["labels"][elem].cpu().numpy()
        label = list(class_dictionary.keys())[
            list(class_dictionary.values()).index(label)]
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
        label = list(class_dictionary.keys())[
            list(class_dictionary.values()).index(label)]
        if score > 0.5:
            draw.rectangle(
                [(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline="red", width=5)
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 50)
            draw.text((boxes[0]-1, boxes[3]), text=label + " " +
                    str(score), font=font, fill=(255, 255, 255, 0))
    image = image.save("image_save_1/"+img_name)
