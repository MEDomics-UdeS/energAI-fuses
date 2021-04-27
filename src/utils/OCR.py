"""
File:
    src/utils/OCR.py

Authors:
    - Simon Giard-Leroux
    - Shreyas Sunil Kulkarni

Description:
    Optical character recognition (OCR) function to detect text on fuse labels.

NOTE : Has not been tested/refactored yet.
"""

import operator
import os
import re
from itertools import chain
import cv2
import pytesseract
from fuzzywuzzy import fuzz
from typing import Tuple

from constants import RAW_PATH, OCR_DICT


def label_ocr(img: str, box: Tuple[float]) -> str:
    """

    :param img:
    :param box:
    :return:
    """
    name = ''.join(chr(i) for i in img)
    path = os.path.join(RAW_PATH, name)
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
