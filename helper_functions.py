import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from fuzzywuzzy import fuzz
import operator
import os
import cv2
import pytesseract
import re
from itertools import chain
from fuse_config import TRAIN_DATAPATH


"""
converts the image to a ten
"""
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


def collate_fn(batch):
    return tuple(zip(*batch))

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def iou(label,box1,box2,score,name,label_ocr="No OCR"):
  name = ''.join(chr(i) for i in name)
  # print('{:^30}'.format(name[-1]))
  iou_list = []
  iou_label = []
  for item in box2:
    try:
      x11,y11,x21,y21 = item["boxes"]
      x12,y12,x22,y22 = box1
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
      print(e)
      continue

  print('{:^30} {:^30} {:^30} {:^30} {:^30} {:^30} {:^30}'.format(name,box2[label_index]["label"], label, label_ocr, label_index,score, max(iou_list)))
  if box2[label_index]["label"] == label:
    return 1
  else:
    return 0


ocr_dict = {
    "Gould-Ferraz Shawmut A4J": ['Gould','Shawmut','Amp-trap','Class J','Current','Limiting','A4J###','### Amps.','### Amp.','### Amp','600 VAC or Less','HRC I-J','UND. LAB. INC.','LISTED FUSE','Gould Inc.','Newburyport, MA','Gould Shawmut','Toronto, Canada','600VAC','200kA I.R.','300VDC 20kA I.R.','Ferraz Shawmut','Certified','Assy. in Mexico','Assy. in CANADA','200kA IR 600V AC','20kA IR 300V DC','600 V.A.C. or Less','Interrupting Rating','200,000 RMS. SYM. Amps.','Assembled in Mexico','IR 200kA','Electronics Inc.','U.S. Pat. Nos. 4,216,457:','4,300,281 and 4,320,376','Newburyport, MA 01950','600 V ~','200k A I.R.','PATENTED','Ferraz','200kA IR AC','300VDC or Less','100kA IR DC','LISTED','FUSE','SA'],
    "Ferraz Shawmut AJT":['AMP-TRAP','2000','DUAL ELEMENT','TIME DELAY','Smart','SpOt','AJT###','###A','600V AC','500V DC','Ferraz','Shawmut','Any Red - Open','Amp-Trap 2000','Dual Element','Time Delay','Mersen','Ferraz Shawmut','Newburyport, MA 01950','Class J Fuse','200,000A IR 600V AC','100,000A IR 500V DC','Current Limiting','LISTED','FUSE','ISSUE NO.: ND57-62','Ferraz Shawmut Certified','300,000A IR 600V AC','600V AC/500V DC','Any Red = Open'],
    "English Electric C-J": ['ENGLISH','ELECTRIC','HRC','ENERGY','LIMITING FUSE','### AMPS','600 VOLTS A.C. OR LESS','CATALOG No. C###J','ENSURE CONTINUED','SAFE PROTECTION','REPLACE WITH','CATALOG No.','Made in England','CSA SPEC','CSA STD.','HRC1','C22.2-106','TESTED AT','200,000 AMPS','EASTERN ELECTRIC','5775 Ferrier Street.','Montreal PQ Canada'],
    "Ferraz Shawmut CRS": ['Ferraz','Shawmut','Type D','Time Delay','Temporise','CRS###amp','600V A.C. or less','C.A. ou moins','10kA IR','Ferraz Shawmut','Toronto, Canada','cefcon','LR14742','Mexico'],
    "Gould Shawmut AJT" : ['AMP-TRAP','2000','TIME DELAY','AJT###','##A 600V AC','GOULD SHAWMUT','HRCI-J','UND. LAB. INC.','LISTED FUSE','U.S. PAT. NO. 4,344,058','Dual Element','Time Delay','## Amp.','600V AC','500V DC','DUAL ELEMENT','Class J Fuse','200,000A IR 600V AC','100,000A IR 500V DC','Current Limiting','Gould Certified','300,000A IR 600V AC','Gould Shawmut','(508) 462-6662','Gould, Inc.','Newburyport, Mass., U.S.A.','Toronto, Ontario, Canada','Made in U.S.A.','U.S. Pat. 4,320,376','Nos. 4,300,281'],
    "GEC HRC I-J" : ['GEC','HRC I-J','Rating','### Amp','CSA','C22.2','No. 106-M1985','IR 200kA','~ 60Hz','600','VOLTS','Can. Pat. No. 148995','188','Made in English','C###J','Cat No.','No. 106-M92','GEC ALSTHOM'],
    "Gould Shawmut TRSR" : ['GOULD','Shawmut','Tri-onic','TRSR ###','Time Delay','Temporisé','HRCI-R','###A','LR14742','600V ~','200k.A.I.R','Dual Element','600 V AC','600 V DC','300V DC','600V AC','Current Limiting','Class RK5 Fuse','UND. LAB. INC.','LISTED FUSE','200.000A IR','20.000A IR','Gould Shawmut','198L','(508) 462-6662','Action','Temporisée','HRC I','600V A.C. or less','C.A. ou moins','TRS###R','### Amps','600 VAC or Less'],
    "English Electric Form II" :['THE CAT. No.','AND RATING','(MARKED ON THIS CAP)','SHOULD BE QUOTED','WHEN RE-ORDERING','ENGLISH','ELECTRIC','TESTED AT','200,000 Amps','FORM II','H.R.C. FUSE','SA','C.S.A.Spec.C22-2No.106','EASTERN ELECTRIC COMPANY LTD.','600','VOLTS','or less','A.C. 60 cycle','EASTERN ELECTRIC FUSE PATENTED','CF###A','CC.###','CAT.NO.CC###.','Complies with','IEC 269-2','CSA STD','C22-2','No 106','Tested at','200,000 Amps','600V (or less)','AC 60HZ','100,000 AMP RMS ASYM','C.S.A. APP. N°12203.','600V. 60 CYCLE A.C.','FORM II.H.R.C.FUSE'],
    "Bussmann LPJ" : ['BUSS','LOW-PEAK','DUAL-ELEMENT TIME-DELAY','FUSE','LPJ-###SP','600 VAC OR LESS','CURRENT LIMITING','AMP','THIS FUSE MAY SUBSTITUTE FOR','A LISTED CLASS J FUSE','HRCI-J','IR 200kA AC','IR 100kA DC','TYPE D','UL LISTED','SPECIAL PURPOSE FUSE FP33-##','IR 300kA AC, IR 100kA DC','600 VAC','300 VDC','CLASS J','LISTED FUSE DL92-##','Bussmann LPJ','LOW-PEAK','ULTIMATE PROTECTION','CLASS J FUSE','600Vac','AC IR 300kA','300Vdc','DC IR 100kA','Self-certified DC rating','Cooper Bussmann, LLC','St. Louis, MO 63178','Assembled in Mexico','www.bussmann.com','Cooper Industries','Bussmann Division','St. Louis, MO','MADE IN U.S.A.','LISTED SPECIAL PURPOSE'],
    "Gould Shawmut CJ" : ['GOULD','Shawmut','CJ ###','HRCI-J','###A','LR14742','Class J','Cat. No.','### Amps','Amp-trap','600 V.A.C. or less','200,000 Amps A.C.','Interrupting Rating','Current Limiting','600 V.C.A. ou moins','200,000 Amps C.A.','Intensité de Rupture','Limitant de Courant','200,000 A.I.R.','Mfd. By/Fab. Par','Gould Shawmut','Toronto, Canada','Int. De Rupt.','Int. Rating','Gould Elec. Fuse Div.','HRC I','200k A.I.R.','600V ~']
}

def label_ocr(img,box,label):
  name = ''.join(chr(i) for i in img)
  path = os.path.join(TRAIN_DATAPATH, "images", name)
  image = cv2.imread(path)
  x1,y1,x2,y2 = box
  
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

    data_1 = pytesseract.image_to_string(gray_1, lang='eng', config='-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz --psm 11')
    dataList_1 = re.split(r',|\.|\n| ',data_1) # split the string
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

    data_2 = pytesseract.image_to_string(gray_2, lang='eng', config='-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz --psm 11')
    dataList_2 = re.split(r',|\.|\n| ',data_2)
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
      for key,value in ocr_dict.items():
        for v in value:
          if fuzz.partial_ratio(v.lower(),list_item.lower()) > 90 and len(list_item) > 3:
            ocr_dict_rank[key] = ocr_dict_rank.get(key, 0) + 1
          elif fuzz.partial_ratio(v.lower(),list_item.lower()) >75 and len(list_item) > 3:
            ocr_dict_rank[key] = ocr_dict_rank.get(key, 0) + 0.5
            # print("v: [{0}] item: [{1}] fuzz: [{2}] fuzz_reverse: [{3}]".format(v.lower(),list_item.lower(),fuzz.partial_ratio(v.lower(),list_item.lower()),fuzz.partial_ratio(list_item.lower(),v.lower())))
            # print(key)
            
  except Exception as e:
    print(e)

  sorted_d = dict( sorted(ocr_dict_rank.items(), key=operator.itemgetter(1),reverse=True))
  if sorted_d[list(sorted_d.keys())[0]]>1:
    print(sorted_d)
    return list(sorted_d.keys())[0]
  else:
    return "ocr fail"