import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


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
