"""
File:
    src/models/models.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    File to load different models
"""

import torch
import torchvision.models.detection as detection
from typing import Optional, Any


def load_model(model_name: str,
               pretrained: bool,
               num_classes: int,
               progress: bool = True,
               trainable_backbone_layers: Optional[int] = None) -> Any:
    """Method to load a model from PyTorch

    Args:
        model_name(str): model name
        pretrained(bool): choose whether to load a pretrained model or a model from scratch
        num_classes(int): number of classes for last fully-connected layer
        progress(bool, optional): if True, displays a progress bar of the download to stderr (Default value = True)
        trainable_backbone_layers(Optional[int], optional): number of trainable (not frozen) resnet layers starting from final block.
                                                            Valid values are between 0 and 5, with 5 meaning all backbone layers are
                                                            trainable. (Default value = None)

    Returns:

    """
    # Check for specified model name, load corresponding model and replace model head with right number of classes
    if model_name == 'fasterrcnn_resnet50_fpn':
        model = detection.fasterrcnn_resnet50_fpn(pretrained=pretrained,
                                                  progress=progress,
                                                  pretrained_backbone=pretrained,
                                                  trainable_backbone_layers=trainable_backbone_layers)
        model = replace_model_head(model, model_name, num_classes)

    elif model_name == 'fasterrcnn_mobilenet_v3_large_fpn':
        model = detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=pretrained,
                                                            progress=progress,
                                                            pretrained_backbone=pretrained,
                                                            trainable_backbone_layers=trainable_backbone_layers)
        model = replace_model_head(model, model_name, num_classes)

    elif model_name == 'fasterrcnn_mobilenet_v3_large_320_fpn':
        model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=pretrained,
                                                                progress=progress,
                                                                pretrained_backbone=pretrained,
                                                                trainable_backbone_layers=trainable_backbone_layers)
        model = replace_model_head(model, model_name, num_classes)

    elif model_name == 'retinanet_resnet50_fpn':
        model = detection.retinanet_resnet50_fpn(pretrained=pretrained,
                                                 progress=progress,
                                                 pretrained_backbone=pretrained,
                                                 trainable_backbone_layers=trainable_backbone_layers)
        model = replace_model_head(model, model_name, num_classes)

    elif model_name == 'detr':
        model = torch.hub.load('facebookresearch/detr',
                               'detr_resnet50', pretrained=False, num_classes=num_classes)
        
        if pretrained:
            model = load_detr_state_dict(model)

    else:
        raise NotImplementedError

    return model


def replace_model_head(model: Any,
                       model_name: str,
                       num_classes: int) -> Any:
    """Replace model head with the right number of classes (for transfer learning)

    Args:
        model(Any): model
        model_name(str): model name
        num_classes(int): number of classes for last fully-connected layer

    Returns:

    """
    if 'fasterrcnn' in model_name:
        in_channels = model.roi_heads.box_predictor.cls_score.in_features

        model.roi_heads.box_predictor = \
            detection.faster_rcnn.FastRCNNPredictor(in_channels=in_channels,
                                                    num_classes=num_classes)

    elif 'retinanet' in model_name:
        in_channels = model.backbone.out_channels
        num_anchors = model.head.classification_head.num_anchors

        model.head = detection.retinanet.RetinaNetHead(in_channels=in_channels,
                                                       num_anchors=num_anchors,
                                                       num_classes=num_classes)
        
    else:
        raise NotImplementedError

    return model


def load_detr_state_dict(model):
    """Load pretrained weights for DE:TR model

    Args:
        model: DETR model

    """
    checkpoint = torch.hub.load_state_dict_from_url(
        url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
        map_location='cpu',
        check_hash=True)

    del checkpoint["model"]["class_embed.weight"]
    del checkpoint["model"]["class_embed.bias"]

    model.load_state_dict(checkpoint["model"], strict=False)

    return model
