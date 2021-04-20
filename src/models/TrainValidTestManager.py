import os
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torch import tensor
from torch.nn.functional import softmax
from src.data.DataLoaderManager import DataLoaderManager
from tqdm import tqdm
from typing import Optional
import torchvision.models as models
import torch.nn as nn
from constants import *
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.models.detection as detection
from constants import CLASS_DICT
from src.models.EarlyStopping import EarlyStopping


class TrainValidTestManager:
    """
    Training, validation and testing manager.
    """

    def __init__(self, data_loader_manager: DataLoaderManager,
                 file_name: Optional[str],
                 model_name: str,
                 learning_rate: float,
                 weight_decay: float,
                 early_stopping,
                 pretrained: bool = False) -> None:

        self.num_classes = len(CLASS_DICT) + 1 # + 1 to include background class
        self.early_stopping = early_stopping

        # Extract the training, validation and testing data loaders
        self.data_loader_train = data_loader_manager.data_loader_train
        self.data_loader_valid = data_loader_manager.data_loader_valid
        self.data_loader_test = data_loader_manager.data_loader_test
        self.batch_size = data_loader_manager.batch_size

        # Save the file name
        self.file_name = file_name

        # Define device as the GPU if available, else use the CPU
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Get model and set last fully-connected layer with the right number of classes
        self.load_model(name=model_name, num_classes=self.num_classes, pretrained=pretrained)

        # Define loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # Find which parameters to train (those with .requires_grad = True)
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Define Adam optimizer
        self.optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

        # Send the model to the device
        self.model.to(self.device)

        # Results dictionary
        self.results = {
            'Training Loss': [],
            'Training Accuracy': [],
            'Validation Loss': [],
            'Validation Accuracy': []
        }

    def train_model(self, epochs: int) -> None:
        """
        Trains the model and saves the trained model.

        :param epochs: int, number of epochs
        """
        # Specify that the model will be trained
        self.model.train()

        if self.early_stopping:
            es = EarlyStopping(patience=self.early_stopping)

        # Declare tqdm progress bar
        pbar = tqdm(total=len(self.data_loader_train), leave=False,
                    desc='Epoch 0', postfix='Training Loss: 0')

        # Main training loop, loop through each epoch
        for epoch in range(epochs):
            # Declare empty loss and accuracy lists for the current epoch
            loss_list_epoch = []
            accuracy_list_epoch = []

            # Loop through each mini-batch from the data loader
            for i, (images, labels) in enumerate(self.data_loader_train):
                # Send images and labels to the device
                images, labels = images.to(self.device), labels.to(self.device)

                # Reset all gradients to zero
                self.optimizer.zero_grad()

                # Perform a forward pass
                outputs = self.model.forward(images)

                # Calculate the loss, comparing outputs with the ground truth labels
                loss = self.criterion(outputs, labels)

                # Appending the current loss to the loss list and the current accuracy to the accuracy list
                loss_list_epoch.append(loss.item())
                accuracy_list_epoch.append(self.get_accuracy(outputs, labels))

                # Perform a backward pass (calculate gradient)
                loss.backward()

                # Perform a parameter update based on the current gradient
                self.optimizer.step()

                # Update progress bar
                pbar.set_description_str(f'Epoch {epoch}')
                pbar.set_postfix_str(f'Training Loss: {loss:.5f}')
                pbar.update()

            # Reset progress bar after epoch completion
            pbar.reset()

            # Save the training loss and accuracy in the object
            self.results['Training Loss'].append(round(np.mean(loss_list_epoch), 5))
            self.results['Training Accuracy'].append(round(np.mean(accuracy_list_epoch), 5))

            # Validate the model
            self.validate_model()

        # We close the tqdm bar
        pbar.close()

        # If file_name is specified, save the trained model
        if self.file_name is not None:
            torch.save(self.model.state_dict(), f'models/{self.file_name}')

    def validate_model(self) -> None:
        """
        Method to validate the model saved in the self.model class attribute.

        :return: None
        """
        # Specify that the model will be evaluated
        self.model.eval()

        # Declare empty loss and accuracy lists
        loss_list = []
        accuracy_list = []

        # Deactivate the autograd engine
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.data_loader_valid_1):
                # Send images and labels to the device
                images, labels = images.to(self.device), labels.to(self.device)

                # Perform a forward pass
                outputs = self.model.forward(images)

                # Calculate the loss, comparing outputs with the ground truth labels
                loss = self.criterion(outputs, labels)

                # Appending the current loss to the loss list and current accuracy to the accuracy list
                loss_list.append(loss.item())
                accuracy_list.append(self.get_accuracy(outputs, labels))

        # Calculate mean loss and mean accuracy over all batches
        mean_loss = np.mean(loss_list)
        mean_accuracy = np.mean(accuracy_list)

        # Save mean loss and mean accuracy in the object
        self.results['Validation Loss'].append(round(mean_loss, 5))
        self.results['Validation Accuracy'].append(round(mean_accuracy, 5))

    def test_model(self, final_eval: bool = False) -> float:
        """
        Method to test the model saved in the self.model class attribute.

        :param final_eval: bool indicating we are evaluating the model on the final test set
                           after active learning is done
        :return: None
        """
        # Specify that the model will be evaluated
        self.model.eval()

        # Initialize empty accuracy list
        accuracy_list = []

        # We select the good loader
        loader = self.data_loader_test if final_eval else self.data_loader_valid_2

        # Deactivate the autograd engine
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):
                # Send images and labels to the device
                images, labels = images.to(self.device), labels.to(self.device)

                # Perform a forward pass
                outputs = self.model.forward(images)

                # Calculate the accuracy for the current batch
                accuracy_list.append(self.get_accuracy(outputs, labels))

        # Print mean test accuracy over all batches
        mean_accuracy = np.mean(accuracy_list)
        return mean_accuracy

    def load_model(self,
                   name: str,
                   num_classes: int,
                   pretrained: bool = False,
                   progress: bool = True,
                   pretrained_backbone: bool = True,
                   trainable_backbone_layers: Optional[int] = None):

        if name == 'fasterrcnn_resnet50_fpn':
            self.model = detection.fasterrcnn_resnet50_fpn(pretrained=pretrained,
                                                           progress=progress,
                                                           num_classes=num_classes,
                                                           pretrained_backbone=pretrained_backbone,
                                                           trainable_backbone_layers=trainable_backbone_layers)
        elif name == 'fasterrcnn_mobilenet_v3_large_fpn':
            self.model = detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=pretrained,
                                                                     progress=progress,
                                                                     num_classes=num_classes,
                                                                     pretrained_backbone=pretrained_backbone,
                                                                     trainable_backbone_layers=trainable_backbone_layers)
        elif name == 'fasterrcnn_mobilenet_v3_large_320_fpn':
            self.model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=pretrained,
                                                                         progress=progress,
                                                                         num_classes=num_classes,
                                                                         pretrained_backbone=pretrained_backbone,
                                                                         trainable_backbone_layers=trainable_backbone_layers)
        elif name == 'retinanet_resnet50_fpn':
            self.model = detection.retinanet_resnet50_fpn(pretrained=pretrained,
                                                          progress=progress,
                                                          num_classes=num_classes,
                                                          pretrained_backbone=pretrained_backbone,
                                                          trainable_backbone_layers=trainable_backbone_layers)
        elif name == 'maskrcnn_resnet50_fpn':
            self.model = detection.maskrcnn_resnet50_fpn(pretrained=pretrained,
                                                         progress=progress,
                                                         num_classes=num_classes,
                                                         pretrained_backbone=pretrained_backbone,
                                                         trainable_backbone_layers=trainable_backbone_layers)
        elif name == 'keypointrcnn_resnet50_fpn':
            self.model = detection.keypointrcnn_resnet50_fpn(pretrained=pretrained,
                                                             progress=progress,
                                                             num_classes=num_classes,
                                                             pretrained_backbone=pretrained_backbone,
                                                             trainable_backbone_layers=trainable_backbone_layers)

        # model = self.replace_model_head(name, model, num_classes)

    # @staticmethod
    # def replace_model_head(name, model, num_classes):
    #     in_features = model.roi_heads.box_predictor.cls_score.in_features
    #
    #     if name.instr('fasterrcnn'):
    #         model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features=in_features,
    #                                                                                 num_classes=num_classes)
    #     elif name.instr('retinanet'):
    #         model.roi_heads.box_predictor = detection.retinanet.RetinaNetHead(in_features=in_features,
    #                                                                           num_classes=num_classes)
    #     elif name.instr('')
    #             f
    #         a
    #
    #

    # from .faster_rcnn import *
    # from .mask_rcnn import *
    # from .keypoint_rcnn import *
    # from .retinanet import *

    # if name == RESNET18:
    #     if pretrained:
    #         # If pretrained, an error occur if num_classes != 1000,
    #         # we have to initialize and THEN change the last layer
    #         m = models.resnet18(pretrained)
    #         m.fc = nn.Linear(512, num_classes)
    #     else:
    #         # If not pretrained, the last layer can be of any size, hence we can do both step
    #         # in one and avoid initializing last layer twice
    #         m = models.resnet18(pretrained, num_classes=num_classes)
    # else:
    #     if pretrained:
    #         # If pretrained, an error occur if num_classes != 1000,
    #         # we have to initialize and THEN change the last layer
    #         m = models.squeezenet1_1(pretrained)
    #         m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    #     else:
    #         # If not pretrained, the last layer can be of any size, hence we can do both step
    #         # in one and avoid initializing last layer twice
    #         m = models.squeezenet1_1(pretrained, num_classes=num_classes)
    # return m

    @staticmethod
    def get_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Method to calculate accuracy of predicted outputs vs ground truth labels.

        :param outputs: torch.Tensor, predicted outputs classes
        :param labels: torch.Tensor, ground truth labels classes
        :return: float, accuracy of the predicted outputs vs the ground truth labels
        """
        return (outputs.argmax(dim=1) == labels).sum().item() / labels.shape[0]
