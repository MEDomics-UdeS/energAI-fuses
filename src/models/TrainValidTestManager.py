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
from torch.utils.tensorboard import SummaryWriter
import torch.cuda.amp as amp
from torch.cuda import memory_reserved, memory_allocated


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
                 mixed_precision,
                 gradient_accumulation,
                 pretrained: bool = False) -> None:
        self.train_step = 0
        self.valid_step = 0
        self.total_step = 0

        self.pretrained = pretrained
        self.model_name = model_name

        self.writer = SummaryWriter('logdir/' + file_name)
        self.mixed_precision = mixed_precision
        self.accumulation_size = gradient_accumulation
        self.gradient_accumulation = False if gradient_accumulation == 1 else True

        self.num_classes = len(CLASS_DICT) + 1 # + 1 to include background class
        self.early_stopping = early_stopping

        # Extract the training, validation and testing data loaders
        self.data_loader_train = data_loader_manager.data_loader_train
        self.data_loader_valid = data_loader_manager.data_loader_valid
        self.data_loader_test = data_loader_manager.data_loader_test
        # self.batch_size = data_loader_manager.batch_size

        # Save the file name
        self.file_name = file_name

        # Define device as the GPU if available, else use the CPU
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Get model and set last fully-connected layer with the right number of classes
        self.load_model()

        # Define loss function
        # self.criterion = torch.nn.CrossEntropyLoss()

        # Find which parameters to train (those with .requires_grad = True)
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Define Adam optimizer
        # self.optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0005)
        self.optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)

        # Send the model to the device
        self.model.to(self.device)

        # Results dictionary
        # self.results = {
        #     'Training Loss': [],
        #     'Training Accuracy': [],
        #     'Validation Loss': [],
        #     'Validation Accuracy': []
        # }

        print(f'\n=== Dataset & Data Loader Sizes ===\n\n'
              f'Training:\t\t{len(self.data_loader_train.dataset)} images\t\t{len(self.data_loader_train)} batches\n'
              f'Validation:\t\t{len(self.data_loader_valid.dataset)} images\t\t{len(self.data_loader_valid)} batches\n'
              f'Testing:\t\t{len(self.data_loader_test.dataset)} images\t\t{len(self.data_loader_test)} batches\n')

    def train_model(self, epochs: int) -> None:
        """
        Trains the model and saves the trained model.

        :param epochs: int, number of epochs
        """
        # Specify that the model will be trained
        self.model.train()

        if self.early_stopping:
            es = EarlyStopping(patience=self.early_stopping)

        scaler = amp.grad_scaler.GradScaler(enabled=self.mixed_precision)

        for epoch in range(epochs):
            # Declare tqdm progress bar
            pbar = tqdm(total=len(self.data_loader_train), leave=False,
                        desc=f'Training Epoch {epoch}', postfix='Loss: 0')

            if self.gradient_accumulation:
                self.optimizer.zero_grad()

            for i, (images, targets) in enumerate(self.data_loader_train):
                images = torch.stack(images).to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                with amp.autocast(enabled=self.mixed_precision):
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                if self.gradient_accumulation and self.mixed_precision:
                    scaler.scale(losses).backward()

                    if (i + 1) % self.accumulation_size == 0:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRAD_CLIP)
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                elif self.gradient_accumulation and not self.mixed_precision:
                    losses.backward()

                    if (i + 1) % self.accumulation_size == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRAD_CLIP)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                elif not self.gradient_accumulation and self.mixed_precision:
                    scaler.scale(losses).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                elif not self.gradient_accumulation and not self.mixed_precision:
                    self.optimizer.zero_grad()
                    losses.backward()
                    self.optimizer.step()

                self.train_step = self.save_losses_accuracy('train', losses, 0, self.train_step)
                self.total_step = self.save_memory(self.total_step)

                # Update progress bar
                pbar.set_description_str(f'Training Epoch {epoch}')
                pbar.set_postfix_str(f'Loss: {losses:.5f}')
                pbar.update()

            # We close the tqdm bar
            pbar.close()

            # Validate the model
            self.validate_model(epoch)
            #self.model.train()



        # If file_name is specified, save the trained model
        if self.file_name is not None:
            torch.save(self.model.state_dict(), f'models/{self.file_name}')

        #     if validation:
        #         if (epoch + 1) % validation == 0:
        #             torch.save(model.state_dict(), "models/" + filename)
        #             val_acc = validate_model(validation_dataset, filename)
        #             if early:
        #                 if es.step(torch.as_tensor(val_acc, dtype=torch.float16)):
        #                     print("Early Stopping")
        #                     break
        #     elif early:
        #         if es.step(losses):
        #             print("Early Stopping")
        #             break
        #
        # torch.save(model.state_dict(), "models/" + filename)

        print('Training done!')

        # # Main training loop, loop through each epoch
        # for epoch in range(epochs):
        #     # Declare empty loss and accuracy lists for the current epoch
        #     loss_list_epoch = []
        #     accuracy_list_epoch = []
        #
        #     # Loop through each mini-batch from the data loader
        #     for i, (images, targets) in enumerate(self.data_loader_train):
        #         # Send images and targets mem_total the device
        #         images, targets = images.mem_total(self.device), targets.mem_total(self.device)
        #
        #         # Reset all gradients mem_total zero
        #         self.optimizer.zero_grad()
        #
        #         # Perform mem_allocated forward pass
        #         outputs = self.model.forward(images)
        #
        #         # Calculate the loss, comparing outputs with the ground truth targets
        #         loss = self.criterion(outputs, targets)
        #
        #         # Appending the current loss mem_total the loss list and the current accuracy mem_total the accuracy list
        #         loss_list_epoch.append(loss.item())
        #         accuracy_list_epoch.append(self.get_accuracy(outputs, targets))
        #
        #         # Perform mem_allocated backward pass (calculate gradient)
        #         loss.backward()
        #
        #         # Perform mem_allocated parameter update based on the current gradient
        #         self.optimizer.step()
        #
        #         # Update progress bar
        #         pbar.set_description_str(mem_free'Epoch {epoch}')
        #         pbar.set_postfix_str(mem_free'Training Loss: {loss:.5f}')
        #         pbar.update()
        #
        #     # Reset progress bar after epoch completion
        #     pbar.reset()
        #
        #     # Save the training loss and accuracy in the object
        #     self.results['Training Loss'].append(round(np.mean(loss_list_epoch), 5))
        #     self.results['Training Accuracy'].append(round(np.mean(accuracy_list_epoch), 5))
        #
        #     # Validate the model
        #     self.validate_model()
        #
        # # We close the tqdm bar
        # pbar.close()
        #
        # # If file_name is specified, save the trained model
        # if self.file_name is not None:
        #     torch.save(self.model.state_dict(), mem_free'models/{self.file_name}')

    def validate_model(self, epoch) -> None:
        """
        Method to validate the model saved in the self.model class attribute.

        :return: None
        """
        # Specify that the model will be evaluated
        # self.model.eval()

        # # Declare empty loss and accuracy lists
        # loss_list = []
        # accuracy_list = []
        # Declare tqdm progress bar

        pbar = tqdm(total=len(self.data_loader_valid), leave=False,
                    desc='Validation Epoch 0', postfix='Loss: 0')

        # Deactivate the autograd engine
        with torch.no_grad():
            for i, (images, targets) in enumerate(self.data_loader_valid):
                images = torch.stack(images).to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # with amp.autocast(enabled=self.mixed_precision):
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                self.valid_step = self.save_losses_accuracy('validation', losses, 0, self.valid_step)
                self.total_step = self.save_memory(self.total_step)

                # Update progress bar
                pbar.set_description_str(f'Validation Epoch {epoch}')
                pbar.set_postfix_str(f'Loss: {losses:.5f}')
                pbar.update()

        pbar.close()

        # # Send images and labels to the device
        # images, labels = images.to(self.device), labels.to(self.device)
        #
        # # Perform a forward pass
        # outputs = self.model.forward(images)
        #
        # # Calculate the loss, comparing outputs with the ground truth labels
        # loss = self.criterion(outputs, labels)

        # # Appending the current loss to the loss list and current accuracy to the accuracy list
        # loss_list.append(loss.item())
        # accuracy_list.append(self.get_accuracy(outputs, labels))

        # # Calculate mean loss and mean accuracy over all batches
        # mean_loss = np.mean(loss_list)
        # mean_accuracy = np.mean(accuracy_list)
        #
        # # Save mean loss and mean accuracy in the object
        # self.results['Validation Loss'].append(round(mean_loss, 5))
        # self.results['Validation Accuracy'].append(round(mean_accuracy, 5))

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
                   progress: bool = True,
                   pretrained_backbone: bool = True,
                   trainable_backbone_layers: Optional[int] = None):
        """
        pretrained (bool) – If True, returns a model pre-trained on COCO train2017
        progress (bool) – If True, displays a progress bar of the download to stderr
        num_classes (int) – number of output classes of the model (including the background)
        pretrained_backbone (bool) – If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int) – number of trainable (not frozen) resnet layers starting from final block.
        Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.

        :param name:
        :param num_classes:
        :param pretrained:
        :param progress:
        :param pretrained_backbone:
        :param trainable_backbone_layers:
        :return:
        """
        # model = detection.fasterrcnn_resnet50_fpn(pretrained=False)
        # # get number of input features for the classifier
        # in_features = model.roi_heads.box_predictor.cls_score.in_features
        # # replace the pre-trained head with a new one
        # model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, self.num_classes)
        #
        # self.model = model
        if self.model_name == 'fasterrcnn_resnet50_fpn':
            self.model = detection.fasterrcnn_resnet50_fpn(pretrained=self.pretrained,
                                                           progress=progress,
                                                           num_classes=self.num_classes,
                                                           pretrained_backbone=pretrained_backbone,
                                                           trainable_backbone_layers=
                                                           trainable_backbone_layers)
        elif self.model_name == 'fasterrcnn_mobilenet_v3_large_fpn':
            self.model = detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=self.pretrained,
                                                                     progress=progress,
                                                                     num_classes=self.num_classes,
                                                                     pretrained_backbone=pretrained_backbone,
                                                                     trainable_backbone_layers=
                                                                     trainable_backbone_layers)
        elif self.model_name == 'fasterrcnn_mobilenet_v3_large_320_fpn':
            self.model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=self.pretrained,
                                                                         progress=progress,
                                                                         num_classes=self.num_classes,
                                                                         pretrained_backbone=pretrained_backbone,
                                                                         trainable_backbone_layers=
                                                                         trainable_backbone_layers)
        elif self.model_name == 'retinanet_resnet50_fpn':
            self.model = detection.retinanet_resnet50_fpn(pretrained=self.pretrained,
                                                          progress=progress,
                                                          num_classes=self.num_classes,
                                                          pretrained_backbone=pretrained_backbone,
                                                          trainable_backbone_layers=
                                                          trainable_backbone_layers)
        elif self.model_name == 'maskrcnn_resnet50_fpn':
            self.model = detection.maskrcnn_resnet50_fpn(pretrained=self.pretrained,
                                                         progress=progress,
                                                         num_classes=self.num_classes,
                                                         pretrained_backbone=pretrained_backbone,
                                                         trainable_backbone_layers=
                                                         trainable_backbone_layers)
        elif self.model_name == 'keypointrcnn_resnet50_fpn':
            self.model = detection.keypointrcnn_resnet50_fpn(pretrained=self.pretrained,
                                                             progress=progress,
                                                             num_classes=self.num_classes,
                                                             pretrained_backbone=pretrained_backbone,
                                                             trainable_backbone_layers=
                                                             trainable_backbone_layers)
        elif self.model_name == 'detr':
            """
            To Do: Implement DETR

            Paper:              https://arxiv.org/abs/2005.12872
            Official Repo:      https://github.com/facebookresearch/detr
            Unofficial Repo:    https://github.com/clive819/Modified-DETR
            """
            raise NotImplementedError

        elif self.model_name == 'perceiver':
            """
            To Do : Implement Perceiver

            Paper:              https://arxiv.org/abs/2103.03206
            Unofficial Repo:    https://github.com/lucidrains/perceiver-pytorch
            Unofficial Repo:    https://github.com/louislva/deepmind-perceiver
            """
            raise NotImplementedError
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

    def save_losses_accuracy(self, bucket, losses, accuracy, step):
        self.writer.add_scalar(f'Loss per iteration/{bucket}', losses, step)
        self.writer.add_scalar(f'Accuracy per iteration/{bucket}', accuracy, step)

        return step + 1

    def save_memory(self, step, scale=1e-9):
        mem_reserved = memory_reserved(0) * scale
        mem_allocated = memory_allocated(0) * scale
        mem_free = mem_reserved - mem_allocated

        self.writer.add_scalar('Memory/reserved', mem_reserved, step)
        self.writer.add_scalar('Memory/allocated', mem_allocated, step)
        self.writer.add_scalar('Memory/free', mem_free, step)

        return step + 1


