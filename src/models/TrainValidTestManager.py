from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.cuda.amp as amp
import torchvision.models.detection as detection
from torch.cuda import memory_reserved, memory_allocated
from src.data.SummaryWriter import SummaryWriter
from torchvision.ops import box_iou, nms
from tqdm import tqdm

from constants import CLASS_DICT
from src.data.DataLoaderManager import DataLoaderManager
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
                 mixed_precision,
                 gradient_accumulation,
                 pretrained: bool,
                 iou_threshold: float,
                 gradient_clip: float,
                 args_dic) -> None:
        self.args_dic = args_dic

        self.train_step = 0
        self.valid_acc_step = 0
        self.valid_loss_step = 0
        self.total_step = 0

        self.gradient_clip = gradient_clip

        self.iou_threshold = iou_threshold

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
        print(f'=== Dataset & Data Loader Sizes ===\n\n'
              f'Training:\t\t{len(self.data_loader_train.dataset)} images\t\t{len(self.data_loader_train)} batches\n'
              f'Validation:\t\t{len(self.data_loader_valid.dataset)} images\t\t{len(self.data_loader_valid)} batches\n'
              f'Testing:\t\t{len(self.data_loader_test.dataset)} images\t\t{len(self.data_loader_test)} batches\n')
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
        self.optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

        # Send the model to the device
        self.model.to(self.device)

        # Results dictionary
        # self.results = {
        #     'Training Loss': [],
        #     'Training Accuracy': [],
        #     'Validation Loss': [],
        #     'Validation Accuracy': []
        # }

    def __call__(self, epochs):
        self.train_model(epochs)
        # self.test_model()
        self.writer.add_hparams(self.args_dic, metric_dict={'metric/loss': 1})
        self.writer.flush()
        self.writer.close()

    def train_model(self, epochs: int) -> None:
        """
        Trains the model and saves the trained model.

        :param epochs: int, number of epochs
        """

        if self.early_stopping:
            es = EarlyStopping(patience=self.early_stopping)

        scaler = amp.grad_scaler.GradScaler(enabled=self.mixed_precision)

        for epoch in range(1, epochs + 1):
            # Declare tqdm progress bar
            pbar = tqdm(total=len(self.data_loader_train), leave=False,
                        desc=f'Training Epoch {epoch}', postfix='Loss: 0')

            # Specify that the model will be trained
            self.model.train()

            loss_list_epoch = []
            accuracy_list_epoch = []



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
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                elif self.gradient_accumulation and not self.mixed_precision:
                    losses.backward()

                    if (i + 1) % self.accumulation_size == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
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

                loss_list_epoch.append(losses.item())

                self.train_step = self.save_batch_loss('Training', losses, self.train_step)
                self.save_memory()

                # Update progress bar
                pbar.set_description_str(f'Training Epoch {epoch}')
                pbar.set_postfix_str(f'Loss: {losses:.5f}')
                pbar.update()

            self.save_epoch('Training', np.mean(loss_list_epoch), 0, epoch)

            # We close the tqdm bar
            pbar.close()
            # print('\n')

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

        # print('Training done!')

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
        pbar = tqdm(total=len(self.data_loader_valid), leave=False,
                    desc=f'Validation Loss Epoch {epoch}')

        loss_list_epoch = []

        # Deactivate the autograd engine
        with torch.no_grad():
            for i, (images, targets) in enumerate(self.data_loader_valid):
                images = torch.stack(images).to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # with amp.autocast(enabled=self.mixed_precision):
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                loss_list_epoch.append(losses.item())

                self.valid_loss_step = self.save_batch_loss('Validation', losses, self.valid_loss_step)
                self.save_memory()

                # Update progress bar
                pbar.set_postfix_str(f'Loss: {losses:.5f}')
                pbar.update()

        pbar.reset()
        pbar.set_postfix_str('')
        pbar.set_description_str(f'Validation Metrics Epoch {epoch}')

        self.model.eval()
        preds_list = []
        targets_list = []

        # Deactivate the autograd engine
        with torch.no_grad():
            for i, (images, targets) in enumerate(self.data_loader_valid):
                images = torch.stack(images).to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                preds = self.model(images)

                preds_list += preds
                targets_list += targets

                self.save_memory()
                pbar.update()

        preds_list = self.apply_nms(preds_list)
        metrics = self.calculate_metrics(preds_list, targets_list)

        self.save_epoch('Validation', np.mean(loss_list_epoch), metrics, epoch)

        pbar.close()

    def apply_nms(self, preds_list):
        keep_nms = [nms(pred['boxes'], pred['scores'], self.iou_threshold) for pred in preds_list]

        preds_nms = []

        for pred, keep in zip(preds_list, keep_nms):
            preds_nms.append({key: torch.index_select(val, dim=0, index=keep) for key, val in pred.items()})

        return preds_nms

    def calculate_metrics(self, preds_list, targets_list):
        targets_boxes = [target['boxes'] for target in targets_list]
        targets_labels = [target['labels'] for target in targets_list]
        preds_boxes = [pred['boxes'] for pred in preds_list]
        preds_labels = [pred['labels'] for pred in preds_list]

        iou_list = []

        for pred_boxes, target_boxes in zip(preds_boxes, targets_boxes):
            iou_list.append(box_iou(pred_boxes, target_boxes))

        max_iou_list = []
        types_list = []

        for iou, target_labels, pred_labels in zip(iou_list, targets_labels, preds_labels):
            if iou.nelement() > 0:
                max_iou_list.append(torch.max(iou, dim=1))

                type_list_iter = []

                for i, (value, index) in enumerate(zip(max_iou_list[-1].values, max_iou_list[-1].indices)):
                    if torch.greater(value, self.iou_threshold) and \
                       torch.equal(target_labels.data[index], pred_labels.data[i]):
                        type_list_iter.append(True)
                    else:
                        type_list_iter.append(False)

                types_list.append(type_list_iter)
            else:
                types_list.append([])

        recall_list = [sum(types)/len(targets_labels[i]) for i, types in enumerate(types_list)]
        precision_list = [0 if len(types) == 0 else sum(types)/len(types) for types in types_list]

        return 0

    def test_model(self) -> float:
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

        # Deactivate the autograd engine
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.data_loader_test):
                # Send images and labels to the device
                images, labels = images.to(self.device), labels.to(self.device)

                # Perform a forward pass
                outputs = self.model.forward(images)

                # Calculate the accuracy for the current batch
                accuracy_list.append(self.get_accuracy(outputs, labels))

        # Print mean test accuracy over all batches
        mean_accuracy = np.mean(accuracy_list)
        return mean_accuracy

    # @staticmethod
    # def iou(label, box1, box2, score, name, index, label_ocr="No OCR", verbose=False):
    #     name = ''.join(chr(i) for i in name)
    #     # print('{:^30}'.format(name[-1]))
    #     iou_list = []
    #     iou_label = []
    #     label_index = -1
    #     for item in box2:
    #         try:
    #             x11, y11, x21, y21 = item["boxes"]
    #             x12, y12, x22, y22 = box1
    #             xi1 = max(x11, x12)
    #             yi1 = max(y11, y12)
    #             xi2 = min(x21, x22)
    #             yi2 = min(y21, y22)
    #
    #             inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    #             # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    #             box1_area = (x21 - x11 + 1) * (y21 - y11 + 1)
    #             box2_area = (x22 - x12 + 1) * (y22 - y12 + 1)
    #             union_area = float(box1_area + box2_area - inter_area)
    #             # compute the IoU
    #             iou = inter_area / union_area
    #             iou_list.append(iou)
    #             label_index = iou_list.index(max(iou_list))
    #         except Exception as e:
    #             print("Error: ", e)
    #             print(iou_list)
    #             continue
    #     score1 = '%.2f' % score
    #     if verbose:
    #         try:
    #             print('{:^3} {:^20} {:^20} {:^25} {:^20} {:^10} {:^10}'.format(index,
    #                                                                            name, box2[label_index]["label"], label,
    #                                                                            label_index, score1,
    #                                                                            round(max(iou_list), 2)))
    #         except Exception:
    #             print('{:^3} {:^20} {:^20} {:^25} {:^20} {:^10} {:^10}'.format(index,
    #                                                                            name, "None", label, label_index, score1,
    #                                                                            "None"))
    #     if box2[label_index]["label"] == label:
    #         return float(score1)
    #     else:
    #         return 0

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
                                                           # num_classes=self.num_classes,
                                                           pretrained_backbone=pretrained_backbone,
                                                           trainable_backbone_layers=
                                                           trainable_backbone_layers)

        elif self.model_name == 'fasterrcnn_mobilenet_v3_large_fpn':
            self.model = detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=self.pretrained,
                                                                     progress=progress,
                                                                     # num_classes=self.num_classes,
                                                                     pretrained_backbone=pretrained_backbone,
                                                                     trainable_backbone_layers=
                                                                     trainable_backbone_layers)

        elif self.model_name == 'fasterrcnn_mobilenet_v3_large_320_fpn':
            self.model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=self.pretrained,
                                                                         progress=progress,
                                                                         # num_classes=self.num_classes,
                                                                         pretrained_backbone=pretrained_backbone,
                                                                         trainable_backbone_layers=
                                                                         trainable_backbone_layers)

        elif self.model_name == 'retinanet_resnet50_fpn':
            self.model = detection.retinanet_resnet50_fpn(pretrained=self.pretrained,
                                                          progress=progress,
                                                          # num_classes=self.num_classes,
                                                          pretrained_backbone=pretrained_backbone,
                                                          trainable_backbone_layers=
                                                          trainable_backbone_layers)

        elif self.model_name == 'maskrcnn_resnet50_fpn':
            self.model = detection.maskrcnn_resnet50_fpn(pretrained=self.pretrained,
                                                         progress=progress,
                                                         # num_classes=self.num_classes,
                                                         pretrained_backbone=pretrained_backbone,
                                                         trainable_backbone_layers=
                                                         trainable_backbone_layers)

        elif self.model_name == 'keypointrcnn_resnet50_fpn':
            self.model = detection.keypointrcnn_resnet50_fpn(pretrained=self.pretrained,
                                                             progress=progress,
                                                             # num_classes=self.num_classes,
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

        # if self.pretrained:
        self.replace_model_head()

    def replace_model_head(self):
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        if 'fasterrcnn' in self.model_name:
            self.model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_channels=in_features,
                                                                                         num_classes=self.num_classes)
        else:
            raise NotImplementedError

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

    # @staticmethod
    # def get_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    #     """
    #     Method to calculate accuracy of predicted outputs vs ground truth labels.
    #
    #     :param outputs: torch.Tensor, predicted outputs classes
    #     :param labels: torch.Tensor, ground truth labels classes
    #     :return: float, accuracy of the predicted outputs vs the ground truth labels
    #     """
    #     return (outputs.argmax(dim=1) == labels).sum().item() / labels.shape[0]

    def save_batch_loss(self, bucket, loss, step):
        self.writer.add_scalar(f'Loss (total per batch)/{bucket}', loss, step)

        return step + 1

    # def save_batch_acc(self, bucket, acc, step):
    #     self.writer.add_scalar(f'mAP (per batch)/{bucket}', acc, step)
    #
    #     return step + 1

    def save_epoch(self, bucket, loss, accuracy, epoch):
        self.writer.add_scalar(f'Loss (mean per epoch)/{bucket}', loss, epoch)
        self.writer.add_scalar(f'Accuracy (mean per epoch)/{bucket}', accuracy, epoch)

    def save_memory(self, scale=1e-9):
        mem_reserved = memory_reserved(0) * scale
        mem_allocated = memory_allocated(0) * scale
        mem_free = mem_reserved - mem_allocated

        self.writer.add_scalar('Memory/Reserved', mem_reserved, self.total_step)
        self.writer.add_scalar('Memory/Allocated', mem_allocated, self.total_step)
        self.writer.add_scalar('Memory/Free', mem_free, self.total_step)

        self.total_step += 1


