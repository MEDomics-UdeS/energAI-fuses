"""
File:
    src/models/PipelineManager.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Training, validation and testing pipeline manager
"""

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda import memory_reserved, memory_allocated
from src.data.DataLoaderManagers.CocoDataLoaderManager import CocoDataLoaderManager
from src.data.DatasetManagers.CocoDatasetManager import CocoDatasetManager
from src.detr.criterion import build_criterion
from src.models.SummaryWriter import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Optional
from copy import deepcopy

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.constants import CLASS_DICT, LOG_PATH, MODELS_PATH, EVAL_METRIC, COCO_PARAMS_LIST
from src.data.DataLoaderManagers.LearningDataLoaderManager import LearningDataLoaderManager
from src.models.EarlyStopper import EarlyStopper
from src.models.models import load_model
from src.coco.coco_utils import get_coco_api_from_dataset
from src.coco.coco_eval import CocoEvaluator
from src.utils.helper_functions import print_dict, format_detr_outputs
from src.detr.box_ops import batch_box_xyxy_to_cxcywh


class PipelineManager:
    """
    Training, validation and testing manager.
    """

    def __init__(self, data_loader_manager: LearningDataLoaderManager,
                 file_name: str,
                 model_name: str,
                 learning_rate: float,
                 weight_decay: float,
                 es_patience: int,
                 es_delta: float,
                 mixed_precision: bool,
                 gradient_accumulation: int,
                 pretrained: bool,
                 gradient_clip: float,
                 args_dict: dict,
                 save_model: bool,
                 image_size: int,
                 save_last: bool,
                 log_training_metrics: bool,
                 log_memory: bool,
                 class_loss_ceof: float,
                 bbox_loss_coef: float, 
                 giou_loss_coef: float, 
                 eos_coef: float) -> None:
        """
        Class constructor

        :param data_loader_manager: DataLoaderManager, contains the training, validation and testing data loaders
        :param file_name: str, file name to save tensorboard runs and model
        :param model_name: str, model name
        :param learning_rate: float, learning rate for the Adam optimizer
        :param weight_decay: float, weight decay (L2 penalty) for the Adam optimizer
        :param es_patience: int, early stopping patience (number of epochs of no improvement)
        :param es_delta: float, early stopping delta (to evaluate improvement)
        :param mixed_precision: bool, to use mixed precision in the training
        :param gradient_accumulation: int, gradient accumulation size
        :param pretrained: bool, to use a pretrained model
        :param iou_threshold: float, iou threshold for non-maximum suppression and score filtering the predicted boxes
        :param gradient_clip: float, value at which to clip the gradient when using gradient accumulation
        :param args_dict: dict, dictionary of all parameters to log the hyperparameters in tensorboard
        :param save_model: bool, to save the trained model in the saved_models/ directory
        """
        # Save arguments as object attributes
        self.__file_name = file_name
        self.__save_model = save_model
        self.__args_dict = args_dict
        self.__gradient_clip = gradient_clip
        self.__pretrained = pretrained
        self.__model_name = model_name
        self.__mixed_precision = mixed_precision
        self.__accumulation_size = gradient_accumulation
        self.__gradient_accumulation = False if gradient_accumulation == 1 else True
        self.__es_patience = es_patience
        self.__image_size = image_size
        self.__save_last = save_last
        self.__log_training_metrics = log_training_metrics
        self.__log_memory = log_memory
        
        # Declare steps for tensorboard logging
        self.__train_step = 0
        self.__valid_step = 0

        if self.__log_memory:
            self.__total_step = 0

        # Declare tensorboard writer
        self.__writer = SummaryWriter(LOG_PATH + file_name)

        # Get number of classes
        self.__num_classes = len(CLASS_DICT)

        # If early stopping patience is specified, declare an early stopper
        if self.__es_patience is not None:
            self.__early_stopper = EarlyStopper(patience=es_patience, min_delta=es_delta)

        # Declare gradient scaler for mixed precision
        self.__scaler = GradScaler(enabled=self.__mixed_precision)

        # Extract the training, validation and testing data loaders
        self.__data_loader_train = data_loader_manager.data_loader_train
        self.__data_loader_valid = data_loader_manager.data_loader_valid
        self.__data_loader_test = data_loader_manager.data_loader_test

        # Display the datasets and data loaders sizes
        print(f'\n=== Dataset & Data Loader Sizes ===\n\n'
              f'Training:\t\t{len(self.__data_loader_train.dataset)} images\t\t'
              f'{len(self.__data_loader_train)} batches\n'
              f'Validation:\t\t{len(self.__data_loader_valid.dataset)} images\t\t'
              f'{len(self.__data_loader_valid)} batches\n'
              f'Testing:\t\t{len(self.__data_loader_test.dataset)} images\t\t'
              f'{len(self.__data_loader_test)} batches\n')

        # Get model and set last fully-connected layer with the right number of classes
        self.__model = load_model(self.__model_name, self.__pretrained, self.__num_classes)

        # Define device as the GPU if available, else use the CPU
        self.__device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Send the model to the device
        self.__model.to(self.__device)
        
        if self.__model_name == 'detr':
            self.__criterion = build_criterion(class_loss_ceof, bbox_loss_coef, giou_loss_coef, eos_coef,
                                               self.__num_classes)
            self.__criterion.to(self.__device)

            params = [
                {"params": [p for n, p in self.__model.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in self.__model.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": learning_rate,
                },
            ]
        else:
            # Find which parameters to train (those with .requires_grad = True)
            params = [p for p in self.__model.parameters() if p.requires_grad]

        # Define Adam optimizer
        self.__optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

        self.__swa_model = AveragedModel(self.__model, device=self.__device)

        self.__swa_started = False

        if not self.__save_last:
            self.__best_model = None
            self.__best_epoch = 0
            self.__best_score = 0
            self.__best_metrics_dict = None
        else:
            self.__last_metrics_dict = None

    def __call__(self, epochs: int) -> None:
        """
        Class __call__ method, called when object() is called

        :param epochs: int, number of epochs
        """

        self.__swa_start = int(0.75 * epochs)
        self.__scheduler = CosineAnnealingLR(self.__optimizer, T_max=epochs)

        # Train the model for a specified number of epochs
        self.__train_model(epochs)

        if not self.__save_last:
            print(f'\nBest epoch:\t\t\t\t\t\t\t{self.__best_epoch}/{epochs}')
            print(f'Best score:\t\t\t\t\t\t\t{EVAL_METRIC}: {self.__best_score:.2%}')

        # Check if we need to save the model
        if self.__save_model:
            # Save the model in the saved_models/ folder
            filename = f'{MODELS_PATH}{self.__file_name}'
            
            if self.__best_epoch >= self.__swa_start:
                ranking_model = self.__swa_model.module if self.__save_last else self.__best_model.module
            else:
                ranking_model = self.__model if self.__save_last else self.__best_model
            
            # Storing the model and meta data in the save state
            save_state = {
                "model": ranking_model.state_dict(),
                "args_dict": self.__args_dict,
                "ranked_imgs": self.__rank_images(ranking_model)
            }
            torch.save(save_state, filename)

            print(f'{"Last" if self.__save_last else "Best"} model saved to:\t\t\t\t{filename}\n')

        # Save best or last epoch validation metrics dict to tensorboard
        metrics_dict = self.__last_metrics_dict if self.__save_last else self.__best_metrics_dict

        # Append 'hparams/' to the start of each metrics dictionary key to log in tensorboard
        for key in metrics_dict.fromkeys(metrics_dict):
            metrics_dict[f'hparams/{key}'] = metrics_dict.pop(key)

        # Save the hyperparameters with tensorboard
        self.__writer.add_hparams(self.__args_dict, metric_dict=metrics_dict)

        # Test the trained model
        self.__test_model()

        # Flush and close the tensorboard writer
        self.__writer.flush()
        self.__writer.close()

    def __train_model(self, epochs: int) -> None:
        """
        Train the model

        :param epochs: int, number of epochs
        """
        # Loop through each epoch
        for epoch in range(1, epochs + 1):
            if epoch >= self.__swa_start:
                if not self.__swa_started:
                    self.__swa_scheduler = SWALR(self.__optimizer, swa_lr=self.__scheduler.get_last_lr()[0])
                    self.__swa_started = True

            # Train the model and get the loss
            loss = self.__evaluate(self.__model, self.__data_loader_train, 'Training', epoch)

            if self.__swa_started:
                self.__swa_model.update_parameters(self.__model)
                self.__swa_scheduler.step()
                torch.optim.swa_utils.update_bn(self.__data_loader_train, self.__swa_model)
            else:
                self.__scheduler.step()

            model = self.__swa_model if self.__swa_started else self.__model

            if self.__log_training_metrics:
                metrics_dict = self.__coco_evaluate(model, deepcopy(self.__data_loader_train),
                                                    f'Training Metrics Epoch {epoch}')
            else:
                metrics_dict = None

            # Save the current epoch loss for tensorboard
            self.__save_epoch('Training', loss, metrics_dict, epoch)

            metric = self.__validate_model(model, epoch)

            # Check if early stopping is enabled
            if self.__es_patience:
                # Check if the early stopping criterion has been reached
                if self.__early_stopper.step(torch.as_tensor(metric, dtype=torch.float16)):
                    # Early stop
                    print(f'Early stopping criterion has been reached after {self.__es_patience} epochs\n')
                    break

    def __evaluate(self, model, data_loader: DataLoader, phase: str, epoch: int) -> float:
        """
        To perform forward passes, compute the losses and perform backward passes on the model

        :param data_loader: DataLoader, data loader object
        :param phase: str, current phase, either 'Training' or 'Validation'
        :param epoch: int, current epoch
        :return: float, mean loss for the current epoch
        """
        # Declare tqdm progress bar
        pbar = tqdm(total=len(data_loader), leave=False, desc=f'{phase} Epoch {epoch}')

        # Specify that the model will be trained
        model.train()

        if self.__model_name == 'detr':
            self.__criterion.train()

        # Declare empty list to save losses
        loss_list_epoch = []

        # Reset the gradient if gradient accumulation is used
        if self.__gradient_accumulation:
            self.__optimizer.zero_grad()

        # Loop through each batch in the data loader
        for i, (images, targets) in enumerate(data_loader):
            if self.__model_name == 'detr':
                batch_box_xyxy_to_cxcywh(targets, self.__image_size)
            
            # Send images and targets to the device
            images = torch.stack(images).to(self.__device)
            targets = [{k: v.to(self.__device) for k, v in t.items()} for t in targets]

            # Get losses for the current batch
            with autocast(enabled=self.__mixed_precision):
                if self.__model_name == 'detr':
                    loss_dict = model(images)
                    loss_dict = self.__criterion(loss_dict, targets)

                    # Calculating the detr losses
                    weight_dict = self.__criterion.weight_dict
                    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                else:
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

            # Append current batch loss to the loss list
            loss_list_epoch.append(losses.item())

            # Check if we are in the 'Training' phase to perform a backward pass
            if phase == 'Training':
                self.__update_model(losses, i)

            self.__save_batch(phase, losses)

            # Save memory usage to tensorboard
            if self.__log_memory:
                self.__save_memory()

            # Update progress bar
            pbar.set_postfix_str(f'Loss: {losses:.5f}')
            pbar.update()

        # Close progress bar
        pbar.close()

        # Return the mean loss for the current epoch
        return float(np.mean(loss_list_epoch))

    def __update_model(self, losses: torch.Tensor, i: int) -> None:
        """

        :param losses:
        :param i:
        :return:
        """
        # Backward pass for no gradient accumulation + no mixed precision
        if not self.__gradient_accumulation and not self.__mixed_precision:
            self.__optimizer.zero_grad()
            losses.backward()
            self.__optimizer.step()

        # Backward pass for no gradient accumulation + mixed precision
        elif not self.__gradient_accumulation and self.__mixed_precision:
            self.__scaler.scale(losses).backward()
            self.__scaler.step(self.__optimizer)
            self.__scaler.update()
            self.__optimizer.zero_grad(set_to_none=True)

        # Backward pass for gradient accumulation + no mixed precision
        elif self.__gradient_accumulation and not self.__mixed_precision:
            losses.backward()

            if (i + 1) % self.__accumulation_size == 0:
                clip_grad_norm_(self.__model.parameters(), max_norm=self.__gradient_clip)
                self.__optimizer.step()
                self.__optimizer.zero_grad()

        # Backward pass for gradient accumulation + mixed precision
        elif self.__gradient_accumulation and self.__mixed_precision:
            self.__scaler.scale(losses).backward()

            if (i + 1) % self.__accumulation_size == 0:
                self.__scaler.unscale_(self.__optimizer)
                clip_grad_norm_(self.__model.parameters(), max_norm=self.__gradient_clip)
                self.__scaler.step(self.__optimizer)
                self.__scaler.update()
                self.__optimizer.zero_grad(set_to_none=True)

    def __validate_model(self, model, epoch: int) -> float:
        """
        Validate the model for the current epoch

        :param epoch: int, current epoch
        :return: float, mean recall per image metric
        """
        # Deactivate the autograd engine
        with torch.no_grad():
            # Evaluate the loss on the validation set
            loss = self.__evaluate(model, self.__data_loader_valid, 'Validation Loss', epoch)

        # Evaluate the object detection metrics on the validation set
        metrics_dict = self.__coco_evaluate(model, deepcopy(self.__data_loader_valid),
                                            f'Validation Metrics Epoch {epoch}')

        if not self.__save_last:
            if metrics_dict[EVAL_METRIC] > self.__best_score:
                self.__best_model = model
                self.__best_score = metrics_dict[EVAL_METRIC]
                self.__best_epoch = epoch
                self.__best_metrics_dict = metrics_dict
        else:
            self.__last_metrics_dict = metrics_dict

        # Save the validation results for the current epoch in tensorboard
        self.__save_epoch('Validation', loss, metrics_dict, epoch)

        # Return the evaluation metric
        return metrics_dict[EVAL_METRIC]

    def __test_model(self) -> None:
        """
        Test the trained model
        """
        model = self.__swa_model if self.__save_last else self.__best_model

        # Update bn statistics for the swa_model at the end
        torch.optim.swa_utils.update_bn(self.__data_loader_train, model)

        # COCO Evaluation
        metrics_dict = self.__coco_evaluate(model, self.__data_loader_test, 'Testing Metrics')

        # Print the testing object detection metrics results
        print('=== Testing Results ===\n')
        print_dict(metrics_dict, 6, '.2%')

    @torch.no_grad()
    def __coco_evaluate(self, model, data_loader: DataLoader, desc: str) -> dict:
        """

        :param model:
        :param data_loader:
        :return:
        """
        pbar = tqdm(total=len(data_loader), leave=False, desc=desc)

        coco = get_coco_api_from_dataset(data_loader.dataset)
        coco_evaluator = CocoEvaluator(coco, ['bbox'])

        model.eval()

        for images, targets in data_loader:
            images = list(img.to(self.__device) for img in images)

            outputs = model(images)

            if self.__model_name == 'detr':
                target_sizes = torch.stack(
                    [torch.tensor([self.__image_size, self.__image_size]) for _ in targets], dim=0)
                outputs = format_detr_outputs(outputs, target_sizes, self.__device)

            outputs = [{k: v.to(self.__device) for k, v in t.items()} for t in outputs]

            results = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(results)

            pbar.update()

        pbar.close()

        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        return dict(zip(COCO_PARAMS_LIST, coco_evaluator.coco_eval['bbox'].stats.tolist()))

    def __save_batch(self, phase: str, loss: float) -> None:
        """
        Save batch losses to tensorboard

        :param phase: str, phase, either 'Training' or 'Validation'
        :param loss: float, total loss per batch
        """
        if phase == 'Training':
            self.__writer.add_scalar(f'Loss/{phase} (total per batch)', loss, self.__train_step)
            self.__train_step += 1
        elif phase == 'Validation':
            self.__writer.add_scalar(f'Loss/{phase} (total per batch)', loss, self.__valid_step)
            self.__valid_step += 1

    def __save_epoch(self, phase: str, loss: float, metrics_dict: Optional[dict], epoch: int) -> None:
        """
        Save epoch results to tensorboard

        :param phase: str, either 'Training' or 'Validation'
        :param loss: float, mean loss per epoch
        :param metrics_dict: dict, contains the object detection evaluation metrics
        :param epoch: int, current epoch
        """
        self.__writer.add_scalar(f'Loss/{phase} (mean per epoch)', loss, epoch)

        if metrics_dict is not None:
            for i, (key, value) in enumerate(metrics_dict.items(), start=1):
                self.__writer.add_scalar(f'{key[:2]} ({phase})/{i}. {key[6:-1]}', value, epoch)

        if phase == 'Training':
            if self.__swa_started:
                self.__writer.add_scalar('Learning Rate', self.__swa_scheduler.get_last_lr()[0], epoch)
            else:
                self.__writer.add_scalar('Learning Rate', self.__scheduler.get_last_lr()[0], epoch)

    def __save_memory(self, scale: float = 1e-9) -> None:
        """
        Save current memory usage to tensorboard

        :param scale: float, scale to apply to the memory values (1e-9 : giga)
        """
        mem_reserved = memory_reserved(0) * scale
        mem_allocated = memory_allocated(0) * scale
        mem_free = mem_reserved - mem_allocated

        self.__writer.add_scalar('Memory/Reserved', mem_reserved, self.__total_step)
        self.__writer.add_scalar('Memory/Allocated', mem_allocated, self.__total_step)
        self.__writer.add_scalar('Memory/Free', mem_free, self.__total_step)

        self.__total_step += 1

    def __rank_images(self, model, metrics: str = 'loss') -> dict:
        
        performance_dict = {
            "training": [],
            "validation": [],
            "testing": []
        }
        if metrics == 'coco':
            # Maybe use deepcopy for coco ranking
            # self.__coco_ranking_pass(model, ds=self.__data_loader_train.dataset,
            #                     data_type="training", performance_dict=performance_dict, desc="Ranking training images by AP")
            self.__coco_ranking_pass(model, ds=self.__data_loader_valid.dataset,
                                data_type="validation", performance_dict=performance_dict, desc="Ranking validation images by AP")
            self.__coco_ranking_pass(model, ds=self.__data_loader_test.dataset,
                                data_type="testing", performance_dict=performance_dict, desc="Ranking test images by AP")
        elif metrics == 'loss':
            self.__loss_ranking_pass(model, data_loader=self.__data_loader_train,
                                     data_type="training", performance_dict=performance_dict, desc="Ranking training images by loss")
            self.__loss_ranking_pass(model, data_loader=self.__data_loader_valid,
                                     data_type="validation", performance_dict=performance_dict, desc="Ranking validation images by loss")
            self.__loss_ranking_pass(model, data_loader=self.__data_loader_test,
                                     data_type="testing", performance_dict=performance_dict, desc="Ranking test images by loss")
        return performance_dict

    @torch.no_grad()
    def __loss_ranking_pass(self, model, data_loader: DataLoader, data_type: str, performance_dict: dict, desc: str) -> None:
        # Declare tqdm progress bar
        pbar = tqdm(total=len(data_loader), leave=False, desc=desc)
        
        # Specify that the model will be trained
        model.train()
        if self.__model_name == 'detr':
            self.__criterion.train()

        # Loop through each batch in the data loader
        for (images, targets) in data_loader:
            if self.__model_name == 'detr':
                batch_box_xyxy_to_cxcywh(targets, self.__image_size)

            # Send images and targets to the device
            images = torch.stack(images).to(self.__device)
            targets = [{k: v.to(self.__device) for k, v in t.items()} for t in targets]

            # Loop through the batch to calculate losses individually
            for i in range(data_loader.batch_size):
                try:
                    # Reshaping image tensor to be of shape [1, 3, img_size, img_size]
                    image = images[i][...].reshape(1, images.shape[1], images.shape[2], images.shape[3])
                except IndexError:
                    break
                else:
                    # Selecting relevant target info
                    target = [targets[i]]

                    if self.__model_name == 'detr':
                        loss_dict = model(image)
                        loss_dict = self.__criterion(loss_dict, target)

                        # Calculate the losses
                        weight_dict = self.__criterion.weight_dict
                        loss_dict = {k: loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict}
                    else:
                        loss_dict = model(image, target)

                    # Adding image path and associated loss to results dict
                    performance_dict[data_type].append({
                        "img_path": data_loader.dataset.image_paths[i],
                        "metrics": loss_dict
                    })
                    
                    # Calculate the total loss for the image
                    performance_dict[data_type][-1]["metrics"]["total_loss"] = sum(loss for loss in loss_dict.values())

            # Updating the progress bar
            pbar.update()

        # Sorting the values in the dictionnary from highest to lowest
        performance_dict[data_type] = sorted(performance_dict[data_type], key=lambda x: x["metrics"]["total_loss"])

        # Closing the progress bar
        pbar.close()
    
    # FIXME do not use, function doesn't work yet  
    @torch.no_grad()
    def __coco_ranking_pass(self, model, ds, data_type: str, performance_dict: dict, desc: str) -> None:
        # Declare tqdm progress bar
        pbar = tqdm(total=len(ds), leave=False, desc=desc)

        # Create the coco dataset manager
        coco_ds = CocoDatasetManager(ds)
        coco_dl = CocoDataLoaderManager(coco_ds, 1, 1, 24, False)

        model.eval()

        for dl in coco_dl.data_loaders:
            # Preparing the coco evaluator class for ranking
            coco = get_coco_api_from_dataset(dl.dataset)
            coco_evaluator = CocoEvaluator(coco, ['bbox'])
            
            # Loop through each batch in the data loader
            for images, targets in dl:
                # Send images and targets to the device
                images = list(img.to(self.__device) for img in images)

                outputs = model(images)

                if self.__model_name == 'detr':
                    target_sizes = torch.stack([torch.tensor([self.__image_size, self.__image_size]) for _ in targets], dim=0)
                    outputs = format_detr_outputs(outputs, target_sizes, self.__device)

                outputs = [{k: v.to(self.__device)
                            for k, v in t.items()} for t in outputs]

                results = {targets[0]['image_id'].item(): outputs[0]}
                coco_evaluator.update(results)

                # This breaks the coco_evaluator after the first pass
                coco_evaluator.synchronize_between_processes()
                coco_evaluator.accumulate()
                coco_evaluator.summarize()

                performance_dict[data_type].append({
                    "img_path": dl.dataset.image_paths[0],
                    "metrics": dict(zip(COCO_PARAMS_LIST, coco_evaluator.coco_eval['bbox'].stats.tolist()))[EVAL_METRIC]
                })
                # Updating the progress bar
                pbar.update()

        # Sorting the values in the dictionnary from highest to lowest
        performance_dict[data_type] = sorted(performance_dict[data_type], key=lambda x: x["metrics"])

        # Closing the progress bar
        pbar.close()
