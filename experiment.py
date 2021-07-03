"""
File:
    src/models/experiment.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Main experiment file to perform a training, validation and testing pipeline on a new model.
"""

import argparse
from datetime import datetime
from multiprocessing import cpu_count

from src.data.DataLoaderManager import DataLoaderManager
from src.data.DatasetManager import DatasetManager
from src.models.PipelineManager import PipelineManager
from src.utils.helper_functions import print_dict
from src.utils.reproducibility import set_deterministic
from src.utils.constants import RESIZED_PATH, TARGETS_PATH

if __name__ == '__main__':
    # Record start time
    start = datetime.now()

    # Declare argument parser
    parser = argparse.ArgumentParser(description='Processing inputs')

    # Number of workers argument
    parser.add_argument('-nw', '--num_workers', action='store', type=int, default=cpu_count(),
                        help='Number of workers')

    # Number of epochs argument
    parser.add_argument('-e', '--epochs', action='store', type=int, default=200,
                        help='Number of epochs')

    # Batch size argument
    parser.add_argument('-b', '--batch', action='store', type=int, default=1,
                        help='Batch size')

    # Validation size argument
    parser.add_argument('-vs', '--validation_size', action='store', type=float, default=0.1,
                        help='Size of validation set (float as proportion of dataset)')

    # Testing size argument
    parser.add_argument('-ts', '--test_size', action='store', type=float, default=0.1,
                        help='Size of test set (float as proportion of dataset)')

    # Resizing argument
    parser.add_argument('-s', '--image_size', action='store', type=int, default=1024,
                        help='Resize the images to size*size')

    # Data augmentation argument
    parser.add_argument('-da', '--data_aug', action='store', type=float, default=0.25,
                        help='Value of data augmentation for training dataset (0: no aug)')

    # Compute mean & std deviation on training set argument
    parser.add_argument('-norm', '--normalize', action='store', type=str,
                        choices=['precalculated',
                                 'calculated',
                                 'disabled'],
                        default='precalculated',
                        help='Normalize the training dataset by mean & std using '
                             'precalculated values, calculated values or disabled')

    # Learning rate argument
    parser.add_argument('-lr', '--learning_rate', action='store', type=float, default=0.0003,
                        help='Learning rate for optimizer')

    # Weight decay argument
    parser.add_argument('-wd', '--weight_decay', action='store', type=float, default=0.0001,
                        help='Weight decay (L2 penalty) for optimizer')

    # Model argument
    parser.add_argument('-mo', '--model', action='store', type=str,
                        choices=['fasterrcnn_resnet50_fpn',
                                 'fasterrcnn_mobilenet_v3_large_fpn',
                                 'fasterrcnn_mobilenet_v3_large_320_fpn',
                                 'retinanet_resnet50_fpn',
                                 'detr'],
                        default='fasterrcnn_resnet50_fpn',
                        help='Specify which object detection model to use')

    # Early stopping patience argument
    parser.add_argument('-esp', '--es_patience', action='store', type=int, default=0,
                        help='Early stopping patience')

    # Early stopping delta argument
    parser.add_argument('-esd', '--es_delta', action='store', type=float, default=0,
                        help='Early stopping delta')

    # Mixed precision argument
    parser.add_argument('-mp', '--mixed_precision', action='store_true',
                        help='To use mixed precision')

    # Gradient accumulation size argument
    parser.add_argument('-g', '--gradient_accumulation', action='store', type=int, default=1,
                        help='Gradient accumulation size')

    # Gradient clipping argument
    parser.add_argument('-gc', '--gradient_clip', action='store', type=float, default=5,
                        help='Gradient clipping value')

    # Random seed argument
    parser.add_argument('-rs', '--random_seed', action='store', type=int, default=54288,
                        help='Set random seed')

    # Deterministic argument
    parser.add_argument('-dt', '--deterministic', action='store_true',
                        help='Set deterministic behaviour')
    # Pretrained argument
    parser.add_argument('-no-pt', '--no_pretrained', action='store_true',
                        help='If specified, the loaded model will not be pretrained')

    # Save model argument
    parser.add_argument('-no-sv', '--no_save_model', action='store_true',
                        help='If specified, the trained model will not be saved')

    # Google Images argument
    parser.add_argument('-no-gi', '--no_google_images', action='store_true',
                        help='If specified, the Google Images photos will be excluded from the training subset')

    # Calculate training set metrics
    parser.add_argument('-ltm', '--log_training_metrics', action='store_true',
                        help='If specified, the AP and AR metrics will be calculated and logged for training set')

    # Calculate training set metrics
    parser.add_argument('-lm', '--log_memory', action='store_true',
                        help='If specified, the memory will be logged')

    # Best or last model saved/used for test inference argument
    parser.add_argument('-sl', '--save_last', action='store_true',
                        help='Specify whether to save/use for inference testing the last model, otherwise'
                             'the best model will be used')

    # Parsing arguments
    args = parser.parse_args()

    # Addtional arguments for DE:TR model
    if args.model == 'detr':
        # DE:TR loss coefficients
        parser.add_argument('--set_cost_class', default=1, type=float,
                            help="Class coefficient in the matching cost")
        parser.add_argument('--set_cost_bbox', default=5, type=float,
                            help="L1 box coefficient in the matching cost")
        parser.add_argument('--set_cost_giou', default=2, type=float,
                            help="giou box coefficient in the matching cost")
        parser.add_argument('--eos_coef', default=0.1, type=float,
                            help="Relative classification weight of the no-object class")
        # Parsing arguments
        args = parser.parse_args()

    # Set deterministic behavior
    set_deterministic(args.deterministic, args.random_seed)

    # Declare file name as yyyy-mm-dd_hh-mm-ss
    file_name = f'{args.model.split("_")[0]}_{args.epochs}_{start.strftime("%Y-%m-%d_%H-%M-%S")}'

    # Display arguments in console
    print('\n=== Arguments & Hyperparameters ===\n')
    print_dict(vars(args), 6)

    # Declare dataset manager
    dataset_manager = DatasetManager(images_path=RESIZED_PATH,
                                     targets_path=TARGETS_PATH,
                                     image_size=args.image_size,
                                     num_workers=args.num_workers,
                                     data_aug=args.data_aug,
                                     validation_size=args.validation_size,
                                     test_size=args.test_size,
                                     norm=args.normalize,
                                     google_images=not args.no_google_images,
                                     seed=args.random_seed)

    # Declare data loader manager
    data_loader_manager = DataLoaderManager(dataset_manager=dataset_manager,
                                            batch_size=args.batch,
                                            gradient_accumulation=args.gradient_accumulation,
                                            num_workers=args.num_workers,
                                            deterministic=args.deterministic)

    # Declare training, validation and testing manager
    train_valid_test_manager = PipelineManager(data_loader_manager=data_loader_manager,
                                               file_name=file_name,
                                               model_name=args.model,
                                               learning_rate=args.learning_rate,
                                               weight_decay=args.weight_decay,
                                               es_patience=args.es_patience,
                                               es_delta=args.es_delta,
                                               mixed_precision=args.mixed_precision,
                                               gradient_accumulation=args.gradient_accumulation,
                                               pretrained=not args.no_pretrained,
                                               gradient_clip=args.gradient_clip,
                                               args_dict=vars(args),
                                               save_model=not args.no_save_model,
                                               image_size=args.image_size,
                                               save_last=args.save_last,
                                               log_training_metrics=args.log_training_metrics,
                                               log_memory=args.log_memory,
                                               class_loss_ceof=args.set_cost_class if args.model == 'detr' else None,
                                               bbox_loss_coef=args.set_cost_bbox if args.model == 'detr' else None,
                                               giou_loss_coef=args.set_cost_giou if args.model == 'detr' else None,
                                               eos_coef=args.eos_coef if args.model == 'detr' else None)

    # Call the training, validation and testing manager to run the pipeline
    train_valid_test_manager(args.epochs)

    # Print the run time of the current experiment
    print(f'\nTotal time for current experiment:\t{str(datetime.now() - start).split(".")[0]}')
