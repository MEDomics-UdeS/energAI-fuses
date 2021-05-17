"""
File:
    src/models/experiment.py

Authors:
    - Simon Giard-Leroux
    - Shreyas Sunil Kulkarni

Description:
    Main experiment file to perform a training, validation and testing pipeline on a new model.
"""

import argparse
from datetime import datetime
from multiprocessing import cpu_count

from src.data.DataLoaderManager import DataLoaderManager
from src.data.DatasetManager import DatasetManager
from src.models.TrainValidTestManager import TrainValidTestManager
from src.utils.helper_functions import print_args
from src.utils.reproducibility import set_deterministic
from src.utils.constants import RESIZED_PATH, TARGETS_PATH

if __name__ == '__main__':
    # Record start time
    start = datetime.now()

    # Get number of cpu threads for PyTorch DataLoader and ray paralleling
    num_workers = cpu_count()

    # Declare argument parser
    parser = argparse.ArgumentParser(description='Processing inputs')

    # Resizing argument
    parser.add_argument('-s', '--size', action='store', type=int, default=1024,
                        help='Resize the images to size*size (takes an argument: max_resize value (int))')

    # Data augmentation argument
    parser.add_argument('-da', '--data_aug', action='store', type=float, default=0.25,
                        help='Value of data augmentation for training dataset (0: no aug)')

    # Validation size argument
    parser.add_argument('-vs', '--validation_size', action='store', type=float, default=0.1,
                        help='Size of validation set (float as proportion of dataset)')

    # Testing size argument
    parser.add_argument('-ts', '--test_size', action='store', type=float, default=0.1,
                        help='Size of test set (float as proportion of dataset)')

    # Number of epochs argument
    parser.add_argument('-e', '--epochs', action='store', type=int, default=1,
                        help='Number of epochs')

    # Batch size argument
    parser.add_argument('-b', '--batch', action='store', type=int, default=20,
                        help='Batch size')

    # Early stopping patience argument
    parser.add_argument('-esp', '--es_patience', action='store', type=int,
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
    parser.add_argument('-rs', '--random_seed', action='store', type=int, default=42,
                        help='Set random seed')

    # Deterministic argument
    parser.add_argument('-dt', '--deterministic', action='store_true',
                        help='Set deterministic behaviour')

    # Compute mean & std deviation on training set argument
    parser.add_argument('-ms', '--mean_std', action='store_true',
                        help='Compute mean & standard deviation on training set if true, '
                             'otherwise use precalculated values')

    # IOU threshold argument
    parser.add_argument('-iou', '--iou_threshold', action='store', type=float, default=0.5,
                        help='IOU threshold for true/false positive box predictions')

    # Learning rate argument
    parser.add_argument('-lr', '--learning_rate', action='store', type=float, default=0.0003,
                        help='Learning rate for optimizer')

    # Weight decay argument
    parser.add_argument('-wd', '--weight_decay', action='store', type=float, default=0,
                        help='Weight decay (L2 penalty) for optimizer')

    # Model argument
    parser.add_argument('-mo', '--model', action='store', type=str,
                        choices=['fasterrcnn_resnet50_fpn',
                                 'fasterrcnn_mobilenet_v3_large_fpn',
                                 'fasterrcnn_mobilenet_v3_large_320_fpn',
                                 'retinanet_resnet50_fpn',
                                 'detr', 'perceiver'],
                        default='fasterrcnn_resnet50_fpn',
                        help='Specify which object detection model to use')

    # Pretrained argument
    parser.add_argument('-pt', '--pretrained', action='store_false',
                        help='Load pretrained model')

    # Save model argument
    parser.add_argument('-sv', '--save_model', action='store_false',
                        help='Save trained model')

    # Parsing arguments
    args = parser.parse_args()

    # Set deterministic behavior
    set_deterministic(args.deterministic, args.random_seed)

    # Declare file name as yyyy-mm-dd_hh-mm-ss
    file_name = start.strftime('%Y-%m-%d_%H-%M-%S')

    # Display arguments in console
    print_args(args)

    # Declare dataset manager
    dataset_manager = DatasetManager(images_path=RESIZED_PATH,
                                     targets_path=TARGETS_PATH,
                                     max_image_size=args.size,
                                     num_workers=num_workers,
                                     data_aug=args.data_aug,
                                     validation_size=args.validation_size,
                                     test_size=args.test_size,
                                     mean_std=args.mean_std)

    # Declare data loader manager
    data_loader_manager = DataLoaderManager(dataset_manager=dataset_manager,
                                            batch_size=args.batch,
                                            gradient_accumulation=args.gradient_accumulation,
                                            num_workers=num_workers,
                                            deterministic=args.deterministic)

    # Declare training, validation and testing manager
    train_valid_test_manager = TrainValidTestManager(data_loader_manager=data_loader_manager,
                                                     file_name=file_name,
                                                     model_name=args.model,
                                                     learning_rate=args.learning_rate,
                                                     weight_decay=args.weight_decay,
                                                     es_patience=args.es_patience,
                                                     es_delta=args.es_delta,
                                                     mixed_precision=args.mixed_precision,
                                                     gradient_accumulation=args.gradient_accumulation,
                                                     pretrained=args.pretrained,
                                                     iou_threshold=args.iou_threshold,
                                                     gradient_clip=args.gradient_clip,
                                                     args_dict=vars(args),
                                                     save_model=args.save_model,
                                                     max_image_size=args.size)

    # Call the training, validation and testing manager to run the pipeline
    train_valid_test_manager(args.epochs)

    # Print the run time of the current experiment
    print(f'\nTotal time for current experiment:\t{str(datetime.now() - start).split(".")[0]}')
