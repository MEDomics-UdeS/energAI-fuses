"""
File:
    test_inference.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Main file to run an inference test on a saved model and show
    model predictions and ground truths boxes on the images
"""

import argparse
from datetime import datetime
from multiprocessing import cpu_count


import torch
from src.data.DataLoaderManager import DataLoaderManager
from src.data.DatasetManager import DatasetManager
from src.utils.helper_functions import print_dict, env_tests
from src.visualization.final_inference import save_test_images
from src.utils.constants import INFERENCE_PATH, MODELS_PATH

if __name__ == '__main__':
    env_tests()

    # Record start time
    start = datetime.now()

    # Get number of cpu threads for PyTorch DataLoader and Ray paralleling
    num_workers = cpu_count()

    # Declare argument parser
    parser = argparse.ArgumentParser(description='Processing inputs')

    # Model file name argument
    parser.add_argument('-mfn', '--model_file_name', action='store', type=str,
                        help=f'Model file name located in {MODELS_PATH}')

    parser.add_argument('-img', '--image_path', action='store', type=str,
                        help="Image directory to use for inference test")

    parser.add_argument('-inf', '--inference_path', action='store', type=str,
                        help="Image directory to store images after inference")

    # Compute mean & std deviation on training set argument
    parser.add_argument('-norm', '--normalize', action='store', type=str,
                        choices=['precalculated',
                                 'calculated',
                                 'disabled'],
                        default='precalculated',
                        help='Normalize the training dataset by mean & std using '
                             'precalculated values, calculated values or disabled')

    # Batch size argument
    parser.add_argument('-b', '--batch', action='store', type=int, default=1,
                        help='Batch size')

    # IOU threshold argument
    parser.add_argument('-iou', '--iou_threshold', action='store', type=float, default=0.5,
                        help='IOU threshold')

    # Score threshold argument
    parser.add_argument('-sc', '--score_threshold', action='store', type=float, default=0.5,
                        help='Score threshold to filter box predictions')

    # Google Images argument
    parser.add_argument('-no-gi', '--no_google_images', action='store_true',
                        help='If specified, the Google Images photos will be excluded from the training subset')

    # Parse arguments
    args = parser.parse_args()

    # Display arguments in console
    print('\n=== Arguments & Hyperparameters ===\n')
    print_dict(vars(args), 6)

    image_size = torch.load(args.model_file_name)["args_dict"]["image_size"]
    
    # Declare dataset manager
    dataset_manager = DatasetManager(images_path=args.image_path,
                                     targets_path=None,
                                     image_size=image_size,
                                     num_workers=num_workers,
                                     data_aug=0,
                                     validation_size=0,
                                     test_size=1,
                                     norm=args.normalize,
                                     google_images=False,
                                     seed=0)
        
    # Declare data loader manager
    data_loader_manager = DataLoaderManager(dataset_manager=dataset_manager,
                                            batch_size=args.batch,
                                            gradient_accumulation=1,
                                            num_workers=num_workers,
                                            deterministic=True)

    # Perform an inference loop and save all images with the ground truth and predicted bounding boxes
    save_test_images(model_file_name=args.model_file_name,
                     data_loader=data_loader_manager.data_loader_test,
                     iou_threshold=args.iou_threshold,
                     score_threshold=args.score_threshold,
                     save_path=INFERENCE_PATH,
                     image_size=image_size)

    # Print file save path location
    print(f'\nInference results saved to: {INFERENCE_PATH}')

    # Print total time for inference testing
    print(
        f'\nTotal time for inference testing: {str(datetime.now() - start).split(".")[0]}')
