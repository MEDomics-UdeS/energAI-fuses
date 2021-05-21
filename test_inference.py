"""
File:
    test_inference.py

Authors:
    - Simon Giard-Leroux
    - Shreyas Sunil Kulkarni

Description:
    Main file to run an inference test on a saved model and show
    model predictions and ground truths boxes on the images
"""

import argparse
from datetime import datetime
from multiprocessing import cpu_count

from src.data.DataLoaderManager import DataLoaderManager
from src.data.DatasetManager import DatasetManager
from src.utils.helper_functions import print_args, env_tests
from src.visualization.inference import save_test_images
from src.utils.constants import RESIZED_PATH, TARGETS_PATH, INFERENCE_PATH, MODELS_PATH

if __name__ == '__main__':
    env_tests()

    # Record start time
    start = datetime.now()

    # Get number of cpu threads for PyTorch DataLoader and Ray paralleling
    num_workers = cpu_count()

    # Declare argument parser
    parser = argparse.ArgumentParser(description='Processing inputs')

    # Model file name argument
    parser.add_argument('-mfn', '--model_file_name', action='store', type=str, default='2021-05-20_12-03-28_s1024',
                        help=f'Model file name located in {MODELS_PATH}')

    # To compute mean & std deviation
    parser.add_argument('-ms', '--mean_std', action='store_true',
                        help='Compute mean & standard deviation on training set if true, '
                             'otherwise use precalculated values')

    # Batch size argument
    parser.add_argument('-b', '--batch', action='store', type=int, default=20,
                        help='Batch size')

    # IOU threshold argument
    parser.add_argument('-iou', '--iou_threshold', action='store', type=float, default=0.5,
                        help='IOU threshold')

    # Score threshold argument
    parser.add_argument('-sc', '--score_threshold', action='store', type=float, default=0.5,
                        help='Score threshold to filter box predictions')

    # Google Images argument
    parser.add_argument('-no_gi', '--no_google_images', action='store_true',
                        help='Exclude the Google Images photos from the training subset')

    # Parse arguments
    args = parser.parse_args()

    # Display arguments in console
    print_args(args)

    # Declare dataset manager
    dataset_manager = DatasetManager(images_path=RESIZED_PATH,
                                     targets_path=TARGETS_PATH,
                                     max_image_size=0,
                                     num_workers=num_workers,
                                     data_aug=0,
                                     validation_size=0,
                                     test_size=1,
                                     mean_std=args.mean_std,
                                     no_gi=args.no_google_images)

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
                     save_path=INFERENCE_PATH)

    # Print file save path location
    print(f'\nInference results saved to: {INFERENCE_PATH}')

    # Print total time for inference testing
    print(f'\nTotal time for inference testing: {str(datetime.now() - start).split(".")[0]}')
