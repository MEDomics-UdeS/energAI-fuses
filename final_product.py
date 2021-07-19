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

from src.gui.GuiDataset import GuiDataset
from src.gui.GuiDataLoader import GuiDataLoader

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

    parser.add_argument('-d', '--device', action='store', type=str, default='cpu',
                        help="Select the device for inference")

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

    # Parse arguments
    args = parser.parse_args()

    # Display arguments in console
    print('\n=== Arguments & Hyperparameters ===\n')
    print_dict(vars(args), 8)

    image_size = torch.load(args.model_file_name)["args_dict"]["image_size"]

    # Loading the images dataset
    ds = GuiDataset(args.image_path, num_workers, norm=args.normalize)
    dl = GuiDataLoader(dataset=ds,
                       batch_size=args.batch,
                       gradient_accumulation=1,
                       num_workers=num_workers,
                       deterministic=True)

    # Perform an inference loop and save all images with the ground truth and predicted bounding boxes
    save_test_images(model_file_name=args.model_file_name,
                     data_loader=dl.data_loader,
                     iou_threshold=args.iou_threshold,
                     score_threshold=args.score_threshold,
                     save_path=INFERENCE_PATH,
                     image_size=image_size,
                     device_type=args.device)

    # Print file save path location
    print(f'\nInference results saved to: {INFERENCE_PATH}')

    # Print total time for inference testing
    print(
        f'\nTotal time for inference testing: {str(datetime.now() - start).split(".")[0]}')
