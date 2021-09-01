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
from src.data.DataLoaderManagers.LearningDataLoaderManager import LearningDataLoaderManager
from src.data.DatasetManagers.LearningDatasetManager import LearningDatasetManager
from src.data.DataLoaderManagers.GuiDataLoaderManager import GuiDataLoaderManager
from src.data.DatasetManagers.GuiDatasetManager import GuiDatasetManager
from src.utils.helper_functions import print_dict, env_tests
from src.visualization.inference import save_test_images
from src.utils.constants import RESIZED_PATH, TARGETS_PATH, INFERENCE_PATH, MODELS_PATH

if __name__ == '__main__':
    # Record start time
    start = datetime.now()

    # Run environment tests
    env_tests()

    # Get number of cpu threads for PyTorch DataLoader and Ray paralleling
    num_workers = cpu_count()

    # Declare argument parser
    parser = argparse.ArgumentParser(description='Processing inputs')

    # Model file name argument
    parser.add_argument('-mfn', '--model_file_name', action='store', type=str,
                        help=f'Model file name located in {MODELS_PATH}')

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

    # Visualization using the GUI
    parser.add_argument('-gui', '--with_gui', action='store_true',
                        help='If specified, the inference results will be shown in the GUI application')

    # Specified images path for inference
    parser.add_argument('-img', '--image_path', action='store', type=str, default=None,
                        help="Image directory to use for inference test")

    # Selected device
    parser.add_argument('-d', '--device', action='store', type=str, default='cpu',
                        help="Select the device for inference")

    # Selected ground truth CSV file
    parser.add_argument("-gtf", "--ground_truth_file", action="store", type=str, default=None,
                        help="Select a CSV file for ground truth drawing on images")

    # Parsing arguments
    args = parser.parse_args()
    
    # Display arguments in console
    print('\n=== Arguments & Hyperparameters ===\n')
    print_dict(vars(args), 6)

    image_size = torch.load(args.model_file_name, map_location=torch.device('cpu'))["args_dict"]["image_size"]

    # Using GUI specific data managers
    if args.with_gui:
        #TODO there should be a try-except for GUI specific settings if someone wants to run the script standalone in a console instead of going through gui.py
        # Loading the images dataset
        dataset_manager = GuiDatasetManager(image_size=image_size,
                                            images_path=args.image_path,
                                            num_workers=num_workers,
                                            gt_file=args.ground_truth_file)
        
        data_loader_manager = GuiDataLoaderManager(dataset=dataset_manager,
                                                   batch_size=args.batch,
                                                   gradient_accumulation=1,
                                                   num_workers=num_workers,
                                                   deterministic=True)
    else:
        # Declare dataset manager
        dataset_manager = LearningDatasetManager(images_path=RESIZED_PATH,
                                                 targets_path=TARGETS_PATH,
                                                 image_size=image_size,
                                                 num_workers=num_workers,
                                                 data_aug=0,
                                                 validation_size=0,
                                                 test_size=1,
                                                 norm=args.normalize,
                                                 google_images=not args.no_google_images,
                                                 seed=0)

        # Declare data loader manager
        data_loader_manager = LearningDataLoaderManager(dataset_manager=dataset_manager,
                                                        batch_size=args.batch,
                                                        gradient_accumulation=1,
                                                        num_workers=num_workers,
                                                        deterministic=True)

    # Perform an inference loop and save all images with the ground truth and predicted bounding boxes
    save_test_images(model_file_name=args.model_file_name,
                     data_loader=data_loader_manager.data_loader_test,
                     with_gui=args.with_gui,
                     iou_threshold=args.iou_threshold,
                     score_threshold=args.score_threshold,
                     save_path=INFERENCE_PATH,
                     img_path=args.image_path,
                     image_size=image_size,
                     device_type=args.device)

    # Print file save path location
    print(f'\nInference results saved to: {INFERENCE_PATH}')

    # Print total time for inference testing
    print(f'\nTotal time for inference testing: {str(datetime.now() - start).split(".")[0]}')
