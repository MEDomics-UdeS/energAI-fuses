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

import torch
from src.data.DataLoaderManagers.DataLoaderManager import DataLoaderManager
from src.data.DatasetManagers.DatasetManager import DatasetManager
from src.utils.helper_functions import print_dict, env_tests
from src.visualization.inference import coco_evaluate
from src.models.models import load_model
from src.utils.constants import *

if __name__ == '__main__':
    env_tests()

    # Record start time
    start = datetime.now()

    # Declare argument parser
    parser = argparse.ArgumentParser(description='Processing inputs')

    # Dataset Choice
    parser.add_argument('-ds', '--dataset', action='store', type=str, default='learning',
                        choices=['learning', 'holdout'],
                        help='Dataset to use for inference test')

    # Model file name argument
    parser.add_argument('-mfn', '--model_file_name', action='store', type=str, default='fasterrcnn_200_2021-07-28_14-38-45',
                        help=f'Model file name located in {MODELS_PATH}')

    # Parse arguments
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    save_state = torch.load(MODELS_PATH + args.model_file_name, map_location=device)
    args_dict = save_state['args_dict']

    # Display arguments in console
    print('\n=== Saved Model Test ===\n')
    print_dict(vars(args), 6)

    print('\n=== Saved Model Arguments & Hyperparameters ===\n')
    print_dict(args_dict, 6)

    # Declare dataset manager
    dataset_manager = DatasetManager(images_path=RESIZED_PATH,
                                     targets_path=TARGETS_PATH,
                                     image_size=args_dict['image_size'],
                                     num_workers=args_dict['num_workers'],
                                     data_aug=args_dict['data_aug'],
                                     validation_size=args_dict['validation_size'] if args.dataset == 'learning' else 0,
                                     test_size=args_dict['test_size'] if args.dataset == 'learning' else 1,
                                     norm=args_dict['normalize'],
                                     google_images=not args_dict['no_google_images'],
                                     seed=args_dict['random_seed'])

    # Declare data loader manager
    data_loader_manager = DataLoaderManager(dataset_manager=dataset_manager,
                                            batch_size=args_dict['batch'],
                                            gradient_accumulation=args_dict['gradient_accumulation'],
                                            num_workers=args_dict['num_workers'],
                                            deterministic=args_dict['deterministic'])

    model = load_model(model_name=args_dict['model'],
                       pretrained=not args_dict['no_pretrained'],
                       num_classes=len(CLASS_DICT),
                       progress=True)
    model.load_state_dict(save_state['model'])
    model.to(device)

    # Update bn statistics for the swa_model at the end
    torch.optim.swa_utils.update_bn(data_loader_manager.data_loader_train, model)

    metrics_dict = coco_evaluate(model=model,
                                 data_loader=data_loader_manager.data_loader_test,
                                 desc='Testing saved model',
                                 device=device,
                                 image_size=args_dict['image_size'],
                                 model_name=args_dict['model'])

    # Print the testing object detection metrics results
    print('=== Testing Results ===\n')
    print_dict(metrics_dict, 6, '.2%')

    # # Perform an inference loop and save all images with the ground truth and predicted bounding boxes
    # save_test_images(model_file_name=args.model_file_name,
    #                  data_loader=data_loader_manager.data_loader_test,
    #                  iou_threshold=args.iou_threshold,
    #                  score_threshold=args.score_threshold,
    #                  save_path=INFERENCE_PATH,
    #                  image_size=image_size)
    #
    # # Print file save path location
    # print(f'\nInference results saved to: {INFERENCE_PATH}')

    # Print total time for inference testing
    print(f'\nTotal time for inference testing: {str(datetime.now() - start).split(".")[0]}')
