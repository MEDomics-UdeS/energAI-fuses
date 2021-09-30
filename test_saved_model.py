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
import os
import json

from src.data.SplittingManager import SplittingManager
from src.data.DataLoaderManagers.LearningDataLoaderManager import LearningDataLoaderManager
from src.data.DatasetManagers.LearningDatasetManager import LearningDatasetManager
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
    parser.add_argument('-ds', '--dataset', action='store', type=str, default='holdout',
                        choices=['learning', 'holdout'],
                        help='Dataset to use for inference test')

    # Model file name argument
    parser.add_argument('-mp', '--models_path', action='store', type=str,
                        default='/home/simon/Desktop/Results_Fuses/C/saved_models/',
                        help='Directory containing the models')

    # JSON results file name argument
    parser.add_argument('-json', '--json_file_name', action='store', type=str,
                        default='test_saved_models_results.json',
                        help='File name and location of JSON results file to save')

    # Parse arguments
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    results_dict = {}

    for filename in os.listdir(args.models_path):
        save_state = torch.load(args.models_path + filename, map_location=device)
        args_dict = save_state['args_dict']

        # Display arguments in console
        print('\n=== Saved Model Test ===\n')
        print_dict(vars(args), 6)

        print('\n=== Saved Model Arguments & Hyperparameters ===\n')
        print_dict(args_dict, 6)

        # Declare splitting manager
        splitting_manager = SplittingManager(
            dataset=args.dataset,
            validation_size=args_dict['validation_size'] if args.dataset == 'learning' else 0,
            test_size=args_dict['test_size'] if args.dataset == 'learning' else 1,
            k_cross_valid=1,
            seed=args_dict['random_seed'],
            google_images=not args_dict['no_google_images'],
            image_size=args_dict['image_size'],
            num_workers=args_dict['num_workers'])

        # Declare dataset manager
        dataset_manager = LearningDatasetManager(
            images_path=RESIZED_LEARNING_PATH if args.dataset == 'learning' else RESIZED_HOLDOUT_PATH,
            targets_path=TARGETS_LEARNING_PATH,
            image_size=args_dict['image_size'],
            num_workers=args_dict['num_workers'],
            data_aug=args_dict['data_aug'],
            validation_size=args_dict['validation_size'] if args.dataset == 'learning' else 0,
            test_size=args_dict['test_size'] if args.dataset == 'learning' else 1,
            norm=args_dict['normalize'],
            google_images=not args_dict['no_google_images'],
            seed=args_dict['random_seed'],
            splitting_manager=splitting_manager,
            current_fold=1
        )

        # Declare data loader manager
        data_loader_manager = LearningDataLoaderManager(dataset_manager=dataset_manager,
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

        if args.dataset == 'learning':
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

        results_dict[filename] = {'args': args_dict,
                                  'results': metrics_dict}

    with open(args.json_file_name, 'w') as fp:
        json.dump(results_dict, fp)

    print(f'\nResults have been saved to the following JSON file: {args.json_file_name}')

    # Print total time for inference testing
    print(f'\nTotal time for inference testing: {str(datetime.now() - start).split(".")[0]}')
