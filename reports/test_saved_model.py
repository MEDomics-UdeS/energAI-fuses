"""
File:
    reports/test_saved_model.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    File to test a saved model and obtain results in LaTeX code
"""

import argparse
from datetime import datetime
import torch
import os
import json
import pandas as pd

from src.data.SplittingManager import SplittingManager
from src.data.DataLoaderManagers.LearningDataLoaderManager import LearningDataLoaderManager
from src.data.DatasetManagers.LearningDatasetManager import LearningDatasetManager
from src.utils.helper_functions import print_dict, env_tests
from src.visualization.inference import coco_evaluate
from src.models.models import load_model
from src.utils.constants import CLASS_DICT
from reports.parsing_utils import get_latex_exp_name, get_latex_ap_table, save_latex
from reports.constants import AP_DICT

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
    parser.add_argument('-letter', '--experiment_letter', action='store', type=str,
                        default='A', choices=['A', 'D'],
                        help='Experiment letter')

    # Model file name argument
    parser.add_argument('-mp', '--models_path', action='store', type=str,
                        help='Directory containing the models')

    # Save LaTeX results argument
    parser.add_argument('-sv_latex', '--save_latex', action='store', type=bool, default=True,
                        help='Specify whether to save LaTeX output or not')

    # Save JSON results argument
    parser.add_argument('-sv_json', '--save_json', action='store', type=bool, default=True,
                        help='Specify whether to save JSON output or not')

    # Parse arguments
    args = parser.parse_args()

    models_path = os.path.join(args.models_path, args.experiment_letter, 'saved_models')
    json_file_name = os.path.join('reports', f'results_phase_{args.experiment_letter}.json')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    results_dict = {}

    for filename in os.listdir(models_path):
        save_state = torch.load(os.path.join(models_path, filename), map_location=device)
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
            seed=args_dict['seed_split'] if 'seed_split' in args_dict else args_dict['random_seed'],
            google_images=not args_dict['no_google_images'],
            image_size=args_dict['image_size'],
            num_workers=args_dict['num_workers'])

        # Declare dataset manager
        dataset_manager = LearningDatasetManager(
            num_workers=args_dict['num_workers'],
            data_aug=args_dict['data_aug'],
            validation_size=args_dict['validation_size'] if args.dataset == 'learning' else 0,
            test_size=args_dict['test_size'] if args.dataset == 'learning' else 1,
            norm=args_dict['normalize'],
            google_images=not args_dict['no_google_images'],
            seed=args_dict['seed_init'] if 'seed_init' in args_dict else args_dict['random_seed'],
            splitting_manager=splitting_manager,
            current_fold=0
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
        print('\n')

        results_dict[filename] = {'args': args_dict,
                                  'results': metrics_dict}

    if args.save_latex:
        columns = ['Metric', 'Result']
        df = pd.DataFrame(columns=columns)

        for key, value in results_dict[filename]['results'].items():
            if key.startswith('AP'):
                df = df.append({columns[0]: AP_DICT[key], columns[1]: value}, ignore_index=True)

        df[columns[1]] = df[columns[1]].round(4)
        df[columns[1]] = df[columns[1]].apply('{:.4f}'.format)

        output_str = get_latex_exp_name(args.experiment_letter)
        output_str += get_latex_ap_table(df, index=200, letter=args.experiment_letter)

        save_latex(output_str, letter=args.experiment_letter, path='reports/')

    if args.save_json:
        with open(json_file_name, 'w') as fp:
            json.dump(results_dict, fp)

        print(f'\nResults have been saved to the following JSON file: {json_file_name}')

    # Print total time for inference testing
    print(f'\nTotal time for inference testing: {str(datetime.now() - start).split(".")[0]}')
