import argparse
from datetime import datetime
from multiprocessing import cpu_count

from src.data.DataLoaderManager import DataLoaderManager
from src.data.DatasetManager import DatasetManager
from src.utils.helper_functions import print_args, env_tests
from src.visualization.inference import save_test_images
from src.utils.constants import RESIZED_PATH, TARGETS_PATH, INFERENCE_PATH

if __name__ == '__main__':
    env_tests()

    # Record start time
    start = datetime.now()

    # Get number of cpu threads for PyTorch DataLoader and Ray paralleling
    num_workers = cpu_count()

    # Declare argument parser
    parser = argparse.ArgumentParser(description='Processing inputs')

    # Model file name argument
    parser.add_argument('-mfn', '--model_file_name', action='store', type=str, default='2021-04-26_14-46-04',
                        help='Model file name located in models/')

    # To compute mean & std deviation on training set
    parser.add_argument('-ms', '--mean_std', action='store_true',
                        help='Compute mean & standard deviation on training set if true, '
                             'otherwise use precalculated values')

    # Batch size argument
    parser.add_argument('-b', '--batch', action='store', type=int, default=20,
                        help='Batch size')

    # To load IOU Threshold
    parser.add_argument('-iou', '--iou_threshold', action='store', type=float, default=0.5,
                        help='IOU threshold for true/false positive box predictions')

    args = parser.parse_args()

    # Display arguments in console
    print_args(args)

    dataset_manager = DatasetManager(images_path=RESIZED_PATH,
                                     annotations_path=TARGETS_PATH,
                                     num_workers=num_workers,
                                     data_aug=0,
                                     validation_size=0,
                                     test_size=1,
                                     mean_std=args.mean_std)

    data_loader_manager = DataLoaderManager(dataset_manager=dataset_manager,
                                            batch_size=args.batch,
                                            gradient_accumulation=1,
                                            num_workers=num_workers,
                                            deterministic=True)

    save_test_images(args.model_file_name, data_loader_manager.data_loader_test, args.iou_threshold, INFERENCE_PATH)

    print(f'Inference results saved to: {INFERENCE_PATH}')
    print(f'\nTotal time for inference testing: {str(datetime.now() - start).split(".")[0]}')
