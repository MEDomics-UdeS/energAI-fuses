import argparse

from multiprocessing import cpu_count


from src.data.DataLoaderManager import DataLoaderManager
from src.data.DatasetManager import DatasetManager
from src.visualization.inference import view_test_images
from env_tests import env_tests
from constants import IMAGES_PATH, ANNOTATIONS_PATH

if __name__ == '__main__':
    env_tests()

    # Get number of cpu threads for PyTorch DataLoader and Ray paralleling
    num_workers = cpu_count()

    # Declare argument parser
    parser = argparse.ArgumentParser(description='Processing inputs')

    # Model file name argument
    parser.add_argument('-mfn', '--model_file_name', action='store', type=str, default='2021-04-25_19-29-08',
                        help='Model file name located in models/')
    # Data source argument
    parser.add_argument('-d', '--data', action='store', type=str, choices=['raw', 'resized'], default='raw',
                        help='Specify which data source')

    # To compute mean & std deviation on training set
    parser.add_argument('-ms', '--mean_std', action='store_true',
                        help='Compute mean & standard deviation on training set if true, '
                             'otherwise use precalculated values')

    # Batch size argument
    parser.add_argument('-b', '--batch', action='store', type=int, default=20,
                        help='Batch size')

    args = parser.parse_args()

    dataset_manager = DatasetManager(images_path=IMAGES_PATH,
                                     annotations_path=ANNOTATIONS_PATH,
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

    view_test_images(args.model_file_name, data_loader_manager.data_loader_test)
