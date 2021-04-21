import argparse
import datetime

from multiprocessing import cpu_count

from src.data.FuseDataset import FuseDataset
from src.data.resize_images import resize_images
from src.models.helper_functions import *
from src.utils.seed import seed_worker, set_seed

from constants import *
from src.data.DatasetManager import DatasetManager
from src.data.DataLoaderManager import DataLoaderManager
from src.models.TrainValidTestManager import TrainValidTestManager

if __name__ == '__main__':
    # Get number of cpu threads for PyTorch DataLoader and Ray paralleling
    num_workers = cpu_count()

    # Declare argument parser
    parser = argparse.ArgumentParser(description='Processing inputs')

    # Data source argument
    parser.add_argument('-d', '--data', action='store', type=str, choices=['raw', 'resized'], default='raw',
                        help='Specify which data source')

    # Resizing argument
    parser.add_argument('-s', '--size', action='store', type=int, default=1000,
                        help='Resize the images to size*size (takes an argument: max_resize value (int))')

    # Training argument
    parser.add_argument('-tr', '--train', action='store_false',
                        help='Train a new model')

    # Data augmentation argument
    parser.add_argument('-da', '--data_aug', action='store', type=float, default=0.25,
                        help='Value of data augmentation for training dataset (0: no aug)')

    # Validation argument
    parser.add_argument('-vs', '--validation_size', action='store', type=float, default=0.1,
                        help='Size of validation set (float as proportion of dataset)')

    # Testing argument
    parser.add_argument('-ts', '--test_size', action='store', type=float, default=0.1,
                        help='Size of test set (float as proportion of dataset)')

    # Number of epochs argument
    parser.add_argument('-e', '--epochs', action='store', type=int, default=1,
                        help='Number of epochs')

    # Batch size argument
    parser.add_argument('-b', '--batch', action='store', type=int, default=30,
                        help='Batch size')

    # Early stopping argument
    parser.add_argument('-es', '--early_stopping', action='store', type=int,
                        help='Early stopping')

    # Mixed precision argument
    parser.add_argument('-mp', '--mixed_precision', action='store_true',
                        help='To use mixed precision')

    # Gradient accumulation argument
    parser.add_argument('-g', '--gradient_accumulation', action='store', type=int, default=1,
                        help='To use gradient accumulation (takes an argument: accumulation_size (int))')

    # Random seed argument
    parser.add_argument('-r', '--random', action='store', type=int, required=False,
                        help='Set random seed', default=1)

    # View images using a saved model argument
    parser.add_argument('-i', '--image', action='store', type=str,
                        help='to view images, input - model name')

    # Test saved model argument
    parser.add_argument('-tst_f', '--test_file', action='store', type=str,
                        help='Filename of saved model to test (located in models/)')

    # To compute mean & std deviation on training set
    parser.add_argument('-ms', '--mean_std', action='store_true',
                        help='Compute mean & standard deviation on training set if true, '
                             'otherwise use precalculated values')

    # Parsing arguments
    args = parser.parse_args()

    # Set random seed to ensure reproducibility of results
    set_seed(args.random)

    # Record start time
    start = time.time()

    # Declare file name as yyyy-mm-dd_hh-mm-ss
    file_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Display arguments in console
    print('\nFilename:\t\t\t\t\t', file_name)

    print('\nData Source:\t\t\t\t', args.data)

    if args.data == 'raw':
        print(f'Image Size:\t\t\t\t\t{args.size} x {args.size}')

    print('Train:\t\t\t\t\t\t', args.train)
    print('Validation Size:\t\t\t', args.validation_size)
    print('Test Size:\t\t\t\t\t', args.test_size)
    print('Epochs:\t\t\t\t\t\t', args.epochs)
    print('Batch Size:\t\t\t\t\t', args.batch)
    print('Early Stopping:\t\t\t\t', args.early_stopping)
    print('Mixed Precision:\t\t\t', args.mixed_precision)
    print('Gradient Accumulation Size:\t', args.gradient_accumulation)
    print('Random Seed:\t\t\t\t', args.random)
    print('View Images Mode:\t\t\t', args.image)
    print('Test File Mode:\t\t\t\t', args.test_file)
    print('Compute Mean & Std:\t\t\t', args.mean_std)
    print('-' * 100)

    # Assign image and annotations file paths depending on data source
    images_path = 'data/resized/'
    annotations_path = 'data/annotations/targets_resized.json'

    # Resize images if 'raw' has been specified as the data source
    if args.data == 'raw':
        resize_images(args.size, num_workers)

    dataset_manager = DatasetManager(images_path, annotations_path, num_workers,
                                     args.data_aug, args.validation_size, args.test_size, args.mean_std)

    data_loader_manager = DataLoaderManager(dataset_manager, args.batch, args.gradient_accumulation, num_workers)

    train_valid_test_manager = TrainValidTestManager(data_loader_manager=data_loader_manager,
                                                     file_name=file_name,
                                                     model_name='fasterrcnn_mobilenet_v3_large_fpn',
                                                     learning_rate=0.0003,
                                                     momentum=0.9,
                                                     weight_decay=0,
                                                     early_stopping=args.early_stopping,
                                                     mixed_precision=args.mixed_precision,
                                                     gradient_accumulation=args.gradient_accumulation,
                                                     pretrained=True,
                                                     iou_threshold=0.5)
    train_valid_test_manager.train_model(args.epochs)

    if args.train:
        train_start = time.time()
        train_model(args.epochs, args.gradient_accumulation, dataloader_manager.train_data_loader, device,
                    args.mixed_precision, True if args.gradient_accumulation > 1 else False, filename,
                    writer, args.early_stopping, args.validation_size, dataset_manager.valid_dataset)
        print('Total Time Taken (minutes): ', round((time.time() - train_start) / 60, 2))
    if args.test:
        test_model(test_dataset, device, filename, writer)

    if args.testfile:
        test_model(test_dataset, device, args.testfile, writer)

    if args.image:
        for i in range(len(total_dataset)):
            print(i, len(total_dataset), end=' ')
            view_test_image(i, total_dataset, filename)
    print('Total Time Taken (minutes): ', round((time.time() - start) / 60, 2))

    writer.flush()
    writer.close()
