import argparse
import datetime
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import copy
import ray

from reproducibility import seed_worker, set_seed
from src.data.Fuse_Class import FuseDataset
from fuse_config import NUM_WORKERS_DL
from src.models.helper_functions import collate_fn, base_transform, train_transform, test_model, train_model, \
     view_test_image, split_train_valid_test

ray.init(include_dashboard=False)

if __name__ == '__main__':
    # Declare argument parser
    parser = argparse.ArgumentParser(description='Processing inputs')

    # Data source argument
    parser.add_argument('-d', '--data', action='store', type=str, choices=['raw', 'resized'],
                        help='Specify which data source')
    # Resizing argument
    parser.add_argument('-s', '--size', action='store', type=int, default=0,
                        help='Resize the images to size*size (takes an argument: max_resize value (int))')

    # Training argument
    parser.add_argument('-tr', '--train', action='store_false',
                        help='Train a new model')

    # Validation argument
    parser.add_argument('-vs', '--validation_size', action='store', type=float, default=0.1,
                        help='Size of validation set (float as proportion of dataset)')

    # Testing argument
    parser.add_argument('-ts', '--test_size', action='store', type=float, default=0.1,
                        help='Size of test set (float as proportion of dataset)')

    # Number of epochs argument
    parser.add_argument('-e', '--epochs', action='store', type=int, default=10,
                        help='Number of epochs')

    # Batch size argument
    parser.add_argument('-b', '--batch', action='store', type=int, default=100,
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

    # Verbose argument
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='to generate and save graphs')

    # View images using a saved model argument
    parser.add_argument('-i', '--image', action='store', type=str,
                        help='to view images, input - model name')

    # Test saved model argument
    parser.add_argument('-tst_f', '--test_file', action='store', type=str,
                        help='Filename of saved model to test (located in models/)')

    # Parsing arguments
    args = parser.parse_args()

    # Set random seed to ensure reproducibility of results
    set_seed(args.random)

    # Record start time
    start = time.time()

    # Declare file name as yyyy-mm-dd_hh-mm-ss
    filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Display arguments in console
    print('Filename:\t\t\t\t\t', filename)
    print('Train:\t\t\t\t\t\t', args.train)
    print('Validation Size:\t\t\t', args.validation_size)
    print('Test Size:\t\t\t\t\t', args.test_size)
    print('Epochs:\t\t\t\t\t\t', args.epochs)
    print('Batch Size:\t\t\t\t\t', args.batch)
    print('Early Stopping:\t\t\t\t', args.early_stopping)
    print('Size:\t\t\t\t\t\t', args.size)
    print('Mixed Precision:\t\t\t', args.mixed_precision)
    print('Gradient Accumulation Size:\t', args.gradient_accumulation)
    print('Random Seed:\t\t\t\t', args.random)
    print('Save Plots:\t\t\t\t\t', args.verbose)
    print('-' * 100)

    # Assign image and annotations file paths depending on data source
    images_path = f'data/{args.data}'
    annotations_path = f'data/annotations/annotations_{args.data}.csv'

    train_dataset = FuseDataset(
        root=images_path, data_file=annotations_path,
        max_image_size=args.size, transforms=train_transform())
    test_dataset = FuseDataset(
        root=None, data_file=annotations_path,
        max_image_size=args.size, transforms=base_transform())
    val_dataset = FuseDataset(
        root=None, data_file=annotations_path,
        max_image_size=args.size, transforms=base_transform())




    total_dataset = copy.deepcopy(train_dataset)
    train_dataset, test_dataset, val_dataset = split_train_valid_test(train_dataset, test_dataset, val_dataset,
                                                                      args.validation_size, args.test_size)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(args.batch / args.gradient_accumulation),
        shuffle=True, num_workers=NUM_WORKERS_DL, collate_fn=collate_fn, worker_init_fn=seed_worker)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=int(args.batch / args.gradient_accumulation),
        shuffle=False, num_workers=NUM_WORKERS_DL, collate_fn=collate_fn, worker_init_fn=seed_worker)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=int(args.batch / args.gradient_accumulation),
        shuffle=False, num_workers=NUM_WORKERS_DL, collate_fn=collate_fn, worker_init_fn=seed_worker)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('We have: {} examples, {} are training, {} are validation and {} testing'.format(
        len(total_dataset), len(train_dataset), len(val_dataset), len(test_dataset)))
    writer = SummaryWriter('runs/' + filename)
    if args.train:
        train_start = time.time()
        train_model(args.epochs, args.gradient_accumulation, train_data_loader, device,
                    args.mixed_precision, True if args.gradient_accumulation > 1 else False, filename, args.verbose,
                    writer, args.early, args.validation, val_dataset)
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
