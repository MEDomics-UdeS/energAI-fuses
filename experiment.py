import argparse
from datetime import datetime
from multiprocessing import cpu_count

from src.data.DataLoaderManager import DataLoaderManager
from src.data.DatasetManager import DatasetManager
from src.data.resize_images import resize_images
from src.models.TrainValidTestManager import TrainValidTestManager
from src.utils.reproducibility import set_deterministic
from constants import IMAGES_PATH, ANNOTATIONS_PATH

if __name__ == '__main__':
    # Record start time
    start = datetime.now()

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
    parser.add_argument('-b', '--batch', action='store', type=int, default=20,
                        help='Batch size')

    # Early stopping argument
    parser.add_argument('-es', '--early_stopping', action='store', type=int,
                        help='Early stopping patience')

    # Mixed precision argument
    parser.add_argument('-mp', '--mixed_precision', action='store_true',
                        help='To use mixed precision')

    # Gradient accumulation argument
    parser.add_argument('-g', '--gradient_accumulation', action='store', type=int, default=1,
                        help='To use gradient accumulation (takes an argument: accumulation_size (int))')

    # Gradient clipping argument
    parser.add_argument('-gc', '--gradient_clip', action='store', type=float, default=5,
                        help='Gradient clipping value')

    # Random seed argument
    parser.add_argument('-rs', '--random_seed', action='store', type=int, default=42,
                        help='Set random seed')

    # Deterministic argument
    parser.add_argument('-dt', '--deterministic', action='store_true',
                        help='Set deterministic behaviour')

    # To compute mean & std deviation on training set
    parser.add_argument('-ms', '--mean_std', action='store_true',
                        help='Compute mean & standard deviation on training set if true, '
                             'otherwise use precalculated values')

    # To load IOU Threshold
    parser.add_argument('-iou', '--iou_threshold', action='store', type=float, default=0.5,
                        help='IOU threshold for true/false positive box predictions')

    # To load learning rate
    parser.add_argument('-lr', '--learning_rate', action='store', type=float, default=0.0003,
                        help='Learning rate for optimizer')

    # To load weight decay
    parser.add_argument('-wd', '--weight_decay', action='store', type=float, default=0,
                        help='Weight decay for optimizer')

    # To load model
    parser.add_argument('-mo', '--model', action='store', type=str,
                        choices=['fasterrcnn_resnet50_fpn',
                                 'fasterrcnn_mobilenet_v3_large_fpn',
                                 'fasterrcnn_mobilenet_v3_large_320_fpn',
                                 'retinanet_resnet50_fpn',
                                 'detr', 'perceiver'],
                        default='fasterrcnn_resnet50_fpn',
                        help='Specify which object detection model to use')

    # To load pretrained model
    parser.add_argument('-pt', '--pretrained', action='store_false',
                        help='Load pretrained model')

    # Parsing arguments
    args = parser.parse_args()
    args_dic = vars(args)

    set_deterministic(args.deterministic, args.random_seed)

    # Declare file name as yyyy-mm-dd_hh-mm-ss
    file_name = start.strftime('%Y-%m-%d_%H-%M-%S')

    # Display arguments in console
    print('\n=== Arguments & Hyperparameters ===\n')

    for key, value in args_dic.items():
        print(f'{key}:{" " * (27 - len(key))}{value}')

    print('\n')

    if args.data == 'raw':
        resize_images(args.size, num_workers)

    dataset_manager = DatasetManager(images_path=IMAGES_PATH,
                                     annotations_path=ANNOTATIONS_PATH,
                                     num_workers=num_workers,
                                     data_aug=args.data_aug,
                                     validation_size=args.validation_size,
                                     test_size=args.test_size,
                                     mean_std=args.mean_std)

    data_loader_manager = DataLoaderManager(dataset_manager=dataset_manager,
                                            batch_size=args.batch,
                                            gradient_accumulation=args.gradient_accumulation,
                                            num_workers=num_workers,
                                            deterministic=args.deterministic)

    train_valid_test_manager = TrainValidTestManager(data_loader_manager=data_loader_manager,
                                                     file_name=file_name,
                                                     model_name=args.model,
                                                     learning_rate=args.learning_rate,
                                                     weight_decay=args.weight_decay,
                                                     early_stopping=args.early_stopping,
                                                     mixed_precision=args.mixed_precision,
                                                     gradient_accumulation=args.gradient_accumulation,
                                                     pretrained=args.pretrained,
                                                     iou_threshold=args.iou_threshold,
                                                     gradient_clip=args.gradient_clip,
                                                     args_dic=args_dic)
    train_valid_test_manager(args.epochs)

    print(f'\nTotal time for current experiment:\t{str(datetime.now() - start).split(".")[0]}')
