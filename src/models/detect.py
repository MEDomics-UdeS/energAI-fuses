import argparse
import datetime
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import copy
import ray

from reproducibility import seed_worker, set_seed
from src.data.Fuse_Class import FuseDataset
from fuse_config import TRAIN_TEST_SPLIT, NUM_WORKERS_DL
from src.models.helper_functions import collate_fn, base_transform, train_transform, test_model, train_model, \
     view_test_image, split_trainset

ray.init(include_dashboard=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing inputs.')

    parser.add_argument('-tr', '--train', action="store_true",
                        help='train a new model')
    parser.add_argument('-val', '--validation', action="store", type=int,
                        help='activate validation a model', default=1)
    parser.add_argument('-ts', '--test', action="store_true",
                        help='test a model')

    parser.add_argument('-tstf', '--testfile', action="store", type=str,
                        help='test a model with filename')

    parser.add_argument('-e', '--epochs', action="store",
                        type=int, help="Number of Epochs")
    parser.add_argument('-b', '--batch', action="store",
                        type=int, help="Batch Size")
    parser.add_argument('-es', '--early', action="store",
                        type=int, help="Early Stopping")

    parser.add_argument('-s', '--size', action="store", type=int,
                        help='resize the images to size*size (takes an argument: max_resize value (int))',
                        default=1000)
    parser.add_argument('-mp', '--mixed_precision', action="store_true",
                        help='to use mixed precision')
    parser.add_argument('-g', '--gradient_accumulation', action="store", type=int,
                        help='to use gradient accumulation (takes an argument: accumulation_size (int))',
                        default=1)

    parser.add_argument('-r', '--random', action="store", type=int, required=False,
                        help='to give a seed value', default=1)
    parser.add_argument('-i', '--image', action="store", type=str,
                        help='to view images, input - model name')

    parser.add_argument('-v', '--verbose', action="store_true",
                        help='to generate and save graphs')
    args = parser.parse_args()

    set_seed(args.random)

    start = time.time()
    filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{filename}_e{args.epochs}_b{args.batch}_s{args.size}_mp{int(args.mixed_precision)}" \
               f"_g{int(args.gradient_accumulation)}"

    if args.image:
        filename = args.image
    print("Filename:\t\t\t\t\t", filename)

    print("Train:\t\t\t\t\t\t", args.train)
    print("Validation:\t\t\t\t\t", bool(args.validation))
    print("Test:\t\t\t\t\t\t", args.test)

    print("Epochs:\t\t\t\t\t\t", args.epochs)
    print("Batch Size:\t\t\t\t\t", args.batch)
    print("Early Stopping:\t\t\t\t", args.early)

    print("Size:\t\t\t\t\t\t", args.size)
    print("Mixed Precision:\t\t\t", args.mixed_precision)
    print("Gradient Accumulation Size:\t", args.gradient_accumulation)

    print("Random Seed:\t\t\t\t", args.random)
    print("Save Plots:\t\t\t\t\t", args.verbose)
    print("-" * 100)

    images_path = 'data/raw'
    annotations_path = 'data/annotations/annotations.csv'

    train_dataset = FuseDataset(
        root=images_path, data_file=annotations_path,
        max_image_size=args.size, transforms=train_transform(), save=True)
    test_dataset = FuseDataset(
        root=None, data_file=annotations_path,
        max_image_size=args.size, transforms=base_transform(), save=True)
    val_dataset = FuseDataset(
        root=None, data_file=annotations_path,
        max_image_size=args.size, transforms=base_transform(), save=True)




    total_dataset = copy.deepcopy(train_dataset)
    train_dataset, test_dataset, val_dataset = split_trainset(train_dataset, test_dataset, val_dataset,
                                                              TRAIN_TEST_SPLIT, args.random)

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
    print("We have: {} examples, {} are training, {} are validation and {} testing".format(
        len(total_dataset), len(train_dataset), len(val_dataset), len(test_dataset)))
    writer = SummaryWriter("runs/" + filename)
    if args.train:
        train_start = time.time()
        train_model(args.epochs, args.gradient_accumulation, train_data_loader, device,
                    args.mixed_precision, True if args.gradient_accumulation > 1 else False, filename, args.verbose,
                    writer, args.early, args.validation, val_dataset)
        print("Total Time Taken (minutes): ", round((time.time() - train_start) / 60, 2))
    if args.test:
        test_model(test_dataset, device, filename, writer)

    if args.testfile:
        test_model(test_dataset, device, args.testfile, writer)

    if args.image:
        for i in range(len(total_dataset)):
            print(i, len(total_dataset), end=" ")
            view_test_image(i, total_dataset, filename)
    print("Total Time Taken (minutes): ", round((time.time() - start) / 60, 2))

    writer.flush()
    writer.close()
