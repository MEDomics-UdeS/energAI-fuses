import argparse
import datetime
import os
import torch


from Fuse_Class import FuseDataset
from fuse_config import (ANNOTATION_FILE, LEARNING_RATE, NO_OF_CLASSES,
                         TRAIN_DATAPATH, train_test_split_index)
from helper_functions import collate_fn, get_transform, test_model, train_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Processing inputs.')

    parser.add_argument('-tr', '--train', action="store_true",
                        help='train a new model')
    parser.add_argument('-ts', '--test', action="store_true",
                        help='test a model')

    parser.add_argument('-e', '--epochs', action="store",
                        type=int, help="Number of Epochs")
    parser.add_argument('-b', '--batch', action="store",
                        type=int, help="Batch Size")

    parser.add_argument('-d', '--downsample', action="store", type=int,
                        help='downsample the data (takes an argument: max_resize value (int))')
    parser.add_argument('-mp', '--mixed_precision', action="store_true",
                        help='to use mixed precision')
    parser.add_argument('-g', '--gradient_accumulation', action="store_true",
                        help='to use gradient acculmulation')

    parser.add_argument('-r', '--random', action="store", type=int, required=False,
                        help='to give a seed value')

    parser.add_argument('-v', '--verbose', action="store_true",
                        help='to generate and save graphs')
    args = parser.parse_args()

    filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = filename+"_e"+str(args.epochs)+"_b"+str(args.batch)+"_d"+str(args.downsample or '0') + \
        "_mp"+str(int(args.mixed_precision))+"_g" + \
        str(int(args.gradient_accumulation))
    print("Filename: ", filename)

    print("Train: ", args.train)
    print("Test: ", args.test)

    print("Epochs: ", args.epochs)
    print("Batch Size: ", args.batch)

    print("Downsampling: ", args.downsample)
    print("Mixed Precision: ", args.mixed_precision)
    print("Gradient Accumulation: ", args.gradient_accumulation)

    print("Random Seed: ", args.random)
    print("-"*100)

    resize = bool(args.downsample)
    max_image_size = args.downsample if args.downsample else 1000

    train_dataset = FuseDataset(
        root=TRAIN_DATAPATH, data_file=TRAIN_DATAPATH + ANNOTATION_FILE, max_image_size=max_image_size, resize=resize, transforms=get_transform())
    test_dataset = FuseDataset(
        root=TRAIN_DATAPATH, data_file=TRAIN_DATAPATH + ANNOTATION_FILE, max_image_size=max_image_size, resize=resize, transforms=get_transform())

    # split the dataset in train and test set
    if not args.random:
        torch.manual_seed(1)
    else:
        torch.manual_seed(args.random)
    indices = torch.randperm(len(train_dataset)).tolist()

    train_dataset = torch.utils.data.Subset(
        train_dataset, indices[:-train_test_split_index])
    test_dataset = torch.utils.data.Subset(
        test_dataset, indices[-train_test_split_index:])

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True, num_workers=1,
        collate_fn=collate_fn)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch, shuffle=False, num_workers=1,
        collate_fn=collate_fn)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("We have: {} examples, {} are training and {} testing".format(
        len(indices), len(train_dataset), len(test_dataset)))

    if args.train:
        train_model(args.epochs, train_data_loader, device,
                    args.mixed_precision, args.gradient_accumulation, filename, args.verbose)
    if args.test:
        test_model(test_dataset, device, filename)
