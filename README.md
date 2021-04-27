# EnergAI : Fuse Detection

This repository contains the fuse detection code for the EnergAI project.

## Authors
* [Simon Giard-Leroux](https://github.com/sgiardl) (Université de Sherbrooke / CIMA+)
* [Shreyas Sunil Kulkarni](https://github.com/Kuyas) (Birla Institute of Technology and Science, Pilani)
* [Martin Vallières](https://github.com/mvallieres) (Université de Sherbrooke)

## Introduction
This project implements a supervised learning PyTorch-based end-to-end object detection pipeline for the purpose of detecting and
classifying fuses in low-voltage electrical installations.

## Installation
Install dependencies on a python environment
```
$ pip3 install -r requirements.txt
```

## Module Details: experiment.py & batch_experiment.py

### Description

This file enables users to run different experiments using the developed pipeline.

### Arguments

| Short 	| Long              	    | Type  | Default           	      | Choices                                                                       	                                                                                            | Description                                                                   	|
|-----------|---------------------------|-------|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------	|
| `-s`    	| `--size`           	    | int   | `1024`    	              |                                                 	                                                                                                                        | Resize the images to size * size                                                  |
| `-da`    	| `--data_aug`         	    | float | `0.25`         	          |                                                         	                                                                                                                | Value of data augmentation for training dataset                                   |
| `-vs`   	| `--validation_size`       | float | `0.1`               	      |                                                                               	                                                                                            | Size of validation set (float as proportion of dataset) 	                        |
| `-ts`   	| `--test_size`           	| float | `0.1`               	      |                                                                               	                                                                                            | Size of test set (float as proportion of dataset)	                                |
| `-e`    	| `--epochs`          	    | int   | `1`                	      |                                                                               	                                                                                            | Number of epochs                        	                                        |
| `-b`   	| `--batch`  	            | int   | `20` 	                      | 	                                                                                                                                                                        | Batch size                                                  	                    |
| `-esp`   	| `--es_patience` 	        | int   | `None`            	      |                                                                               	                                                                                            | Early stopping patience (number of epochs without improvement)                    |
| `-esd`    | `--es_delta`        	    | float | `0`                 	      |                                                                               	                                                                                            | Early stopping delta (tolerance to evaluate improvement)                     	    |
| `-mp`    	| `--mixed_precision`  	    | bool  | `False`                     |                                                                               	                                                                                            | Boolean to use mixed precision                  	                                |
| `-g`   	| `--gradient_accumulation` | int   | `1`            	          |                                                                               	                                                                                            | Gradient accumulation size (1 : no gradient accumulation)                         |
| `-gc`   	| `--gradient_clip`    	    | float | `5`                 	      |                                                                               	                                                                                            | Gradient clipping value                                                          	|
| `-rs`   	| `--random_seed`      	    | int 	| `42`            	          |                                                                               	                                                                                            | Random seed, only set if deterministic is set to True           	                |
| `-dt`   	| `--deterministic`        	| bool 	| `False`             	      |                                                                               	                                                                                            | Boolean to force deterministic behavior           	                            |
| `-ms`   	| `--mean_std`        	    | bool  | `False`                     |                                                                               	                                                                                            | Boolean to compute mean & standard deviation RGB normalization values            	|
| `-iou`   	| `--iou_threshold`         | float | `0.5`                       |                                                                               	                                                                                            | Intersection-over-union (IOU) threshold to filter bounding box predictions        |
| `-lr`   	| `--learning_rate`         | float | `0.0003`                    |                                                                               	                                                                                            | Learning rate for Adam optimizer                                                  |
| `-wd`   	| `--weight_decay`          | float | `0`                         |                                                                               	                                                                                            | Weight decay (L2 penalty) for Adam optimizer                                      |
| `-mo`   	| `--model`                 | str   | `'fasterrcnn_resnet50_fpn'` | `'fasterrcnn_resnet50_fpn'`<br>`'fasterrcnn_mobilenet_v3_large_fpn'`<br>`'fasterrcnn_mobilenet_v3_large_320_fpn'`<br>`'retinanet_resnet50_fpn'`<br>`'detr'`<br>`'perceiver'`| Object detection model                                                            |
| `-pt`   	| `--pretrained`            | bool  | `True`                      |                                                                               	                                                                                            | Boolean to specify to load a pretrained model                                     |
| `-sv`   	| `--save_model`            | bool  | `True`                      |                                                                               	                                                                                            | Boolean to specify to save the trained model                                      |

``-h``, ``--help``
show this help message and exit

### Examples of basic use:

To run a single experiment:
```
python src/models/experiment.py --epochs 3
```

To run a batch of multiple experiments that are specified in `batch_experiment.py`:
```
python batch_experiment.py
```

To view log runs and hyperparameters in tensorboard:
```
tensorboard --logdir=logdir
```
## Module Details: test_inference.py

### Description

This file enables users to run an inference test on a saved model and show model predictions and ground truths boxes on the images.

### Arguments

| Short 	| Long              	    | Type    	| Default           	| Description                                                   |
|-----------|-----------------------	|---------	|-----------------------|---------------------------------------------------------------|
| `-mfn`    | `--model_file_name`       | str     	|     	                | File name of the saved model to load                          |
| `-ms`    	| `--mean_std`              | bool     	| `False`               | Boolean to compute mean & standard deviation RGB normalization values       |
| `-b`   	| `--batch`         	    | int     	| `20`                  | Batch size
| `-iou`   	| `--iou_threshold`         | float     | `0.5`                 | Intersection-over-union (IOU) threshold to filter bounding box predictions        |

``-h``, ``--help``
show this help message and exit

### Examples of basic use:

To plot the active learning curve for a particular experiments batch:
```
python test_inference.py --model_file_name 2021-04-27_16-34-27
```

## Project Organization

    ├── data
    │   └── annotations            	 	<- The original annotations .csv and rezised targets .json are saved here.
    │   └── inference            	 	<- Inference test images showing the predicted vs ground truth bounding boxes are saved here.
    │   └── raw            	 	        <- The original, immutable data dump, where the data gets downloaded.
    │   └── resized            	 	<- Resized images are saved here.
    ├── logir                               <- Tensorboard run logs are saved here.
    │
    ├── models             	 	        <- Trained models are saved here.
    │
    ├── src                	 	        <- Source code for use in this project.
    │   ├── data           	 	        <- Scripts to download or generate data.
    │   │   ├── DataLoaderManager.py
    │   │   ├── DatasetManager.py
    │   │   └── FuseDataset.py
    │   │
    │   ├── models         	 	        <- Scripts to train models and then use trained models to make predictions.
    │   │   ├── EarlyStopper.py
    │   │   ├── experiment.py
    │   │   ├── SummaryWriter.py
    │   │   └── TrainValidTestManager.py
    │   │
    │   ├── utils         	 	        <- Utility scripts.
    │   │   ├── constants.py
    │   │   ├── helper_functions.py
    │   │   ├── OCR.py
    │   │   └── reproducibility.py
    │   │
    │   └── visualization  	 	        <- Scripts to create exploratory and results oriented visualizations
    │       └── inference.py
    │
    ├── .gitignore			        <- File that lists which files git can ignore.
    │
    ├── batch_experiment.py      	 	<- File to run a batch of experiments.
    │
    ├── LICENSE        	 	        <- License file.
    │
    ├── README.md          	 	        <- The top-level README for developers using this project.
    │
    ├── requirements.txt   	 	        <- The requirements file for reproducing the analysis environment,
    │					   generated with `pipreqs path/to/folder`
    │
    └── test_inference.py      	        <- File to test a trained model and save inference results on images.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. </small></p>

