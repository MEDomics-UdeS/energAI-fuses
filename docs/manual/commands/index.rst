===========
CLI Options
===========

Module Details: experiment.py & experiment_batch.py
===================================================

Description
-----------

There are currently two ways to run the application. You can launch a single experiment using 
the file experiment.py or you can run a batch of experiments by running experiement_batch.py 
and feeding the script a JSON file that follows the templatethat we provide `here`_.
The experiement_batch.py script simply calls repeatedly experiment.py with every combinations of arguments
found in the JSON file. You can see the table of every possible arguments below.

.. list-table:: Arguments
    :widths: 6 20 10 50 50 100
    :header-rows: 1

    * - Short
      - Long
      - Type
      - Default
      - Choices
      - Description
    * - -s
      - --size
      - int
      - 1024
      -
      - Resize the images to size * size
    * - -da
      - --data_aug
      - float
      - 0.25
      -
      - Value of data augmentation for training dataset
    * - -vs
      - --validation_size
      - float
      - 0.1
      -
      - Size of validation set (float as proportion of dataset)
    * - -ts
      - --test_size	
      - float
      - 0.1
      -
      - Size of test set (float as proportion of dataset)
    * - -e
      - --epochs
      - int
      - 1
      -
      - Number of epochs
    * - -b
      - --batch
      - int
      - 1
      -
      - Batch size
    * - -esp
      - --es_patience
      - int
      - None
      -
      - Early stopping patience (number of epochs without improvement)
    * - -esd
      - --es_delta	
      - float
      - 0
      -
      - Early stopping delta (tolerance to evaluate improvement)
    * - -mp
      - --mixed_precision
      - bool
      - False
      -
      - Boolean to use mixed precision
    * - -g
      - --gradient_accumulation
      - int
      - 1
      - 
      - Gradient accumulation size (1 : no gradient accumulation)
    * - -gc
      - --gradient_clip
      - float
      - 5
      - 
      - Gradient clipping value
    * - -rs
      - --random_seed
      - int
      - 42
      -
      - Random seed, only set if deterministic is set to True
    * - -dt
      - --deterministic
      - bool
      - False
      -
      - Boolean to compute mean & standard deviation RGB normalization values
    * - -ms
      - --mean_std
      - bool
      - False
      -
      - Boolean to compute mean & standard deviation RGB normalization values
    * - -iou
      - --iou_threshold
      - float
      - 0.5
      -
      - Intersection-over-union (IOU) threshold to filter bounding box predictions
    * - -lr
      - --learning_rate
      - float
      - 0.0003
      -
      - Learning rate for Adam optimizer
    * - -wd
      - --weight_decay
      - float
      - 0
      -
      - Weight decay (L2 penalty) for Adam optimizer
    * - -mo
      - --model
      - str
      - 'fasterrcnn_resnet50_fpn'
      - fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn, detr
      - Object detection model
    * - -pt
      - --pretrained
      - bool
      - True
      -
      - Boolean to specify to load a pretrained model
    * - -sv
      - --save_model
      - bool
      - True
      -
      - Boolean to specify to save the trained model


-h, --help show this help message and exit

Examples of basic use
---------------------

To run a single experiment:

.. code-block:: bash

    python3 experiment.py --epochs 3

To run a batch of multiple experiments with a JSON file specified in batch_experiment.py:

.. code-block:: bash

    python3 experiment_batch.py

To view log runs and hyperparameters in tensorboard:

.. code-block:: bash

    tensorboard --logdir=logdir


Module Details: test_inference.py
=================================

Description
-----------

This file enables users to run an inference test on a saved 
model and show model predictions and ground truths boxes on the images.


.. list-table:: Arguments
    :widths: 6 20 10 50 100
    :header-rows: 1

    * - Short
      - Long
      - Type
      - Default
      - Description
    * - -mfn
      - --model_file_name
      - str
      -
      - File name of the saved model to load
    * - -ms
      - --mean_std	
      - bool
      - False
      - Boolean to compute mean & standard deviation RGB normalization values
    * - -b
      - --batch
      - int
      - 1
      - Batch size
    * - -iou
      - --iou_threshold	
      - float
      - 0.5
      - Intersection-over-union (IOU) threshold to filter bounding box predictions

-h, --help show this help message and exit

Examples of basic use
---------------------

To plot the active learning curve for a particular experiments batch:

.. code-block:: bash

    python3 inference_test.py --model_file_name <YOUR_FILENAME_HERE>

.. _`here`:

JSON template for experiement_batch.py
======================================

When configuring your JSON file for experiement_batch.py, two fields need 
to be set up in the json file. 

"fixed": Represents the shared arguments between every experiments

"variable": Every arguments that you want to test. The variables will be applied 
to every experiments with a dot product.

.. code-block:: json

    {
    "fixed":{
        "--model": "detr",
        "--batch": "14",
        "--data_aug": "0.1"
        },
    "variable":{
        "--learning_rate": ["3e-5", "5e-5"],
        "--weight_decay": ["3e-4"]
        }
    }