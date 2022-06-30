# EnergAI : Fuse Detection

## Table of Contents
  * [1. Introduction](#1-introduction)
  * [2. Installation](#2-installation)
  * [3. Data](#3-data)
  * [4. Python Module Details](#4-python-module-details)
  * [5. Project Files Organization](#5-project-files-organization)
  * [6. Authors](#6-authors)
  * [7. Acknowledgements](#7-acknowledgements)
  * [8. Statement](#8-statement)

## 1. Introduction
This repository contains the fuse detection code for the EnergAI 
project and the following article published in **IEEE Transactions on
Industral Informatics**: 

[***Electric Power Fuse Identification with Deep Learning***](about:blank)

This project implements a supervised learning PyTorch-based end-to-end object detection pipeline for the purpose of detecting and
classifying fuses in low-voltage electrical installations.

## 2. Installation
Install dependencies in a Python environment:
```
$ pip install -r requirements.txt
```
The use of a CUDA GPU with at least 24 GB of VRAM is required to run the experiments
in this repository. To enable users with lower specifications to train a model and obtain
a trained model quickly, a 'quick run' script has been created, see [here](docs/readme/README_experiment_batch.md) for
details on how to run this script.

## 3. Data

### Datasets

The datasets have been split into two parts: learning
and holdout. The learning dataset is used
to train the neural networks and perform the experiments
required to obtain the final optimized model hyperparameters, 
while the holdout dataset is used to test the final 
trained model on a new dataset of fuse pictures it has never 
encountered before to make sure the model can generalize 
on new data.

**Survey Dataset** (n = 3,189)
- S0001.jpg to S3189.jpg

**Google Images Dataset** (n = 1,116)
- G0001.jpg to G1116.jpg

**TO-DO**
- Zenodo
- Auto download

### Best Model

Best trained model hosted somewhere and downloadable

**TO-DO**
- Zenodo
- Auto download

## 4. Python Module Details

### experiment.py

This file enables users to run a single experiment with the specified parameters using the developed pipeline.

More details on how to specify experiment arguments and an example of basic use can be found [here](docs/readme/README_experiment.md).

### experiment_batch.py

This file enables users to run a batch of different experiments using the 
developed pipeline. 

More details on how to specify fixed or variable experiment parameters for each run,
an example of basic use and how to execute the quick run script can be found [here](docs/readme/README_experiment_batch.md).

### experiment_batch_all.py

This file enables users to run all experiments for phases A, B, C and D from
a single file. 

More details on which experiments are performed when running this script and an
example of basic use can be found [here](docs/readme/README_experiment_batch_all.md).

### inference_test.py

This file enables users to run an inference test on a saved model and show model 
predictions and ground truths boxes on the images.

More details on the arguments and an example of basic use can be found [here](docs/readme/README_inference_test.md).

### reports/test_saved_model.py

This file enables users to test a saved model, either on a subset of the 'learning' or the
'holdout' dataset.

More details on the arguments and an example of basic use can be found [here](docs/readme/README_test_saved_model.md).

### reports/parse_results_[...].py

The following scripts can be used to parse the results generated
when executing phases A, B and C experiments.

More details on the purpose of each parsing script and an example of basic use can be found [here](docs/readme/README_parse_results.md).

### gui.py

A graphical user interface (GUI) has been created to allow users to 
locate and classify fuses in new pictures.

More details on each window, option and button and an example of basic use can be found [here](docs/readme/README_gui.md).

## 5. Project Files Organization
```
├── data                                 <- Main folder containing raw & processed data (images & bounding box annotations)
│   ├── annotations                      <- Bounding box ground truth annotations are automatically downloaded here, as well as resized annotations
│   ├── gui_resized                      <- Resized images for GUI inference
│   ├── inference                        <- Inference test images are saved here
│   ├── raw                              <- Raw images are automatically downloaded here
│   ├── resized                          <- Resized images are saved here
│   └── sample                           <- Sample image and annotations (used to test the GUI without downloading all the data)
├── docs                                 <- Contains markdown README & documentation files
├── logdir                               <- Tensorboard logs are saved here for each run in subfolders
├── reports                              <- Contains results parsing scripts for each phase (see module details for more info)
│   ├── constants.py
│   ├── parse_results_phase_A.py
│   ├── parse_results_phase_B.py
│   ├── parse_results_phase_C.py
│   ├── parse_results_time.py
│   ├── parsing_utils.py
│   └── test_saved_model.py
├── runs                                 <- JSON settings scripts for each experiment (see module details for 'experiment_batch.py' for more info)
│   ├── A1_fasterrcnn_experiment.json
│   ├── A2_retinanet_experiment.json
│   ├── A3_detr_experiment.json
│   ├── B1_size_experiment_2048.json
│   ├── B1_size_experiment.json
│   ├── B2_pretrained_experiment.json
│   ├── B3_gi_experiment.json
│   ├── C_sensitivity_experiment_1.json
│   ├── C_sensitivity_experiment_2.json
│   ├── C_sensitivity_experiment.json
│   ├── D_final_training.json
│   └── quick_run.json
├── saved_models                         <- Trained models are saved here
├── src                                  <- Python source scripts
│   ├── coco                             <- COCO metrics files (copy from 'torchvision' repo)
│   │   ├── coco_eval.py
│   │   ├── coco_utils.py
│   │   └── utils.py
│   ├── data                             <- Custom classes for data, dataset and data loader handling
│   │   ├── DataLoaderManagers           
│   │   │   ├── CustomDataLoaderManager.py
│   │   │   ├── GuiDataLoaderManager.py
│   │   │   └── LearningDataLoaderManager.py
│   │   ├── DatasetManagers              
│   │   │   ├── CustomDatasetManager.py
│   │   │   ├── GuiDatasetManager.py
│   │   │   └── LearningDatasetManager.py
│   │   ├── Datasets
│   │   │   ├── CustomDataset.py
│   │   │   └── FuseDataset.py
│   │   └── SplittingManager.py
│   ├── detr                             <- Scripts imported from DETR repo (Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved)                
│   │   ├── box_ops.py
│   │   ├── criterion.py
│   │   └── matcher.py
│   ├── gui                              <- Graphical User Interface (GUI) scripts
│   │   ├── ImageViewer.py     
│   │   └── modules                       
│   │       ├── AdvancedOptionsWindow.py    
│   │       ├── CSVFileLoader.py
│   │       ├── DeviceSelector.py
│   │       ├── ImageLoader.py
│   │       ├── IOUSlider.py
│   │       ├── ModelLoader.py
│   │       ├── OuputRedirector.py
│   │       ├── ReadOnlyTextBox.py
│   │       ├── ResetButton.py
│   │       └── ScoreSlider.py
│   ├── models                           <- Main model loading & training scripts
│   │   ├── EarlyStopper.py
│   │   ├── models.py
│   │   ├── PipelineManager.py           <- Main script for model training
│   │   └── SummaryWriter.py
│   ├── utils                            <- Various utility functions & constants
│   │   ├── constants.py
│   │   ├── helper_functions.py
│   │   └── reproducibility.py
│   └── visualization                    <- Visualization scripts for inference
│       └── inference.py
├── experiment.py                        <- Script to perform a single experiment (see module details README for more info)
├── experiment_batch.py                  <- Script to perform a batch of multiple experiments with different settings (see module details README for more info)
├── experiment_batch_all.py              <- Script to run all experiments for phases A, B, C and D (see module details README for more info)
├── gui.py                               <- Script to run the Graphical User Interface (GUI) (see module details README for more info)
├── inference_test.py                    <- Script to perform an inference test using a saved model (see module details README for more info)
├── LICENSE                              <- GPLv3 license details
├── README.md                            <- File you are currently reading
└── requirements.txt                     <- Python packages & versions requirements
```

## 6. Authors
* [Simon Giard-Leroux](https://github.com/sgiardl) (Université de Sherbrooke / CIMA+)
* [Guillaume Cléroux](https://github.com/gcleroux) (Université de Sherbrooke)
* [Shreyas Sunil Kulkarni](https://github.com/Kuyas) (Birla Institute of Technology and Science, Pilani / Amazon)
* [François Bouffard](https://www.mcgill.ca/ece/francois-bouffard) (McGill University)
* [Martin Vallières](https://github.com/mvallieres) (Université de Sherbrooke)

## 7. Acknowledgements

The authors would like to thank all partner organizations 
that were involved during this project: CIMA+, HEXACODE Solutions, 
Université de Sherbrooke and Université McGill, as well as the 
organizations that supplied the funding for this project: 
CIMA+, HEXACODE Solutions, InnovÉÉ, Mitacs and the 
Natural Sciences and Engineering Research Council of Canada 
(NSERC). Martin Vallières also acknowledges funding from the 
Canada CIFAR AI Chairs Program.

## 8. Statement

This project's code and data are published under the GPLv3 license.

```
  Energ-AI : Fuse Detection
    Copyright (C) 2022, CIMA+, HEXACODE Solutions, Université de Sherbrooke, 
    McGill University

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
```

The following scripts have been imported from other repositories
which are subject to their own specific copyright and licenses:
- **src/coco/coco_eval.py, src/coco/coco_utils.py, src/coco/utils.py**
  - Copyright Holder: Facebook, Inc. and its affiliates
  - Source: https://github.com/pytorch/vision/tree/main/references/detection
  - License: BSD-3-Clause: https://opensource.org/licenses/BSD-3-Clause
- **src/detr/box_ops.py, src/detr/criterion.py** (modified)**, src/detr/matcher.py**
  - Copyright Holder: Facebook, Inc. and its affiliates
  - Source: https://github.com/facebookresearch/detr
  - License: Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0
- **src/models/EarlyStopper.py** (modified)
  - Copyright Holder: Stefano Nardo
  - Source: https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
  - License: MIT: https://opensource.org/licenses/MIT

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. </small></p>
