## Module Details: experiment.py

### Description

This file enables users to run a single experiment with the specified parameters using the developed pipeline.

### Arguments

| Short 	      | Long              	               | Type   | Default           	                | Choices                                                                       	                                                                             | Description                                                                   	                                     |
|--------------|-----------------------------------|--------|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `-nw`    	   | `--num_workers`           	       | int    | `multiprocessing.cpu_count()`    	 | 	                                                                                                                                                           | Number of CPU threads for multiprocessing purposes                                                                  |
| `-e`    	    | `--epochs`          	             | int    | `200`                	             | 	                                                                                                                                                           | Number of epochs                        	                                                                           |
| `-b`   	     | `--batch`  	                      | int    | `1` 	                              | 	                                                                                                                                                           | Batch size                                                  	                                                       |
| `-vs`   	    | `--validation_size`               | float  | `0.1`               	              | 	                                                                                                                                                           | Size of validation set (float as proportion of dataset) 	                                                           |
| `-ts`   	    | `--test_size`           	         | float  | `0.1`               	              | 	                                                                                                                                                           | Size of test set (float as proportion of dataset)	                                                                  |
| `-s`    	    | `--image_size`           	        | int    | `1024`    	                        | 	                                                                                                                                                           | Resize the images to size * size                                                                                    |
| `-da`    	   | `--data_aug`         	            | float  | `0.25`         	                   | 	                                                                                                                                                           | Value of data augmentation for training dataset                                                                     |
| `-norm`   	  | `--normalize`        	            | str    | `precalculated`                    | `'precalculated'`<br>`'calculated'`<br>`'disabled'`	 	                                                                                                      | Normalize the training dataset by mean & std using precalculated values, calculated values or disabled            	 |
| `-lr`   	    | `--learning_rate`                 | float  | `0.0003`                           | 	                                                                                                                                                           | Learning rate for Adam optimizer                                                                                    |
| `-wd`   	    | `--weight_decay`                  | float  | `0.00003`                          | 	                                                                                                                                                           | Weight decay (L2 penalty) for Adam optimizer                                                                        |
| `-mo`   	    | `--model`                         | str    | `'fasterrcnn_resnet50_fpn'`        | `'fasterrcnn_resnet50_fpn'`<br>`'fasterrcnn_mobilenet_v3_large_fpn'`<br>`'fasterrcnn_mobilenet_v3_large_320_fpn'`<br>`'retinanet_resnet50_fpn'`<br>`'detr'` | Object detection model                                                                                              |
| `-esp`   	   | `--es_patience` 	                 | int    | `None`            	                | 	                                                                                                                                                           | Early stopping patience (number of epochs without improvement)                                                      |
| `-esd`       | `--es_delta`        	             | float  | `0`                 	              | 	                                                                                                                                                           | Early stopping delta (tolerance to evaluate improvement)                     	                                      |
| `-mp`    	   | `--mixed_precision`  	            | bool   | `False`                            | 	                                                                                                                                                           | Boolean to use mixed precision                  	                                                                   |
| `-g`   	     | `--gradient_accumulation`         | int    | `1`            	                   | 	                                                                                                                                                           | Gradient accumulation size (1 : no gradient accumulation)                                                           |
| `-gc`   	    | `--gradient_clip`    	            | float  | `5`                 	              | 	                                                                                                                                                           | Gradient clipping value                                                          	                                  |
| `-ss`   	    | `--seed_split`      	             | int 	  | `54288`            	               | 	                                                                                                                                                           | Random seed for training, validation and test splitting           	                                                 |
| `-si`   	    | `--seed_init`      	              | int 	  | `54288`            	               | 	                                                                                                                                                           | Random seed for RNG initialization 	                                                                                |
| `-dt`   	    | `--deterministic`        	        | bool 	 | `False`             	              | 	                                                                                                                                                           | Boolean to force deterministic behavior           	                                                                 |
| `-no-pt`   	 | `--no_pretrained`        	        | bool 	 | `False`             	              | 	                                                                                                                                                           | If specified, the loaded model will not be pretrained           	                                                   |
| `-no-sv`   	 | `--no_save_model`        	        | bool 	 | `False`             	              | 	                                                                                                                                                           | If specified, the trained model will not be saved           	                                                       |
| `-no-gi`   	 | `--no_google_images`        	     | bool 	 | `False`             	              | 	                                                                                                                                                           | If specified, the Google Images photos will be excluded from the training subset          	                         |
| `-ltm`   	   | `--log_training_metrics`        	 | bool 	 | `False`             	              | 	                                                                                                                                                           | If specified, the AP and AR metrics will be calculated and logged for training set          	                       |
| `-sl`   	    | `--save_last`        	            | bool 	 | `False`             	              | 	                                                                                                                                                           | Specify whether to save/use for inference testing the last model, otherwise the best model will be used           	 |
| `-kcv`   	   | `--k_cross_valid`        	        | int 	  | `1`             	                  | 	                                                                                                                                                           | Number of folds for k-fold cross validation (1 = no k-fold cross validation)           	                            |

If the model architecture chosen is `detr`, the following arguments are included.

| Long              	            | Type  | Default           	 | Description                                                                   	 |
|--------------------------------|-------|---------------------|---------------------------------------------------------------------------------|
| `--set_cost_class`           	 | float | `1`    	            | Class coefficient in the matching cost                                          |
| `--set_cost_bbox`           	  | float | `5`    	            | L1 box coefficient in the matching cost                                         |
| `--set_cost_giou`           	  | float | `2`    	            | giou box coefficient in the matching cost                                       |
| `--eos_coef`           	       | float | `0.1`    	          | Relative classification weight of the no-object class                           |


``-h``, ``--help``
show this help message and exit

### Examples of basic use:

To run a single experiment:
```
python experiment.py --epochs 3
```

To view log runs and hyperparameters in tensorboard:
```
tensorboard --logdir=logdir
```