## Module Details: inference_test.py

### Description

This file enables users to run an inference test on a saved model and show model 
predictions and ground truths boxes on the images.

### Arguments

| Short 	      | Long              	            | Type    	 | Default         | Choices                                                | Description                                                                                                         |
|--------------|--------------------------------|-----------|-----------------|--------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `-mfn`       | `--model_file_name`            | str     	 | 	               |                                                        | File name of the saved model to load                                                                                |
| `-norm`   	  | `--normalize`        	         | str       | `precalculated` | `'precalculated'`<br>`'calculated'`<br>`'disabled'`	 	 | Normalize the training dataset by mean & std using precalculated values, calculated values or disabled            	 |
| `-b`   	     | `--batch`         	            | int     	 | `1`             |                                                        | Batch size                                                                                                          |
| `-iou`   	   | `--iou_threshold`              | float     | `0.5`           |                                                        | Intersection-over-union (IOU) threshold to filter bounding box predictions                                          |
| `-sc`   	    | `--score_threshold`            | float     | `0.5`           |                                                        | Objectness score threshold to filter bounding box predictions                                                       |
| `-no-gi`   	 | `--no_google_images`        	  | bool 	    | `False`         | 	                                                      | If specified, the Google Images photos will be excluded from the training subset          	                         |
| `-gui`   	   | `--with_gui`        	          | bool 	    | `False`         | 	                                                      | If specified, the inference results will be shown in the GUI application          	                                 |
| `-img`   	   | `--image_path`        	        | str 	     |                 | 	                                                      | Image directory to use for inference test         	                                                                 |
| `-d`   	     | `--device`        	            | str 	     | `cpu`           | `cpu`<br/>`cuda`	                                      | Select the device for inference         	                                                                           |
| `-gtf`   	   | `--ground_truth_file`        	 | str 	     |                 | 	                                                      | Select a CSV file for ground truth drawing on images         	                                                      |
``-h``, ``--help``
show this help message and exit

### Examples of basic use:

To perform inference test on a saved model:
```
python test_inference.py --model_file_name 2021-04-27_16-34-27
```****