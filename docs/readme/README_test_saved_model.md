## Module Details: reports/test_saved_model.py

### Description

This file enables users to test a saved model, either on a subset of the 'learning' or the
'holdout' dataset.

### Arguments

| Short 	     | Long              	   | Type    	  | Default     | Choices                  | Description                                 |
|-------------|-----------------------|------------|-------------|--------------------------|---------------------------------------------|
| `-ds`       | `--dataset`           | str     	  | 	`learning` | `learning`<br/>`holdout` | Dataset to use for inference test           |
| `-letter`   | `--experiment_letter` | str     	  | 	`A`        | `A`<br/>`D`              | Experiment letter                           |
| `-mp`       | `--models_path`       | str     	  | 	           |                          | Directory containing the models             |
| `-sv_latex` | `--save_latex`        | bool     	 | `True`	     |                          | Specify whether to save LaTeX output or not |
| `-sv_json`  | `--save_json`         | bool     	 | `True`	     |                          | Specify whether to save JSON output or not  |

``-h``, ``--help``
show this help message and exit

### Example of basic use

To test a saved model:
```
python reports/test_saved_model.py --models_path saved_models/
```