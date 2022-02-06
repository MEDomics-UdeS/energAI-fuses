## Module Details: experiment_batch_all.py

### Description

This file enables users to run all experiments for phases A, B, C and D from
a single file. It will run the following `experiment_batch.py` calls 
sequentially:
- A1_fasterrcnn_experiment.json
- A2_retinanet_experiment.json
- A3_detr_experiment.json
- B1_size_experiment.json
- B1_size_experiment_2048.json
- B2_pretrained_experiment.json
- B3_gi_experiment.json
- C_sensitivity_experiment.json
- D_final_training.json

### Examples of basic use:

To run all experiments in one line:
```
python experiment_batch_all.py
```