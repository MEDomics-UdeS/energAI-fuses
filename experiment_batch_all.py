"""
File:
    batch_experiment_all.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Allows to perform all experiments for phases A, B, C and D from a single script.
"""

import subprocess as sp

if __name__ == '__main__':
    cmds = ['A1_fasterrcnn_experiment.json',
            'A2_retinanet_experiment.json',
            'A3_detr_experiment.json',
            'B1_size_experiment.json',
            'B1_size_experiment_2048.json',
            'B2_pretrained_experiment.json',
            'B3_gi_experiment.json',
            'C_sensitivity_experiment.json',
            'D_final_training.json']

    cmds = [['python', 'experiment_batch.py', '-p'] + ['runs/' + cmd] for cmd in cmds]

    # Loop through each command
    for cmd in cmds:
        # Execute current command
        p = sp.Popen(cmd)

        # Wait until the command finishes before continuing
        p.wait()
