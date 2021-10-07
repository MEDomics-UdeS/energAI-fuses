"""
File:
    batch_experiment_all.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:

"""

import subprocess as sp

if __name__ == '__main__':
    cmds = ['A1_fasterrcnn_experiment.json',
            'A2_retinanet_experiment.json',
            'A3_detr_experiment.json',
            'B1_size_experiment.json',
            'B2_pretrained_experiment.json',
            'B3_gi_experiment.json',
            'C_sensitivity_experiment_1.json',
            'C_sensitivity_experiment_2.json']

    cmds = [['python', 'experiment_batch.py', '-p'] + [cmd] for cmd in cmds]

    # Loop through each command
    for cmd in cmds:
        # Execute current command
        p = sp.Popen(cmd)

        # Wait until the command finishes before continuing
        p.wait()
