"""
File:
    batch_experiment.py

Authors:
    - Simon Giard-Leroux
    - Shreyas Sunil Kulkarni

Description:
    Allows to perform multiple src/saved_models/experiment.py runs with different parameters
"""

import subprocess as sp
from datetime import datetime
from itertools import product

from src.utils.helper_functions import env_tests

if __name__ == '__main__':
    # Record start time
    start = datetime.now()

    # Run environment tests
    env_tests()

    hparams = {
        '-da': ['0.1', '0.25', '0.5'],
        '-lr': ['3e-4', '3e-5', '3e-6'],
        '-wd': ['3e-2', '3e-3', '3e-4']
    }

    # Declare list of commands to be executed
    cmds = list(list(cmd) for cmd in product(*hparams.values()))

    for i in range(len(cmds)):
        for j in range(0, len(hparams) + 2, 2):
            cmds[i].insert(j, list(hparams)[j // 2])

    cmds = [['python', 'experiment.py', '-mo', 'retinanet_resnet50_fpn', '-b', '20'] + cmd for cmd in cmds]

    # Loop through each command
    for i, cmd in enumerate(cmds, start=1):
        # Print experiment details
        print('-' * 100)
        print(f'Experiment {i}/{len(cmds)}:\t\t\t\t{" ".join(cmd)}')

        # Execute current command
        p = sp.Popen(cmd)

        # Wait until the command finishes before continuing
        p.wait()

    # Print time taken for all experiments
    print('-' * 100)
    print(f'Total time for all experiments:\t\t{str(datetime.now() - start).split(".")[0]}')
