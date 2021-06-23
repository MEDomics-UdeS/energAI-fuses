"""
File:
    batch_experiment.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
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

    da_list = ['0.1', '0.25', '0.5']
    lr_list = ['3e-4', '3e-5', '3e-6']
    wd_list = ['3e-2', '3e-3', '3e-4']

    # Declare list of commands to be executed
    cmds = list(list(cmd) for cmd in product(da_list, lr_list, wd_list))

    for i in range(len(cmds)):
        cmds[i].insert(0, '-da')
        cmds[i].insert(2, '-lr')
        cmds[i].insert(4, '-wd')

    cmds = [['python', 'experiment.py', '-mo', 'detr', '-b', '14'] + cmd for cmd in cmds]

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
