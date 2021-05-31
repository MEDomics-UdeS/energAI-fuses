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

from src.utils.helper_functions import env_tests

if __name__ == '__main__':
    # Record start time
    start = datetime.now()

    # Run environment tests
    env_tests()

    # Declare list of commands to be executed
    cmds = [
        ['python', 'src/models/experiment.py', '--epochs', '2'],
    ]

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
