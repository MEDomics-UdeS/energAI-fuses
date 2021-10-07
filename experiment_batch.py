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

from itertools import product
import subprocess as sp
from datetime import datetime
import json
import argparse

from src.utils.helper_functions import env_tests


if __name__ == '__main__':
    # Record start time
    start = datetime.now()

    # Run environment tests
    env_tests()

    # Declare argument parser
    parser = argparse.ArgumentParser(description='Processing inputs')

    # Number of workers argument
    parser.add_argument('-p', '--path', action='store', type=str, required=True,
                        help='Experiment settings .json file path')

    args = parser.parse_args()

    try:
        with open(args.path) as f_obj:
            json_dict = json.load(f_obj)
            fixed_params = json_dict["fixed"]
            variable_params = json_dict["variable"]
            f_obj.close()
    except FileNotFoundError:
        print("No valid JSON file entered for experiments. Please verify in project directory.")
    else:
        # Declare list of commands to be executed
        if variable_params:
            cmds = list(list(cmd) for cmd in product(*variable_params.values()))

            if len(variable_params) > 1:
                [cmds[i].insert(j, list(variable_params)[j // 2]) for j in range(0, len(variable_params) + 2, 2) for i in range(len(cmds))]
            else:
                [cmds[i].insert(0, list(variable_params)[0]) for i in range(len(cmds))]

            if fixed_params:
                [cmd.extend((key, value)) for key, value in fixed_params.items() for cmd in cmds]
        else:
            cmds = [[]]

            if fixed_params:
                [cmd.extend((key, value)) for key, value in fixed_params.items() for cmd in cmds]

        cmds = [['python', 'experiment.py'] + cmd for cmd in cmds]

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
