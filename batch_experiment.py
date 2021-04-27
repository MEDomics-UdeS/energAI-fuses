import subprocess as sp
from datetime import datetime

from src.utils.helper_functions import env_tests

if __name__ == '__main__':
    env_tests()

    cmds = [
        ['python', 'src/models/experiment.py', '--epochs', '10',
         '--model', 'fasterrcnn_mobilenet_v3_large_320_fpn'],
    ]

    start = datetime.now()

    for i, cmd in enumerate(cmds, start=1):
        print('-' * 100)
        print(f'Experiment {i}/{len(cmds)}:\t\t\t\t{" ".join(cmd)}')
        p = sp.Popen(cmd)
        p.wait()

    print('-' * 100)
    print(f'Total time for all experiments:\t\t{str(datetime.now() - start).split(".")[0]}')
