import sys
import subprocess as sp
from datetime import datetime

REQUIRED_PYTHON = "python3"


def main():
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(
            REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version))
    else:
        print(">>> Development environment passes all tests!")


if __name__ == '__main__':
    main()

    cmds = [
        ['python', 'detect.py', '--data', 'resized', '--epochs', '5'],
    ]

    start = datetime.now()

    for i, cmd in enumerate(cmds, start=1):
        print('-' * 100)
        print(f'Experiment {i}/{len(cmds)}:\t\t\t\t{" ".join(cmd)}')
        p = sp.Popen(cmd)
        p.wait()

    print('-' * 100)
    print(f'Total time for all experiments:\t\t{str(datetime.now() - start).split(".")[0]}')
