import sys
import os
import subprocess as sp
import time

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
    # Help Call
    # sp.run(['python3', 'detect.py', '-h'])
    epoch_list = [10]
    batch_list = [10]
    mp = [0]
    g = [0]
    count = 0
    cmd_list = []
    for i in epoch_list:
        for j in batch_list:
            for k in mp:
                for l in g:
                    l1 = ['python3', 'detect.py', '--train', '--test', '--verbose', '-e', str(i), '-b', str(j)]
                    if k:
                        l1.append('-mp')
                    if l:
                        l1.append('-g')
                    count += 1
                    cmd_list.append(l1)

    # print(cmd_list[-1],count)
    # exit()
    # cmds = [['python3', 'detect.py', '--train', '--epochs', '1', '--batch', '10', '--downsample', '500', '-g',
    #          '--verbose'],
    #        ['python3', 'detect.py', '--train', '--epochs', '1', '--batch', '10', '--downsample', '500', '--verbose']]
    cmds = [['python3', 'fuses/src/models/detect.py', '--train', '--epochs', '5', '--batch', '10',
             '--s', '1000', '-g', '1', '--verbose']]
    start = time.time()
    for cmd in cmds:
        print(cmd)
        p = sp.Popen(cmd)
        p.wait()

    print("Time Taken (minutes): ", round((time.time() - start) / 60, 2))
