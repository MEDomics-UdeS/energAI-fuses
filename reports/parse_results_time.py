"""
File:
    reports/parse_results_time.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Parsing script for experiment times for all phases
"""

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import pandas as pd
from datetime import timedelta
from tqdm import tqdm

from constants import PATH
from parsing_utils import get_latex_exp_name, get_latex_ap_table, save_latex


def parse_time(path: str,
               min_time_init: float = 1e30,
               max_time_init: float = 0) -> pd.DataFrame:
    """Function to parse experiment times for tensorboard runs

    Args:
        path(str): path to tensorboard runs
        min_time_init(float, optional): minimum time found (Default value = 1e30)
        max_time_init(float, optional): maximum time found (Default value = 0)

    Returns: pandas DataFrame containing experiment times

    """
    columns = ['Experiment Phase', 'Total Execution Time']
    df = pd.DataFrame(columns=columns)

    for phase in tqdm(os.listdir(path), desc=f'Parsing...'):
        total_time = 0

        phase_path = os.path.join(path, phase, 'logdir')

        experiments = [exp for exp in os.listdir(phase_path) if exp != '.gitkeep']

        for experiment in experiments:
            min_time = min_time_init
            max_time = max_time_init

            event_acc = EventAccumulator(os.path.join(phase_path, experiment))
            event_acc.Reload()

            for scalar in event_acc.Tags()['scalars']:
                time, _, _ = zip(*event_acc.Scalars(scalar))

                if min(time) < min_time:
                    min_time = min(time)

                if max(time) > max_time:
                    max_time = max(time)

            if min_time != min_time_init and max_time != max_time_init:
                total_time += max_time - min_time

        df = df.append({columns[0]: phase,
                        columns[1]: str(timedelta(seconds=total_time)).split('.')[0]},
                       ignore_index=True)

    df = df.sort_values(columns[0])

    return df


if __name__ == '__main__':
    time_df = parse_time(PATH)

    output_str = get_latex_exp_name(letter='A-B-C',
                                    hparam='Execution Time')

    output_str += get_latex_ap_table(time_df,
                                     index=0,
                                     letter='A-B-C',
                                     hparam='Execution Time')

    save_latex(output_str, letter='A-B-C')
