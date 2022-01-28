"""
File:
    reports/parsing_utils.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Parsing utility functions
"""

import pandas as pd
from math import log10, floor
from typing import Optional

from reports.constants import AP_DICT


def get_digits_precision(x: float) -> int:
    """Function to get number of digits to round at for significant digit purposes

    Args:
        x(float): floating point value

    Returns:
        int: digits of precision to round at

    """
    if x == 0:
        return 0
    else:
        return -int(floor(log10(abs(x))))


def get_latex_exp_name(letter: str, *,
                       phase: Optional[str] = None,
                       hparam: Optional[str] = None) -> str:
    """Function to generate LaTeX code for an experiment name section title

    Args:
        letter(str): experiment letter for current experiment
        phase(Optional[str], optional): phase for current experiment (Validation or Testing) (Default value = None)
        hparam(Optional[str], optional): hyperparameter for current experiment (Default value = None)

    Returns:
        str: LaTeX code

    """
    title = f'Experiment {letter}'

    if phase:
        title += f': {phase}'

    if hparam:
        title += f': {hparam}'.replace('_', '\\_')

    return f'\\section{{{title}}}\n'


def get_latex_ap_table(df: pd.DataFrame, *,
                       index: int,
                       letter: str,
                       phase: Optional[str] = None,
                       hparam: Optional[str] = None,
                       metric: Optional[str] = None) -> str:
    """Function to get LaTeX code for an AP results table

    Args:
        df(pd.DataFrame): pandas DataFrame containing AP results
        index(int): table index
        letter(str): experiment letter
        phase(Optional[str], optional): experiment phase (Validation or Testing) (Default value = None)
        hparam(Optional[str], optional): hyperparameter for current experiment (Default value = None)
        metric(Optional[str], optional): metric being evaluated for current experiment (Default value = None)

    Returns:
        str: LaTeX code

    """
    title = f'Experiment {letter}'

    if phase:
        title += f': {phase}'

    if hparam:
        title += f': {hparam}'.replace('_', '\\_')

    if metric:
        title += f': {metric}'.replace('_', '\\textsubscript')

    output_str = '\\begin{table}[H]\n\\centerline{\n'
    output_str += df.to_latex(index=False, escape=False)
    output_str += f'}}\n\\caption{{\\label{{tab:table-{index}}}{title}}}\n\\end{{table}}\n'

    return output_str


def save_latex(input_str: str,
               letter: str,
               path: Optional[str] = None) -> None:
    """Function to save LaTeX code for a specific experiment to a file

    Args:
        input_str(str): LaTeX code string
        letter(str): experiment letter
        path(Optional[str], optional): saved file path (Default value = None)

    """
    file_path = f'{path}latex_phase_{letter}.txt' if path else f'latex_phase_{letter}.txt'

    with open(file_path, 'w') as file:
        file.write(input_str)

    print(f'LaTeX output has been saved to: {file_path}')


def get_scalars_dict(phase: str) -> dict:
    """Function to create a scalar dictionary

    Args:
      phase(str): experiment phase

    Returns:
        dict: scalar dictionary

    """
    scalars_dict = {}

    for key, value in AP_DICT.items():
        scalars_dict[f'hparams/{phase}/{key}'] = value

    return scalars_dict
