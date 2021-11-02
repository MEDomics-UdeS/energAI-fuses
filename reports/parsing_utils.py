import pandas as pd
from math import log10, floor
from typing import Optional

from reports.constants import AP_DICT


def get_digits_precision(x: float) -> int:
    if x == 0:
        return 0
    else:
        return -int(floor(log10(abs(x))))


def get_latex_exp_name(letter: str, *,
                       phase: Optional[str] = None,
                       hparam: Optional[str] = None) -> str:
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
    file_path = f'{path}latex_phase_{letter}.txt' if path else f'latex_phase_{letter}.txt'

    with open(file_path, 'w') as file:
        file.write(input_str)

    print(f'LaTeX output has been saved to: {file_path}')


def get_scalars_dict(phase: str) -> dict:
    scalars_dict = {}

    for key, value in AP_DICT.items():
        scalars_dict[f'hparams/{phase}/{key}'] = value

    return scalars_dict
