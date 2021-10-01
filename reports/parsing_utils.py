import pandas as pd
from math import log10, floor


def get_digits_precision(x: float) -> int:
    return -int(floor(log10(abs(x))))


def print_latex_header() -> None:
    print('*' * 50)
    print('LaTeX CODE START')
    print('*' * 50)


def print_experiment_name(letter: str, hparam: str) -> None:
    print(f'\\section{{Experiment {letter}: {hparam}}}'.replace('_', '\\_'))


def print_ap_table(letter: str,
                   hparam: str,
                   metric: str,
                   index: int,
                   df: pd.DataFrame) -> None:
    print('\\begin{table}[H]')
    print(df.to_latex(index=False, escape=False))
    print(f'\\caption{{\\label{{tab:table-{index}}}Experiment {letter}: {hparam}: {metric}}}'.replace('_', '\\_'))
    print('\\end{table}\n')


def print_latex_footer() -> None:
    print('*' * 50)
    print('LaTeX CODE END')
    print('*' * 50)
