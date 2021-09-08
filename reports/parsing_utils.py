import pandas as pd


def print_ap_table(df: pd.DataFrame) -> None:
    print('*' * 50)
    print('LaTeX CODE START')
    print('*' * 50)
    print(df.to_latex(index=False, escape=False))
    print('*' * 50)
    print('LaTeX CODE END')
    print('*' * 50)
