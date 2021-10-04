import os
import json
import pandas as pd

from parsing_utils import get_latex_ap_table, get_latex_exp_name, get_digits_precision, save_latex


def parse_results(json_path: str,
                  round_to_1_sign_digit: bool = False,
                  num_decimals: int = 4) -> pd.DataFrame:
    with open(json_path) as json_file:
        results_dict = json.load(json_file)

    columns = ['Seed']

    scalar_dict = {
        'AP @ [IoU=0.50:0.95 | area=all | maxDets=100]': 'AP',
        'AP @ [IoU=0.50 | area=all | maxDets=100]': 'AP_{50}',
        'AP @ [IoU=0.75 | area=all | maxDets=100]': 'AP_{75}',
        'AP @ [IoU=0.50:0.95 | area=small | maxDets=100]': 'AP_{S}',
        'AP @ [IoU=0.50:0.95 | area=medium | maxDets=100]': 'AP_{M}',
        'AP @ [IoU=0.50:0.95 | area=large | maxDets=100]': 'AP_{L}'
    }

    for value in scalar_dict.values():
        columns.append(value)

    df = pd.DataFrame(columns=columns)

    for run_dict in results_dict.values():
        row = {columns[0]: str(run_dict["args"]["random_seed"])}

        for scalar_long, scalar_short in scalar_dict.items():
            row[scalar_short] = run_dict["results"][scalar_long]

        df = df.append(row, ignore_index=True)

    row = {columns[0]: 'mean ± std'}

    for column in columns[1:]:
        precision = get_digits_precision(df[column].std())
        format_str = f'{{:.{precision}f}}'

        row[column] = f'{format_str.format(round(df[column].mean(), precision))} ± ' \
                      f'{format_str.format(round(df[column].std(), precision))}'

        if round_to_1_sign_digit:
            df[column] = df[column].round(precision)
            df[column] = df[column].apply(format_str.format)
        else:
            format_str = f'{{:.{num_decimals}f}}'
            df[column] = df[column].round(num_decimals)
            df[column] = df[column].apply(format_str.format)

    df = df.append(row, ignore_index=True)

    return df


if __name__ == '__main__':
    os.chdir('..')
    json_path = f'{os.getcwd()}/D_results.json'

    experiment_letter = 'D'

    results_df = parse_results(json_path)

    output_str = get_latex_exp_name(experiment_letter)
    output_str += get_latex_ap_table(results_df, 99, experiment_letter)

    save_latex(output_str, letter=experiment_letter, path='reports/')
