import os
import json
import pandas as pd

from parsing_utils import get_digits_precision, print_latex_header, print_latex_footer


def parse_results(json_path: str) -> pd.DataFrame:
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
        format_str = '{:.' + str(precision) + 'f}'

        row[column] = f'{format_str.format(round(df[column].mean(), precision))} ± ' \
                      f'{format_str.format(round(df[column].std(), precision))}'

        df[column] = df[column].round(precision)

        df[column] = df[column].apply(format_str.format)

    df = df.append(row, ignore_index=True)

    return df


if __name__ == '__main__':
    os.chdir('..')
    json_path = f'{os.getcwd()}/D_results.json'

    results_df = parse_results(json_path)

    print_latex_header()
    print(results_df.to_latex(index=False, escape=False))
    print_latex_footer()
