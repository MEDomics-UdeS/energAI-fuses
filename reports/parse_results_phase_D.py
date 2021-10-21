import json
import pandas as pd

from parsing_utils import get_latex_ap_table, get_latex_exp_name, get_digits_precision, save_latex


def parse_results(json_path: str,
                  round_to_1_sign_digit: bool = False,
                  num_decimals: int = 4) -> pd.DataFrame:
    with open(json_path) as json_file:
        results_dict = json.load(json_file)

    columns_all_seeds = ['Seed']
    columns_std = ['Metric', 'Mean', 'Std (Absolute)', 'Std (Relative)']

    scalar_dict = {
        'AP @ [IoU=0.50:0.95 | area=all | maxDets=100]': 'AP',
        'AP @ [IoU=0.50 | area=all | maxDets=100]': 'AP_{50}',
        'AP @ [IoU=0.75 | area=all | maxDets=100]': 'AP_{75}',
        'AP @ [IoU=0.50:0.95 | area=small | maxDets=100]': 'AP_{S}',
        'AP @ [IoU=0.50:0.95 | area=medium | maxDets=100]': 'AP_{M}',
        'AP @ [IoU=0.50:0.95 | area=large | maxDets=100]': 'AP_{L}'
    }

    metrics_list = []

    for value in scalar_dict.values():
        metrics_list.append(value)

    columns_all_seeds = columns_all_seeds + metrics_list

    df_all_seeds = pd.DataFrame(columns=columns_all_seeds)
    df_std = pd.DataFrame(columns=columns_std)

    for metric in metrics_list:
        df_std = df_std.append({columns_std[0]: metric}, ignore_index=True)

    for run_dict in results_dict.values():
        row = {columns_all_seeds[0]: str(run_dict["args"]["random_seed"])}

        for scalar_long, scalar_short in scalar_dict.items():
            row[scalar_short] = run_dict["results"][scalar_long]

        df_all_seeds = df_all_seeds.append(row, ignore_index=True)

    row = {columns_all_seeds[0]: 'mean ± std'}

    for column in columns_all_seeds[1:]:
        mean = df_all_seeds[column].mean()
        std = df_all_seeds[column].std()

        precision = get_digits_precision(std)
        std_round = round(std, precision)
        precision = get_digits_precision(std_round)
        std_round = round(std, precision)

        mean_round = round(mean, precision)

        format_str = f'{{:.{precision}f}}'

        row[column] = f'{format_str.format(mean_round)} ± ' \
                      f'{format_str.format(std_round)}'

        if round_to_1_sign_digit:
            df_all_seeds[column] = df_all_seeds[column].round(precision)
            df_all_seeds[column] = df_all_seeds[column].apply(format_str.format)
        else:
            df_all_seeds[column] = df_all_seeds[column].round(num_decimals)
            df_all_seeds[column] = df_all_seeds[column].apply(f'{{:.{num_decimals}f}}'.format)

        df_std.loc[df_std[columns_std[0]] == column, columns_std[1]] = format_str.format(mean_round)
        df_std.loc[df_std[columns_std[0]] == column, columns_std[2]] = format_str.format(std_round)
        df_std.loc[df_std[columns_std[0]] == column, columns_std[3]] = '{:.2%}'.format(std / mean).replace('%', '\\%')

    df_all_seeds = df_all_seeds.append(row, ignore_index=True)

    return df_all_seeds, df_std


if __name__ == '__main__':
    experiment_letter = 'D'

    df_all_seeds, df_std = parse_results('../json/D_results.json')

    output_str = get_latex_exp_name(experiment_letter)
    output_str += get_latex_ap_table(df_all_seeds, 99, experiment_letter)
    output_str += get_latex_ap_table(df_std, 100, experiment_letter)

    save_latex(output_str, letter=experiment_letter, path='../reports/')
