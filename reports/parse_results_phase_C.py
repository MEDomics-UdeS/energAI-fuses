import os
import pandas as pd
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from parsing_utils import get_latex_ap_table, get_latex_exp_name, get_digits_precision, save_latex, get_scalars_dict
from constants import PATH_C, AP_DICT, PHASES_LIST


def parse_results(saved_models_path: str,
                  log_path: str,
                  scalars_dict: dict,
                  round_to_1_sign_digit: bool = False,
                  num_decimals: int = 4) -> pd.DataFrame:
    columns_all_seeds = ['Seed']
    columns_std = ['Metric', 'Mean', 'Std (Absolute)', 'Std (Relative)']

    metrics_list = []

    for value in AP_DICT.values():
        metrics_list.append(value)

    columns_all_seeds = columns_all_seeds + metrics_list

    df_all_seeds = pd.DataFrame(columns=columns_all_seeds)
    df_std = pd.DataFrame(columns=columns_std)

    for metric in metrics_list:
        df_std = df_std.append({columns_std[0]: metric}, ignore_index=True)

    files = os.listdir(saved_models_path)
    files = [file for file in files if file != '.gitkeep']

    for file in files:
        results_dict = {}

        save_state = torch.load(os.path.join(saved_models_path, file), map_location=torch.device('cpu'))
        seed_init = save_state['args_dict']['seed_init']

        results_dict[columns_all_seeds[0]] = str(seed_init)

        event_acc = EventAccumulator(os.path.join(log_path, file))
        event_acc.Reload()

        for tag_long, tag_short in scalars_dict.items():
            _, _, metric_value = zip(*event_acc.Scalars(tag_long))
            results_dict[tag_short] = metric_value[0]

        df_all_seeds = df_all_seeds.append(results_dict, ignore_index=True)

    df_all_seeds = df_all_seeds.sort_values('AP', ascending=False)

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
    experiment_letter = 'C'
    output_str = ''
    index = 99

    for phase in PHASES_LIST:
        df_all_seeds, df_std = parse_results(saved_models_path=os.path.join(PATH_C, 'saved_models'),
                                             log_path=os.path.join(PATH_C, 'logdir'),
                                             scalars_dict=get_scalars_dict(phase))

        output_str += get_latex_exp_name(experiment_letter,
                                         phase=phase)

        output_str += get_latex_ap_table(df_all_seeds,
                                         index=index,
                                         letter=experiment_letter,
                                         phase=phase,
                                         hparam='All seeds')

        index += 1

        output_str += get_latex_ap_table(df_std,
                                         index=index,
                                         letter=experiment_letter,
                                         phase=phase,
                                         hparam='mean ± std')

        index += 1

    save_latex(output_str,
               letter=experiment_letter,
               path='../reports/')
