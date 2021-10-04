from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
import os
import pandas as pd
from tqdm import tqdm
from constants import RESULTS_B_DICT, SCALARS_B_DICT
from parsing_utils import get_latex_ap_table, get_latex_exp_name, get_digits_precision, save_latex


def parse_results_k(saved_models_path: str,
                    log_path: str,
                    hyperparameter: str,
                    metric: str,
                    round_to_1_digit: bool = False,
                    num_decimals: int = 4) -> pd.DataFrame:
    hp_print_name = hyperparameter.replace('_', '\\_')

    df = pd.DataFrame(columns=[hp_print_name])

    for subdir, dirs, files in os.walk(saved_models_path):
        files = [file for file in files if file != '.gitkeep']

        cv_runs = list(set(["_".join(file.split("_")[-2:]) for file in files]))

        row = 0

        for cv_run in cv_runs:
            files_run = [i for i in files if cv_run in i]

            save_state = torch.load(saved_models_path + files_run[0], map_location=torch.device('cpu'))
            hp_value = save_state['args_dict'][hyperparameter]

            df.loc[row, hp_print_name] = hp_value

            for file_run in files_run:
                event_acc = EventAccumulator(log_path + file_run)
                event_acc.Reload()
                # scalars = event_acc.Tags()['scalars']
                _, _, metric_value = zip(*event_acc.Scalars(metric))

                k = f"K={''.join(c for c in file_run.split('_')[2] if c.isdigit())}"

                # temp
                # save_state = torch.load(saved_models_path + file_run, map_location=torch.device('cpu'))
                # hp_value = save_state['args_dict'][hyperparameter]

                df.loc[row, k] = metric_value[0]  # hp_value

            row += 1

    cols = df.columns.tolist()
    k_cols = cols[1:]
    k_cols.sort()
    df = df[[hp_print_name] + k_cols]

    df.fillna('', inplace=True)

    mean_list = []

    for i, row in df.iterrows():
        precision = get_digits_precision(row[1:].std())
        format_str = f'{{:.{precision}f}}'

        mean_list.append(f'{format_str.format(round(row[1:].mean(), precision))} ± ' \
                         f'{format_str.format(round(row[1:].std(), precision))}')

        if round_to_1_digit:
            for column in df.columns.to_list()[1:]:
                df.loc[i, column] = format_str.format(round(df.loc[i, column], precision))
        else:
            format_str = f'{{:.{num_decimals}f}}'

            for column in df.columns.to_list()[1:]:
                df.loc[i, column] = format_str.format(round(df.loc[i, column], num_decimals))

    df['mean ± std'] = mean_list

    df = df.sort_values(hp_print_name)

    return df


def parse_results_all():
    pass


if __name__ == '__main__':
    index = 1

    output_str = ''

    for hparam, path in tqdm(RESULTS_B_DICT.items(), desc='Parsing results...'):
        letter = path.split('/')[-1]

        output_str += get_latex_exp_name(letter=letter, hparam=hparam)

        for scalar_raw, scalar_clean in SCALARS_B_DICT.items():
            ap_table_k = parse_results_k(saved_models_path=path + '/saved_models/',
                                         log_path=path + '/logdir/',
                                         hyperparameter=hparam,
                                         metric=scalar_raw,
                                         round_to_1_digit=False)
            output_str += get_latex_ap_table(df=ap_table_k,
                                             index=index,
                                             letter=letter,
                                             hparam=hparam,
                                             metric=scalar_clean)

            index += 1

    save_latex(output_str, letter='B')

    # ap_table_all = parse_results_all()
