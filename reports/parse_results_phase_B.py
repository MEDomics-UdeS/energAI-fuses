from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
import os
import pandas as pd
from tqdm import tqdm
from constants import RESULTS_B_DICT, SCALARS_B_DICT
from parsing_utils import get_latex_ap_table, get_latex_exp_name, get_digits_precision, save_latex


def parse_results_k(results_all_df: pd.DataFrame,
                    saved_models_path: str,
                    log_path: str,
                    hyperparameter: str,
                    metric: str,
                    round_to_1_digit: bool = False,
                    num_decimals: int = 4) -> pd.DataFrame:
    hp_print_name = hyperparameter.replace('_', '\\_')

    df = pd.DataFrame(columns=[hp_print_name])

    files = os.listdir(saved_models_path)
    files = [file for file in files if file != '.gitkeep']

    cv_runs = list(set(["_".join(file.split("_")[-2:]) for file in files]))

    for row, cv_run in enumerate(cv_runs):
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

            df.loc[row, k] = metric_value[0]

    cols = df.columns.tolist()
    k_cols = cols[1:]
    k_cols.sort()
    df = df[[hp_print_name] + k_cols]

    df.fillna('', inplace=True)

    mean_list = []

    for i, row in df.iterrows():
        std = row[1:].std()

        precision = get_digits_precision(std)
        std_round = round(std, precision)
        precision = get_digits_precision(std_round)
        std_round = round(std, precision)

        format_str = f'{{:.{precision}f}}'

        result_str = f'{format_str.format(round(row[1:].mean(), precision))} ± ' \
                     f'{format_str.format(std_round)}'

        mean_list.append(result_str)

        first_col = results_all_df.columns.to_list()[0]
        hp_value = f'{df.loc[i, df.columns.to_list()[0]]}'
        hp_name_value = f'{hp_print_name}/{hp_value}'

        if hp_name_value in results_all_df[first_col].to_list():
            results_all_df.loc[results_all_df[first_col] == hp_name_value, SCALARS_B_DICT[metric]] = result_str
        else:
            idx = len(results_all_df)

            results_all_df.loc[idx, first_col] = hp_name_value
            results_all_df.loc[idx, SCALARS_B_DICT[metric]] = result_str

        if round_to_1_digit:
            for column in df.columns.to_list()[1:]:
                df.loc[i, column] = format_str.format(round(df.loc[i, column], precision))
        else:
            for column in df.columns.to_list()[1:]:
                df.loc[i, column] = f'{{:.{num_decimals}f}}'.format(round(df.loc[i, column], num_decimals))

    df['mean ± std'] = mean_list

    df = df.sort_values(hp_print_name)

    return df, results_all_df


if __name__ == '__main__':
    index = 1

    output_str = ''

    results_all_df = pd.DataFrame(columns=['hyperparameter'] + list(SCALARS_B_DICT.values()))

    for hparam, path in tqdm(RESULTS_B_DICT.items(), desc='Parsing results...'):
        letter = path.split('/')[-1]

        output_str += get_latex_exp_name(letter=letter, hparam=hparam)

        for scalar_raw, scalar_clean in SCALARS_B_DICT.items():
            ap_table_k, results_all_df = parse_results_k(results_all_df=results_all_df,
                                                         saved_models_path=path + '/saved_models/',
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

    output_str = get_latex_exp_name(letter='B') + get_latex_ap_table(df=results_all_df,
                                                                     index=0,
                                                                     letter='B') + output_str

    save_latex(output_str, letter='B')
