from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
import os
import pandas as pd
from tqdm import tqdm
from constants import RESULTS_B_DICT, PHASES_LIST
from parsing_utils import get_latex_ap_table, get_latex_exp_name, get_digits_precision, save_latex, get_scalars_dict


def parse_results_k(results_all_df: pd.DataFrame,
                    phase: str,
                    scalars_dict: dict,
                    saved_models_path: str,
                    log_path: str,
                    hyperparameter: str,
                    metric: str,
                    round_to_1_digit: bool = False,
                    num_decimals: int = 4) -> pd.DataFrame:
    """

    Args:
        results_all_df(pd.DataFrame): 
        phase(str): 
        scalars_dict(dict): 
        saved_models_path(str): 
        log_path(str): 
        hyperparameter(str): 
        metric(str): 
        round_to_1_digit(bool, optional):  (Default value = False)
        num_decimals(int, optional):  (Default value = 4)

    Returns:

    """
    hp_print_name = hyperparameter.replace('_', '\\_')

    df = pd.DataFrame(columns=[hp_print_name])

    files = os.listdir(saved_models_path)
    files = [file for file in files if file != '.gitkeep']

    cv_runs = list(set(["_".join(file.split("_")[-2:]) for file in files]))

    for row, cv_run in enumerate(cv_runs):
        files_run = [i for i in files if cv_run in i]

        save_state = torch.load(os.path.join(saved_models_path, files_run[0]), map_location=torch.device('cpu'))
        hp_value = save_state['args_dict'][hyperparameter]

        df.loc[row, hp_print_name] = hp_value

        for file_run in files_run:
            event_acc = EventAccumulator(os.path.join(log_path, file_run))
            event_acc.Reload()
            # scalars = event_acc.Tags()['scalars']

            _, _, metric_value = zip(*event_acc.Scalars(metric))

            k = f"K={''.join(c for c in file_run.split('_')[2] if c.isdigit())}"

            df.loc[row, k] = metric_value[0] if metric_value[0] >= 0 else 0

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
        hp_name_value = f'{phase[0]}/{hp_print_name}/{hp_value}'

        if hp_name_value in results_all_df[first_col].to_list():
            results_all_df.loc[results_all_df[first_col] == hp_name_value, scalars_dict[metric]] = result_str
        else:
            idx = len(results_all_df)

            results_all_df.loc[idx, first_col] = hp_name_value
            results_all_df.loc[idx, scalars_dict[metric]] = result_str

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
    exp_letter = 'B'
    output_str = ''
    main_phase = 'Validation'

    results_all_df = pd.DataFrame(columns=['hyperparameter'] + list(get_scalars_dict(phase=main_phase).values()))

    for hparam, path in tqdm(RESULTS_B_DICT.items(), desc='Parsing results...'):
        letter = path.split('/')[-1]

        for phase in PHASES_LIST:
            output_str += get_latex_exp_name(letter=letter,
                                             phase=phase,
                                             hparam=hparam)

            scalars_dict = get_scalars_dict(phase=phase)

            for scalar_raw, scalar_clean in scalars_dict.items():
                ap_table_k, results_all_df = parse_results_k(results_all_df=results_all_df,
                                                             phase=phase,
                                                             scalars_dict=scalars_dict,
                                                             saved_models_path=os.path.join(path, 'saved_models'),
                                                             log_path=os.path.join(path, 'logdir'),
                                                             hyperparameter=hparam,
                                                             metric=scalar_raw,
                                                             round_to_1_digit=False)
                output_str += get_latex_ap_table(df=ap_table_k,
                                                 index=index,
                                                 letter=letter,
                                                 phase=phase,
                                                 hparam=hparam,
                                                 metric=scalar_clean)

                index += 1

    output_str = get_latex_exp_name(letter=exp_letter) + get_latex_ap_table(df=results_all_df,
                                                                            index=0,
                                                                            letter=exp_letter) + output_str

    save_latex(output_str, letter=exp_letter)
