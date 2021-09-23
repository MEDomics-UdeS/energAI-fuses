from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
import os
import pandas as pd
from tqdm import tqdm
from constants import *
from parsing_utils import *
from numpy import log10, floor


# def round_to_1_sign_fig(x: pd.DataFrame) -> pd.DataFrame:
#     return round(x, -int(floor(log10(abs(x)))))


def parse_results_k(saved_models_path: str,
                    log_path: str,
                    hyperparameter: str,
                    metric: str,
                    round_to_1_digit: bool = True) -> pd.DataFrame:
    hp_print_name = hyperparameter.replace('_', '\\_')

    df = pd.DataFrame(columns=[hp_print_name])

    for subdir, dirs, files in os.walk(saved_models_path):
        files = [file for file in files if file != '.gitkeep']

        cv_runs = list(set(["_".join(file.split("_")[-2:]) for file in files]))

        row = 0

        for cv_run in cv_runs:  # tqdm(cv_runs, desc='Parsing results...'):
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

    df['AP_{mean}'] = df[k_cols].mean(axis=1)
    df['AP_{std}'] = df[k_cols].std(axis=1)

    if round_to_1_digit:
        df['precision'] = -(floor(log10(abs(df['AP_{std}'])))).astype(int)
        first_col = df[df.columns.tolist()[0]]
        df = df.apply(lambda x: round(x[df.columns.tolist()[1:-1]], int(x['precision'])), axis=1)
        df = pd.concat([first_col, df], axis=1)

    df = df.sort_values(hp_print_name)

    return df


def parse_results_all():
    pass


if __name__ == '__main__':
    index = 1

    print_latex_header()

    for hparam, path in RESULTS_B_DICT.items():
        letter = path.split('/')[-1]

        print_experiment_name(letter=letter,
                              hparam=hparam)

        for scalar_raw, scalar_clean in SCALARS_B_DICT.items():
            ap_table_k = parse_results_k(saved_models_path=path + '/saved_models/',
                                         log_path=path + '/logdir/',
                                         hyperparameter=hparam,
                                         metric=scalar_raw,
                                         round_to_1_digit=False)

            print_ap_table(letter=letter,
                           hparam=hparam,
                           metric=scalar_clean,
                           index=index,
                           df=ap_table_k)

            index += 1

    print_latex_footer()

    # ap_table_all = parse_results_all()
