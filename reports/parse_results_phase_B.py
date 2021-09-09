from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
import os
import pandas as pd
from tqdm import tqdm
from parsing_utils import print_ap_table


def parse_results(saved_models_path: str,
                  log_path: str,
                  hyperparameter: str,
                  metric: str,
                  decimals: int = 4) -> pd.DataFrame:
    hp_print_name = hyperparameter.replace('_', ' ')

    df = pd.DataFrame(columns=[hp_print_name])

    for subdir, dirs, files in os.walk(saved_models_path):
        files = [file for file in files if file != '.gitkeep']

        cv_runs = list(set(["_".join(file.split("_")[-2:]) for file in files]))

        row = 0

        for cv_run in tqdm(cv_runs, desc='Parsing results...'):
            files_run = [i for i in files if cv_run in i]

            save_state = torch.load(saved_models_path + files_run[0], map_location=torch.device('cpu'))
            hp_value = save_state['args_dict'][hyperparameter]

            df.loc[row, hp_print_name] = hp_value

            for file_run in files_run:
                event_acc = EventAccumulator(log_path + file_run)
                event_acc.Reload()
                _, _, metric_value = zip(*event_acc.Scalars(metric))

                k = f"K={''.join(c for c in file_run.split('_')[2] if c.isdigit())}"
                df.loc[row, k] = round(metric_value[0], decimals)

            row += 1

    cols = df.columns.tolist()
    k_cols = cols[1:]
    k_cols.sort()
    df = df[[hp_print_name] + k_cols]

    df.fillna('', inplace=True)

    df['AP_{mean}'] = round(df[k_cols].mean(axis=1), decimals)
    df['AP_{std}'] = round(df[k_cols].std(axis=1), decimals)

    df = df.sort_values(hp_print_name)

    return df


if __name__ == '__main__':
    os.chdir('..')
    models_path = os.getcwd() + '/saved_models/'
    logs_path = os.getcwd() + '/logdir/'

    ap_table = parse_results(saved_models_path=models_path,
                             log_path=logs_path,
                             hyperparameter='image_size',
                             metric='hparams/Validation/AP @ [IoU=0.50:0.95 | area=all | maxDets=100]',
                             decimals=4)
    print_ap_table(ap_table)
