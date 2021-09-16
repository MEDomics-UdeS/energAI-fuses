from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
import os
import pandas as pd
from tqdm import tqdm
from parsing_utils import print_ap_table
from numpy import log10, floor


# def round_to_1_sign_fig(x: pd.DataFrame) -> pd.DataFrame:
#     return round(x, -int(floor(log10(abs(x)))))


def parse_results_k(saved_models_path: str,
                    log_path: str,
                    hyperparameter: str,
                    metric: str) -> pd.DataFrame:
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
                # scalars = event_acc.Tags()['scalars']
                _, _, metric_value = zip(*event_acc.Scalars(metric))

                k = f"K={''.join(c for c in file_run.split('_')[2] if c.isdigit())}"
                df.loc[row, k] = metric_value[0]

            row += 1

    cols = df.columns.tolist()
    k_cols = cols[1:]
    k_cols.sort()
    df = df[[hp_print_name] + k_cols]

    df.fillna('', inplace=True)

    df['AP_{mean}'] = df[k_cols].mean(axis=1)
    df['AP_{std}'] = df[k_cols].std(axis=1)

    df['precision'] = -(floor(log10(abs(df['AP_{std}'])))).astype(int)
    first_col = df[df.columns.tolist()[0]]
    df = df.apply(lambda x: round(x[df.columns.tolist()[1:-1]], int(x['precision'])), axis=1)
    df = pd.concat([first_col, df], axis=1)

    df = df.sort_values(hp_print_name)

    return df


def parse_results_all():
    scalar_dict = {
        'hparams/Validation/AP @ [IoU=0.50:0.95 | area=all | maxDets=100]': 'AP',
        'hparams/Validation/AP @ [IoU=0.50 | area=all | maxDets=100]': 'AP_{50}',
        'hparams/Validation/AP @ [IoU=0.75 | area=all | maxDets=100]': 'AP_{75}',
        'hparams/Validation/AP @ [IoU=0.50:0.95 | area=small | maxDets=100]': 'AP_{S}',
        'hparams/Validation/AP @ [IoU=0.50:0.95 | area=medium | maxDets=100]': 'AP_{M}',
        'hparams/Validation/AP @ [IoU=0.50:0.95 | area=large | maxDets=100]': 'AP_{L}'
    }


if __name__ == '__main__':
    os.chdir('..')
    models_path = os.getcwd() + '/saved_models/'
    logs_path = os.getcwd() + '/logdir/'

    ap_table_k = parse_results_k(saved_models_path=models_path,
                                 log_path=logs_path,
                                 hyperparameter='image_size',
                                 metric='hparams/Validation/AP @ [IoU=0.50:0.95 | area=all | maxDets=100]')

    print_ap_table(ap_table_k)

    ap_table_all = parse_results_all()
