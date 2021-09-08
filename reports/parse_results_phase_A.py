from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import math
import torch
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from parsing_utils import print_ap_table


def generate_figure(metric: str, curves_dict: dict, save: bool = False, show: bool = True) -> None:
    x_max = 0
    y_max = 0
    y_min = 1e15

    for key, value in curves_dict.items():
        if metric == 'Learning Rate':
            plt.semilogy(value['x'], value['y'], label=key)
        else:
            plt.plot(value['x'], value['y'], label=key)

        x_val_max = max(value['x'])
        y_val_max = max(value['y'])
        y_val_min = min(value['y'])

        if x_val_max > x_max:
            x_max = x_val_max

        if y_val_max > y_max:
            y_max = y_val_max

        if y_val_min < y_min:
            y_min = y_val_min

    plt.subplots_adjust(right=0.7)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.xlim((0, x_max))

    if metric == 'AP':
        plt.ylim((0, 1))
    elif metric == 'Mean Loss':
        plt.ylim((0, y_max))
    elif metric == 'Learning Rate':
        plt.ylim((10 ** math.floor(math.log10(y_min)), 10 ** math.ceil(math.log10(y_max))))

    if save:
        file_path = f'reports/{metric}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pdf'
        plt.savefig(file_path)
        print(f'{metric} figure has been saved to: {file_path}')

    if show:
        plt.show()


def parse_results(saved_models_path: str, log_path: str) -> pd.DataFrame:
    columns = ['Model',
               'LR',
               'WD',
               'DA']

    scalar_dict = {
        'hparams/AP @ [IoU=0.50:0.95 | area=all | maxDets=100]': 'AP',
        'hparams/AP @ [IoU=0.50 | area=all | maxDets=100]': 'AP_{50}',
        'hparams/AP @ [IoU=0.75 | area=all | maxDets=100]': 'AP_{75}',
        'hparams/AP @ [IoU=0.50:0.95 | area=small | maxDets=100]': 'AP_{S}',
        'hparams/AP @ [IoU=0.50:0.95 | area=medium | maxDets=100]': 'AP_{M}',
        'hparams/AP @ [IoU=0.50:0.95 | area=large | maxDets=100]': 'AP_{L}'
    }

    ap_curves_dict = {}
    ap_curves_dict_key = 'AP (Validation)/1. IoU=0.50:0.95 | area=all | maxDets=100'

    loss_curves_dict = {}
    loss_curves_dict_key = 'Loss/Training (mean per epoch)'

    lr_curves_dict = {}
    lr_curves_dict_key = 'Learning Rate'

    for value in scalar_dict.values():
        columns.append(value)

    df = pd.DataFrame(columns=columns)

    for subdir, dirs, files in os.walk(saved_models_path):
        files = [file for file in files if file != '.gitkeep']
        for file in tqdm(files, desc='Parsing results...'):
            save_state = torch.load(saved_models_path + file, map_location=torch.device('cpu'))
            model = save_state['args_dict']['model'].split('_')[0]
            lr = format(save_state['args_dict']['learning_rate'], '.0E')
            wd = format(save_state['args_dict']['weight_decay'], '.0E')
            da = save_state['args_dict']['data_aug']

            event_acc = EventAccumulator(log_path + file)
            event_acc.Reload()
            # scalars = event_acc.Tags()['scalars']

            results_list = []

            for key, value in scalar_dict.items():
                time, step, val = zip(*event_acc.Scalars(key))
                best_ap = round(val[0] * 100, 1)

                results_list.append(best_ap)

            df = df.append(pd.DataFrame([[model, lr, wd, da, *results_list]], columns=df.columns))

            run_key = f'{model}/{lr}/{wd}/{da}'

            ap_curves_dict = add_curve_to_dict(event_acc, ap_curves_dict_key, run_key, ap_curves_dict)
            loss_curves_dict = add_curve_to_dict(event_acc, loss_curves_dict_key, run_key, loss_curves_dict)
            lr_curves_dict = add_curve_to_dict(event_acc, lr_curves_dict_key, run_key, lr_curves_dict)

    return df, ap_curves_dict, loss_curves_dict, lr_curves_dict


def add_curve_to_dict(acc: EventAccumulator, scalar_key: str, run_key: str, curves_dict: dict) -> dict:
    times, steps, vals = zip(*acc.Scalars(scalar_key))
    curves_dict[run_key] = {'x': steps, 'y': vals}

    return curves_dict


def print_best_result(df: pd.DataFrame, metric: str) -> None:
    print('\nBest Result:\n')
    print(df.iloc[df[metric].argmax()])


if __name__ == '__main__':
    os.chdir('..')
    models_path = os.getcwd() + '/saved_models/'
    logs_path = os.getcwd() + '/logdir/'

    ap_table, ap_curves, loss_curves, lr_curves = parse_results(models_path, logs_path)

    print_ap_table(ap_table)

    generate_figure('AP', ap_curves)
    generate_figure('Mean Loss', loss_curves)
    generate_figure('Learning Rate', lr_curves)

    print_best_result(ap_table, 'AP')
