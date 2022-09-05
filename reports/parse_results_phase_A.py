"""
File:
    reports/parse_results_phase_A.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Parsing script for phase A results
"""


from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from math import floor, ceil, log10
import torch
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from typing import Optional

from reports.parsing_utils import get_latex_ap_table, get_latex_exp_name, save_latex
from reports.constants import PATH_A, PHASES_LIST


def generate_figure(metric: str,
                    curves_dict: dict,
                    save: bool = True,
                    show: bool = False) -> None:
    """Function to generate individual AP, loss or learning rate figure for phase A

    Args:
        metric(str): metric to plot, either 'AP', 'Mean Loss' or 'Learning Rate'
        curves_dict(dict): curves dictionary
        save(bool, optional): choose whether to save the figure (Default value = True)
        show(bool, optional): choose whether to show the figure  (Default value = False)

    """
    x_max = 0
    y_max = 0
    y_min = 1e15

    plt.clf()

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
        plt.ylim((10 ** floor(log10(y_min)), 10 ** ceil(log10(y_max))))

    plt.tight_layout()

    if save:
        file_path = f'../reports/{metric}.pdf'
        plt.savefig(file_path)
        print(f'{metric} figure has been saved to: {file_path}')

    if show:
        plt.show()


def generate_figure_all(best_ap_curves: dict,
                        best_loss_curves: dict,
                        best_lr_curves: dict,
                        save: bool = True,
                        show: bool = False) -> None:
    """Function to plot all figures for phase A: AP, mean loss and learning rate

    Args:
        best_ap_curves(dict): Dictionary of AP curves for best performing model for each architecture
        best_loss_curves(dict): Dictionary of mean loss curves for best performing model for each architecture
        best_lr_curves(dict): Dictionary of learning rate curves for best performing model for each architecture
        save(bool, optional): Choose whether to save the figure (Default value = True)
        show(bool, optional): Choose whether to show the figure (Default value = False)

    """
    fig, axs = plt.subplots(3, 1, figsize=(6, 12))

    x_max = 0

    # AP Curve
    for key, value in best_ap_curves.items():
        axs[0].plot(value['x'], value['y'], label=key)

        x_val_max = max(value['x'])

        if x_val_max > x_max:
            x_max = x_val_max

    axs[0].set_ylabel('AP')
    axs[0].set_xlim((0, x_max))
    axs[0].set_ylim((0, 1))

    x_max = 0
    y_max = 0

    # Loss Curve
    for key, value in best_loss_curves.items():
        axs[1].plot(value['x'], value['y'], label=key)

        x_val_max = max(value['x'])
        y_val_max = max(value['y'])

        if x_val_max > x_max:
            x_max = x_val_max

        if y_val_max > y_max:
            y_max = y_val_max

    axs[1].set_ylabel('Mean Loss')
    axs[1].set_xlim((0, x_max))
    axs[1].set_ylim((0, y_max))

    x_max = 0
    y_max = 0
    y_min = 1e15

    # LR Curve
    for key, value in best_lr_curves.items():
        axs[2].semilogy(value['x'], value['y'], label=key)

        x_val_max = max(value['x'])
        y_val_max = max(value['y'])
        y_val_min = min(value['y'])

        if x_val_max > x_max:
            x_max = x_val_max

        if y_val_max > y_max:
            y_max = y_val_max

        if y_val_min < y_min:
            y_min = y_val_min

    axs[2].set_ylabel('Learning Rate')
    axs[2].set_xlim((0, x_max))
    axs[2].set_ylim((10 ** floor(log10(y_min)), 10 ** ceil(log10(y_max))))

    for ax in axs:
        ax.grid()
        ax.legend()
        ax.set_xlabel('Epoch')

    # handles, labels = axs[-1].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='center right')

    fig.tight_layout()

    # fig.subplots_adjust(right=0.7)

    if save:
        file_path = '../reports/figure.svg'
        fig.savefig(file_path)
        print(f'Figure has been saved to: {file_path}')

    if show:
        fig.show()


def parse_results(model_name: str, *,
                  models_path: Optional[str] = None,
                  logs_path: Optional[str] = None,
                  json_path: Optional[str] = None,
                  num_decimals: int = 4) -> pd.DataFrame:
    """Function to parse the results for phase A from the tensorboard results

    Args:
        model_name(str): model name
        models_path(Optional[str], optional): models path (Default value = None)
        logs_path(Optional[str], optional): logdir path (Default value = None)
        json_path(Optional[str], optional): json path (Default value = None)
        num_decimals(int, optional): number of decimals for results printing (Default value = 4)

    Returns: pandas dataframe with parsed results

    """
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

    if json_path:
        with open(json_path) as json_file:
            results_dict = json.load(json_file)

        for run_dict in results_dict.values():
            model = run_dict['args']['model'].split('_')[0]

            if model == model_name:
                lr = format(run_dict['args']['learning_rate'], '.0E')
                wd = format(run_dict['args']['weight_decay'], '.0E')
                da = run_dict['args']['data_aug']

                results_list = []

                for key, value in scalar_dict.items():
                    ap = round(run_dict['results'][key.split('/')[-1]], num_decimals)

                    results_list.append(ap)

                df = df.append(pd.DataFrame([[model, lr, wd, da, *results_list]], columns=df.columns))
    else:
        files = os.listdir(models_path)
        files = [file for file in files if file != '.gitkeep']

        for file in files:
            save_state = torch.load(os.path.join(models_path, file), map_location=torch.device('cpu'))
            model = save_state['args_dict']['model'].split('_')[0]

            if model == model_name:
                lr = format(save_state['args_dict']['learning_rate'], '.0E')
                wd = format(save_state['args_dict']['weight_decay'], '.0E')
                da = save_state['args_dict']['data_aug']

                event_acc = EventAccumulator(logs_path + file)
                event_acc.Reload()

                results_list = []

                for key, value in scalar_dict.items():
                    time, step, val = zip(*event_acc.Scalars(key))
                    best_ap = round(val[0], num_decimals)

                    results_list.append(best_ap)

                df = df.append(pd.DataFrame([[model, lr, wd, da, *results_list]], columns=df.columns))

                run_key = f'{model}/{lr}/{wd}/{da}'

                ap_curves_dict = add_curve_to_dict(event_acc, ap_curves_dict_key, run_key, ap_curves_dict)
                loss_curves_dict = add_curve_to_dict(event_acc, loss_curves_dict_key, run_key, loss_curves_dict)
                lr_curves_dict = add_curve_to_dict(event_acc, lr_curves_dict_key, run_key, lr_curves_dict)

    df = df.sort_values(df.columns.to_list()[1:4])

    if json_path:
        return df
    else:
        return df, ap_curves_dict, loss_curves_dict, lr_curves_dict


def add_curve_to_dict(acc: EventAccumulator, scalar_key: str, run_key: str, curves_dict: dict) -> dict:
    """Function to add curve from tensorboard EventAccumulator to curves dictionary

    Args:
        acc(EventAccumulator): Tensorboard EventAccumulator object instance
        scalar_key(str): scalar key in tensorboard
        run_key(str): run key in tensorboard
        curves_dict(dict): curves dictionary

    Returns: curves dictionary

    """
    times, steps, vals = zip(*acc.Scalars(scalar_key))
    curves_dict[run_key] = {'x': steps, 'y': vals}

    return curves_dict


def get_best_results(results_dict: dict,
                     metric: str) -> (pd.DataFrame, dict, dict, dict):
    """Function to find the best results inside a results dictionary

    Args:
        results_dict(dict): results dictionary
        metric(str): metric chosen to evaluate the best results

    Returns: pandas DataFrame with best results, best ap curves dictionary, best loss curves dictionary and
             best learning rate curves dictionary

    """
    best_results_df = pd.DataFrame()

    best_ap_curves = {}
    best_loss_curves = {}
    best_lr_curves = {}

    for model_name, results in results_dict.items():
        for phase in PHASES_LIST:
            df = results[f'ap_table_{phase}']
            argmax_df = df.iloc[df[metric].argmax()]
            model = argmax_df['Model']
            argmax_df = argmax_df.replace(model, f"{phase[0]}/{argmax_df['Model']}")

            best_results_df = best_results_df.append(argmax_df)

            if phase == PHASES_LIST[0]:
                argmax_str = f'{model}/{argmax_df["LR"]}/' \
                             f'{argmax_df["WD"]}/{argmax_df["DA"]}'

                best_ap_curves[model_name] = results['ap_curves'][argmax_str]
                best_loss_curves[model_name] = results['loss_curves'][argmax_str]
                best_lr_curves[model_name] = results['lr_curves'][argmax_str]

    return best_results_df, best_ap_curves, best_loss_curves, best_lr_curves


if __name__ == '__main__':
    models_path = PATH_A + '/saved_models/'
    logs_path = PATH_A + '/logdir/'
    json_path = 'reports/results_phase_A.json'

    results_dict = {'fasterrcnn': None,
                    'retinanet': None,
                    'detr': None}

    for model_name in tqdm(results_dict.keys(), desc='Parsing results...'):
        ap_table_valid, ap_curves, loss_curves, lr_curves = parse_results(model_name,
                                                                          models_path=models_path,
                                                                          logs_path=logs_path)
        ap_table_test = parse_results(model_name,
                                      json_path='../reports/results_phase_A.json')

        results_dict[model_name] = {f'ap_table_{PHASES_LIST[0]}': ap_table_valid,
                                    f'ap_table_{PHASES_LIST[1]}': ap_table_test,
                                    'ap_curves': ap_curves,
                                    'loss_curves': loss_curves,
                                    'lr_curves': lr_curves}

    best_results_df, best_ap_curves, best_loss_curves, best_lr_curves = get_best_results(results_dict, 'AP')

    output_str = get_latex_exp_name('A')
    output_str += get_latex_ap_table(best_results_df, index=0, letter='A')

    i = 0

    for model_name, results in results_dict.items():
        for phase in PHASES_LIST:
            output_str += get_latex_exp_name('A',
                                             phase=phase,
                                             hparam=model_name)
            output_str += get_latex_ap_table(results[f'ap_table_{phase}'],
                                             index=i,
                                             letter='A',
                                             phase=phase,
                                             hparam=model_name)
            i += 1

    generate_figure_all(best_ap_curves, best_loss_curves, best_lr_curves)

    save_latex(output_str, letter='A', path='../reports/')
