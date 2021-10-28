from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from math import floor, ceil, log10
import torch
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from parsing_utils import get_latex_ap_table, get_latex_exp_name, save_latex
from constants import PATH_A


def generate_figure(metric: str,
                    curves_dict: dict,
                    save: bool = True,
                    show: bool = False) -> None:
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
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

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
        ax.set_xlabel('Epoch')

    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')

    fig.tight_layout()

    fig.subplots_adjust(right=0.7)

    if save:
        file_path = '../reports/figure.svg'
        fig.savefig(file_path)
        print(f'Figure has been saved to: {file_path}')

    if show:
        fig.show()


def parse_results(saved_models_path: str,
                  log_path: str,
                  model_name: str,
                  num_decimals: int = 4) -> pd.DataFrame:
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

    files = os.listdir(saved_models_path)
    files = [file for file in files if file != '.gitkeep']

    for file in files:
        save_state = torch.load(saved_models_path + file, map_location=torch.device('cpu'))
        model = save_state['args_dict']['model'].split('_')[0]

        if model == model_name:
            lr = format(save_state['args_dict']['learning_rate'], '.0E')
            wd = format(save_state['args_dict']['weight_decay'], '.0E')
            da = save_state['args_dict']['data_aug']

            event_acc = EventAccumulator(log_path + file)
            event_acc.Reload()
            # scalars = event_acc.Tags()['scalars']

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

    return df, ap_curves_dict, loss_curves_dict, lr_curves_dict


def add_curve_to_dict(acc: EventAccumulator, scalar_key: str, run_key: str, curves_dict: dict) -> dict:
    times, steps, vals = zip(*acc.Scalars(scalar_key))
    curves_dict[run_key] = {'x': steps, 'y': vals}

    return curves_dict


def get_best_results(results_dict: dict,
                     metric: str) -> pd.DataFrame:
    best_results_df = pd.DataFrame()

    best_ap_curves = {}
    best_loss_curves = {}
    best_lr_curves = {}

    for model_name, results in results_dict.items():
        df = results['ap_table']
        argmax_df = df.iloc[df[metric].argmax()]
        argmax_str = f'{argmax_df["Model"]}/{argmax_df["LR"]}/' \
                     f'{argmax_df["WD"]}/{argmax_df["DA"]}'

        best_results_df = best_results_df.append(argmax_df)
        best_ap_curves[model_name] = results['ap_curves'][argmax_str]
        best_loss_curves[model_name] = results['loss_curves'][argmax_str]
        best_lr_curves[model_name] = results['lr_curves'][argmax_str]

    return best_results_df, best_ap_curves, best_loss_curves, best_lr_curves


if __name__ == '__main__':
    models_path = PATH_A + '/saved_models/'
    logs_path = PATH_A + '/logdir/'

    results_dict = {'fasterrcnn': None,
                    'retinanet': None,
                    'detr': None}

    for model_name in tqdm(results_dict.keys(), desc='Parsing results...'):
        ap_table, ap_curves, loss_curves, lr_curves = parse_results(models_path, logs_path, model_name=model_name)
        results_dict[model_name] = {'ap_table': ap_table,
                                    'ap_curves': ap_curves,
                                    'loss_curves': loss_curves,
                                    'lr_curves': lr_curves}

    best_results_df, best_ap_curves, best_loss_curves, best_lr_curves = get_best_results(results_dict, 'AP')

    output_str = get_latex_exp_name('A')
    output_str += get_latex_ap_table(best_results_df, 0, 'A')

    for i, (model_name, results) in enumerate(results_dict.items()):
        output_str += get_latex_exp_name('A', hparam=model_name)
        output_str += get_latex_ap_table(results['ap_table'], i, 'A', hparam=model_name)

    generate_figure_all(best_ap_curves, best_loss_curves, best_lr_curves)

    save_latex(output_str, letter='A', path='../reports/')
