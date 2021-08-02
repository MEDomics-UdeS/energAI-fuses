from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
import os
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    os.chdir('..')
    saved_models_path = os.getcwd() + '/saved_models/'
    log_path = os.getcwd() + '/logdir/'

    columns = ['Model',
               'LR',
               'WD',
               'DA']

    scalar_dict = {
        'AP (Validation)/1. IoU=0.50:0.95 | area=all | maxDets=100': 'AP',
        'AP (Validation)/2. IoU=0.50 | area=all | maxDets=100': 'AP_{50}',
        'AP (Validation)/3. IoU=0.75 | area=all | maxDets=100': 'AP_{75}',
        'AP (Validation)/4. IoU=0.50:0.95 | area=small | maxDets=100': 'AP_{S}',
        'AP (Validation)/5. IoU=0.50:0.95 | area=medium | maxDets=100': 'AP_{M}',
        'AP (Validation)/6. IoU=0.50:0.95 | area=large | maxDets=100': 'AP_{L}'
    }

    for value in scalar_dict.values():
        columns.append(value)

    df = pd.DataFrame(columns=columns)

    for subdir, dirs, files in os.walk(saved_models_path):
        for file in tqdm(files, desc='Parsing results...'):
            if file != '.gitkeep':
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
                    times, steps, vals = zip(*event_acc.Scalars(key))
                    best_ap = round(max(vals) * 100, 1)
                    results_list.append(best_ap)

                df = df.append(pd.DataFrame([[model, lr, wd, da, *results_list]], columns=df.columns))

    print('*' * 50)
    print('LaTeX CODE START')
    print('*' * 50)
    print(df.to_latex(index=False, escape=False))
    print('*' * 50)
    print('LaTeX CODE END')
    print('*' * 50)
