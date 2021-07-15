import pandas as pd
import os

if __name__ == '__main__':
    ods_path = '/home/simon/Documents/holdout.ods'
    from_path = '/home/simon/Documents/training'
    to_path = '/home/simon/Documents/holdout'

    df = pd.read_excel(ods_path, engine="odf", header=None)

    for index, row in df.iterrows():
        os.replace(f'{from_path}/S{row[0]:04}.jpg', f'{to_path}/S{row[0]:04}.jpg')


