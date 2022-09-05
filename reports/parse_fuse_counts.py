"""
File:
    reports/parse_fuse_counts.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Parsing script for fuse counts.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


if __name__ == '__main__':
    df = pd.read_excel('count_files.xlsx')

    classes = df.Class.to_list()
    n_samples = df.Samples.to_list()
    used = df.Used.to_list()

    n_classes_half = len(classes)//2

    classes_dict = {
        0: classes[:n_classes_half],
        1: classes[n_classes_half:],
    }
    n_samples_dict = {
        0: n_samples[:n_classes_half],
        1: n_samples[n_classes_half:],
    }
    used_dict = {
        0: used[:n_classes_half],
        1: used[n_classes_half:],
    }

    color_list = ['tab:blue' if status else 'tab:orange' for status in used]

    color_list_dict = {
        0: color_list[:n_classes_half],
        1: color_list[n_classes_half:],
    }

    fig, ax = plt.subplots(1, 2, figsize=(10, 14))  # 20, 24

    for i in range(len(ax)):
        y_pos = range(len(classes_dict[i]))

        bars = ax[i].barh(y_pos, n_samples_dict[i],
                          align='center',
                          color=color_list_dict[i],
                          tick_label=classes_dict[i])
        ax[i].bar_label(bars)
        ax[i].invert_yaxis()  # labels read top-to-bottom
        # ax.set_ylabel('Fuse Class')
        ax[i].set_xlabel('Number of Images')
        ax[i].set_xlim([0, 1100])
        ax[i].xaxis.grid()

    custom_lines = [Line2D([0], [0], color='tab:blue', lw=4),
                    Line2D([0], [0], color='tab:orange', lw=4)]

    ax[0].legend(custom_lines, ['Used Classes (n=3,189)', 'Unused Classes (n=2,850)'])

    plt.tight_layout()

    plt.savefig('giard5.pdf')

    plt.show()
