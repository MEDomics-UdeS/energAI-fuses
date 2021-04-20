"""
File:
    visualization/VisualizationManager.py

Authors:
    - Abir Riahi
    - Nicolas Raymond
    - Simon Giard-Leroux

Description:
    Defines the VisualizationManager class.
"""

import matplotlib.pyplot as plt
from typing import Union
from src.models.Expert import Expert
import numpy as np
import os
import json


class VisualizationManager:
    """
    Visualization manager to generate charts.
    """
    def __init__(self) -> None:
        """
        Declare some member attributes for chart formatting.
        """
        self.legend_loc = 'best'
        self.marker = '.'
        self.train_label = 'Training'
        self.valid_label = 'Validation'

    def show_loss_acc_chart(self, results: dict,
                            show: bool = True, save_path: Union[str, None] = None,
                            fig_format: str = 'pdf') -> None:
        """
        Display and format chart of loss and accuracy per epoch

        :param results: Dictionary, contains each metric to plot
        :param show: Boolean indicating we want to show the figure
        :param save_path: Path to save the image. The paths must include the file name. (None == unsaved)
        :param fig_format: Format used to save the figure

        :return: None
        """
        # Declare figure and axes
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # Loss chart
        ax1.plot(results['Training Loss'], marker=self.marker, label=self.train_label)
        ax1.plot(results['Validation Loss'], marker=self.marker, label=self.valid_label)
        ax1.legend(loc=self.legend_loc)
        ax1.set_title('Mean loss per epoch')
        ax1.set(xlabel='Epoch', ylabel='Mean loss')

        # Accuracy chart
        ax2.plot(results['Training Accuracy'], marker=self.marker, label=self.train_label)
        ax2.plot(results['Validation Accuracy'], marker=self.marker, label=self.valid_label)
        ax2.legend(loc=self.legend_loc)
        ax2.set_ylim([0, 1])
        ax2.set_title('Mean accuracy per epoch')
        ax2.set(xlabel='Epoch', ylabel='Mean accuracy')

        # Apply a tight layout so axes titles do not overlap with charts
        fig.tight_layout()

        # We save it
        if save_path is not None:
            plt.savefig(f"{save_path}.{fig_format}")

        # We show the plot
        if show:
            plt.show()

    def show_labels_history(self, expert: Expert,
                            show: bool = True, save_path: Union[str, None] = None,
                            fig_format: str = 'pdf') -> None:
        """
        Plot the growth of labeled items per class throughout the active learning iteration

        :param expert: Expert class, contains the labels history to plot
        :param show: Boolean indicating we want to show the figure
        :param save_path: Path to save the image. The paths must include the file name. (None == unsaved)
        :param fig_format: Format used to save the figure

        :return: None
        """

        # We save the number of active learning iterations done
        x = range(len(expert.labeled_history[0]))
        for k, history in expert.labeled_history.items():
            plt.plot(x, history, marker=self.marker, label=expert.idx_to_class[k])

        # We set x-axis steps
        plt.xticks(x)

        # We set axis labels and legend
        plt.ylabel('Number of labeled images')
        plt.xlabel('Active learning iterations')
        plt.legend(loc="upper center",
                   fontsize='x-small',
                   ncol=2,
                   bbox_to_anchor=(1.30, 1))
        plt.tight_layout()

        # We save it
        if save_path is not None:
            plt.savefig(f"{save_path}.{fig_format}")

        # We show the plot
        if show:
            plt.show()

    def show_labels_piechart(self, expert: Expert,
                             show: bool = True, save_path: Union[str, None] = None,
                             fig_format: str = 'pdf') -> None:
        """
        Create a piechart showing the percentage of each class represented by the labeled data.
        Code from : https://matplotlib.org/stable/gallery/pie_and_polar_charts/
                    pie_and_donut_labels.html#sphx-glr-gallery-pie-and-polar-charts-pie-and-donut-labels-py

        :param expert: Expert class, contains the labels history to plot
        :param show: Boolean indicating we want to show the figure
        :param save_path: Path to save the image. The paths must include the file name. (None == unsaved)
        :param fig_format: Format used to save the figure

        :return: None
        """

        # We extract the number of labeled images from each class
        labels_count = [(expert.idx_to_class[k], v[-1] - v[0]) for (k, v) in expert.labeled_history.items()]

        # We only keep the top 10 classes with most labels (if there are more than 10 labels)
        if len(labels_count) > 10:
            labels_count.sort(key=lambda t: t[1], reverse=True)
            labels_count = labels_count[:10]

        # We sort remaining labels by alphabetical order
        labels_count.sort(key=lambda t: t[0])
        labels, count = list(zip(*labels_count))

        # We add count string to labels
        labels = [f"{labels[i]} ({count[i]})" for i in range(len(labels))]

        # We create the figure
        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

        wedges, texts = ax.pie(count, wedgeprops=dict(width=0.5), startangle=-40)

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                  bbox=bbox_props, zorder=0, va="center")

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1) / 2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(labels[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                        horizontalalignment=horizontalalignment, **kw)

        # We save it
        if save_path is not None:
            plt.savefig(f"{save_path}.{fig_format}")

        # We show the plot
        if show:
            plt.show()

    def show_learning_curve(self, folder_prefix: str, model: str, curve_label: str = 'query_strategy',
                            show: bool = True, save_path: Union[str, None] = None,
                            fig_format: str = 'pdf') -> None:
        """
        Method to show the learning curve, accuracy vs the number of active learning instance queries

        :param folder_prefix: string, prefix of the folders to consider
        :param model: string, name of the neural network model to consider
        :param curve_label: string, name of the initialization parameter to plot for each curve
        :param show: Boolean indicating we want to show the figure
        :param save_path: Path to save the image. The paths must include the file name. (None == unsaved)
        :param fig_format: Format used to save the figure

        :return: None
        """
        # Load json records list
        records_list = self.load_results(folder_prefix, model)

        # Plot each accuracy list  with the corresponding parameter
        records_list.sort(key=lambda r: r['Initialization'][curve_label])
        for records in records_list:
            plt.plot(records['Query Instances'],
                     records['Validation-2 Accuracy'],
                     label=records['Initialization'][curve_label])

        # We set axis labels and legend
        plt.ylabel('Training Accuracy')
        plt.xlabel('Number of Instance Queries')
        plt.legend(loc=self.legend_loc)

        # We save it
        if save_path is not None:
            plt.savefig(f"{save_path}.{fig_format}")

        # We show the plot
        if show:
            plt.show()

    @staticmethod
    def load_results(folder_prefix: str, model: str) -> list:
        """
        Method to load results from json files

        :param folder_prefix: string, prefix of the folders to consider
        :param model: string, name of the neural network model to consider

        :return: list of json records
        """
        # Get list of all folders in the current working directory
        folder_list = [x[0].rsplit('/', 1)[-1] for x in os.walk(os.getcwd())]

        # Only keep folders that start with the folder_prefix string
        folder_list = [x for x in folder_list if x.startswith(folder_prefix)]

        # Declare blank records list
        records_list = []

        # Loop through each folder
        for folder in folder_list:
            # Load json records file
            records = json.load(open(os.path.join(folder, 'records.json')))

            # If model is the specified one, add the records to the records list
            if records['Initialization']['model'] == model:
                records_list.append(records)

        return records_list
