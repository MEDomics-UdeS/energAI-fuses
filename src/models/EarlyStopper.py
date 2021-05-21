"""
File:
    src/models/EarlyStopper.py

Authors:
    - Simon Giard-Leroux
    - Shreyas Sunil Kulkarni

Description:
    Early stopping object to stop the training once a certain validation metric (with a delta variation)
    has been reached for a fixed number of epochs (patience).

    Inspired by: https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
"""

import torch


class EarlyStopper:
    """
    Early stopping class
    """
    def __init__(self, patience: int, min_delta: float, mode: str = 'max', percentage: bool = False) -> None:
        """
        Class constructor

        :param patience: int, number of epochs with no improvement before early stopping occurs
        :param min_delta: float, delta of tolerance for improvement evaluation
        :param mode: str, improvement evaluation mode 'min' or 'max'
        :param percentage: bool, if True, percentage mode
                                 if False, absolute value mode
        """
        self.__mode = mode
        self.__min_delta = min_delta
        self.__patience = patience
        self.__best = None
        self.__num_bad_epochs = 0
        self.__is_better = None
        self.__init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.__is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics: float) -> bool:
        """
        Step function

        :param metrics: float, metric to evaluate
        :return: bool, if True, early stop
                       if False, continue
        """
        if self.__best is None:
            self.__best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.__is_better(metrics, self.__best):
            self.__num_bad_epochs = 0
            self.__best = metrics
        else:
            self.__num_bad_epochs += 1

        if self.__num_bad_epochs >= self.__patience:
            return True

        return False

    def __init_is_better(self, mode: str, min_delta: float, percentage: bool) -> None:
        """
        Method to initialize the self.__is_better method depending on the mode, min_delta and percentage parameters

        :param mode: str, improvement evaluation mode 'min' or 'max'
        :param min_delta: float, delta of tolerance for improvement evaluation
        :param percentage: bool, if True, percentage mode
                                 if False, absolute value mode
        """
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.__is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.__is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.__is_better = lambda a, best: a < best - (
                        best * min_delta / 100)
            if mode == 'max':
                self.__is_better = lambda a, best: a > best + (
                        best * min_delta / 100)
