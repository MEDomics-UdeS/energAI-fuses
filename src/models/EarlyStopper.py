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

# MIT License
#
# Copyright (c) 2018 Stefano Nardo https://gist.github.com/stefanonardo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
        assert mode in {'min', 'max'}

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
