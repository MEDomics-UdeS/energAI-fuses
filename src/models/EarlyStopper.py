"""
EarlyStopper

  - Copyright Holder: Stefano Nardo
  - Source: https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
  - License: MIT: https://opensource.org/licenses/MIT
"""

import torch


class EarlyStopper:
    """Early stopping class"""
    def __init__(self,
                 patience: int,
                 min_delta: float,
                 mode: str = 'max',
                 percentage: bool = False) -> None:
        """Class constructor

        Args:
            patience(int): number of epochs with no improvement before early stopping occurs
            min_delta(float): delta of tolerance for improvement evaluation
            mode(str, optional): improvement evaluation mode 'min' or 'max' (Default value = 'max')
            percentage(bool, optional): if True, percentage mode
                                        if False, absolute value mode (Default value = False)

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

    def step(self,
             metrics: float) -> bool:
        """Step function

        Args:
            metrics(float): metric to evaluate

        Returns:
            bool: if True, early stop
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

    def __init_is_better(self,
                         mode: str,
                         min_delta: float,
                         percentage: bool) -> None:
        """Method to initialize the self.__is_better method depending on the mode, min_delta and percentage parameters

        Args:
            mode(str): improvement evaluation mode 'min' or 'max'
            min_delta(float): delta of tolerance for improvement evaluation
            percentage(bool): if True, percentage mode
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
