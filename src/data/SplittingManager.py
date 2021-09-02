from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from typing import List
import os
import json
import torch

from src.data.Datasets.FuseDataset import FuseDataset
from src.utils.constants import *


class SplittingManager:
    def __init__(self,
                 validation_size: float,
                 test_size: float,
                 k_cross_valid: int,
                 seed: int,
                 num_workers: int,
                 google_images: bool) -> None:

        self.__validation_size = validation_size
        self.__test_size = test_size
        self.__k_cross_valid = k_cross_valid
        self.__seed = seed
        self.__google_images = google_images

        images = [img for img in sorted(os.listdir(RESIZED_LEARNING_PATH)) if img.startswith('.') is False]

        if not google_images:
            google_imgs = [image for image in images if image.startswith('G')]
            google_indices = [images.index(google_image) for google_image in google_imgs]
            images = [e for i, e in enumerate(images) if i not in google_indices]

        self.__image_paths = [os.path.join(RESIZED_LEARNING_PATH, img) for img in images]

        self.__targets = json.load(open(TARGETS_LEARNING_PATH))

        if not google_images:
            self.__targets = [e for i, e in enumerate(self._targets) if i not in google_indices]

        # Convert the targets to tensors
        for target in self.__targets:
            for key, value in target.items():
                target[key] = torch.as_tensor(value, dtype=torch.int64)

        self.__split_dataset()

        # For backwards results reproducibility, valid and test sets are reverse ordered
        for i in range(k_cross_valid):
            self.__image_paths_valid.reverse()
            self.__targets_valid.reverse()
            self.__image_paths_test.reverse()
            self.__targets_test.reverse()

    @property
    def image_paths_train(self) -> list:
        return self.__image_paths_train

    @property
    def targets_train(self) -> list:
        return self.__targets_train

    @property
    def image_paths_valid(self) -> list:
        return self.__image_paths_valid

    @property
    def targets_valid(self) -> list:
        return self.__targets_valid

    @property
    def image_paths_test(self) -> list:
        return self.__image_paths_test

    @property
    def targets_test(self) -> list:
        return self.__targets_test

    def __split_dataset(self) -> None:
        image_paths = self.__image_paths
        targets = self.__targets

        total_size = sum(image_path.rsplit('/')[-1].startswith('S') for image_path in image_paths)

        if 0 < self.__validation_size < 1:
            self.__validation_size = self.__validation_size / (1 - HOLDOUT_SIZE)

        if 0 < self.__test_size < 1:
            self.__test_size = self.__test_size / (1 - HOLDOUT_SIZE)

        if self.__google_images:
            google_image_paths = [image_path for image_path in image_paths
                                  if image_path.rsplit('/')[-1].startswith('G')]

            google_indices = [image_paths.index(google_image_path)
                              for google_image_path in google_image_paths]

            google_targets = self.__filter_list(targets, google_indices, True)

            image_paths = self.__filter_list(image_paths, google_indices, False)
            targets = self.__filter_list(targets, google_indices, False)

        most_freq_labels = self.__get_most_frequent_labels(targets)

        if self.__k_cross_valid > 1:
            if self.__test_size > 0:
                strat_split_test = StratifiedShuffleSplit(n_splits=1,
                                                          test_size=self.__test_size,
                                                          random_state=self.__seed)

                for _, indices_test in strat_split_test.split([0] * len(image_paths), most_freq_labels):
                    break

                indices_test = list(indices_test)

                self.__image_paths_test = self.__filter_list(image_paths, indices_test, True)
                self.__targets_test = self.__filter_list(targets, indices_test, True)

                image_paths = self.__filter_list(image_paths, indices_test, False)
                targets = self.__filter_list(targets, indices_test, False)

                most_freq_labels = self.__get_most_frequent_labels(targets)
            else:
                self.__image_paths_test = []
                self.__targets_test = []

            strat_split_valid = StratifiedKFold(n_splits=self.__k_cross_valid,
                                                random_state=self.__seed, shuffle=True)

            indices_valid = []

            for _, indices in strat_split_valid.split([0] * len(image_paths), most_freq_labels):
                indices_valid.append(list(indices))

            image_paths_train = []
            targets_train = []

            image_paths_valid = []
            targets_valid = []

            for i in range(self.__k_cross_valid):
                image_paths_train.append(self.__filter_list(image_paths, indices_valid[i], False))
                targets_train.append(self.__filter_list(targets, indices_valid[i], False))

                if self.__google_images:
                    image_paths_train[i] = image_paths_train[i] + google_image_paths
                    targets_train[i] = targets_train[i] + google_targets

                image_paths_valid.append(self.__filter_list(image_paths, indices_valid[i], True))
                targets_valid.append(self.__filter_list(targets, indices_valid[i], True))

            self.__image_paths_train = image_paths_train
            self.__targets_train = targets_train

            self.__image_paths_valid = image_paths_valid
            self.__targets_valid = targets_valid
        else:
            if self.__validation_size == 0 and self.__test_size == 0:
                self.__image_paths_train = [image_paths + google_image_paths]
                self.__targets_train = [targets + google_image_paths]

                self.__image_paths_valid = [[]]
                self.__targets_valid = [[]]

                self.__image_paths_test = []
                self.__targets_test = []
            else:
                if self.__validation_size > 0:
                    strat_split_valid = StratifiedShuffleSplit(n_splits=1,
                                                               test_size=self.__validation_size,
                                                               random_state=self.__seed)

                    for _, indices_valid in strat_split_valid.split([0] * len(image_paths), most_freq_labels):
                        break

                    indices_valid = list(indices_valid)

                    self.__image_paths_valid = [self.__filter_list(image_paths, indices_valid, True)]
                    self.__targets_valid = [self.__filter_list(targets, indices_valid, True)]

                    image_paths = self.__filter_list(image_paths, indices_valid, False)
                    targets = self.__filter_list(targets, indices_valid, False)

                    most_freq_labels = self.__get_most_frequent_labels(targets)
                else:
                    self.__image_paths_valid = [[]]
                    self.__targets_valid = [[]]

                if self.__test_size > 0:
                    strat_split_test = StratifiedShuffleSplit(n_splits=1,
                                                              test_size=self.__test_size * total_size / len(image_paths),
                                                              random_state=self.__seed)

                    for _, indices_test in strat_split_test.split([0] * len(image_paths), most_freq_labels):
                        break

                    indices_test = list(indices_test)

                    self.__image_paths_train = [self.__filter_list(image_paths, indices_test, False) + google_image_paths]
                    self.__targets_train = [self.__filter_list(targets, indices_test, False) + google_targets]

                    self.__image_paths_test = self.__filter_list(image_paths, indices_test, True)
                    self.__targets_test = self.__filter_list(targets, indices_test, True)
                else:
                    self.__image_paths_train = [image_paths + google_image_paths]
                    self.__targets_train = [targets + google_targets]

                    self.__image_paths_test = []
                    self.__targets_test = []

    @staticmethod
    def __filter_list(my_list: list, indices: str, logic: bool) -> list:
        if logic:
            return [my_list[i] for i in range(len(my_list)) if i in indices]
        else:
            return [my_list[i] for i in range(len(my_list)) if i not in indices]

    @staticmethod
    def __get_most_frequent_labels(targets: list) -> list:
        return [max(set(target['labels'].tolist()), key=target['labels'].tolist().count)
                for target in targets]
