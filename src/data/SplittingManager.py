from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from typing import List

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

        dataset = FuseDataset(images_path=RESIZED_LEARNING_PATH,
                              targets_path=TARGETS_LEARNING_PATH,
                              num_workers=num_workers,
                              google_images=google_images,
                              load_to_ram=False)

        self.__split_dataset(dataset.image_paths, dataset.targets)

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

    def __split_dataset(self,
                        image_paths: List[str],
                        targets: list) -> List[tuple]:
        total_size = sum(image_path.rsplit('/')[-1].startswith('S') for image_path in image_paths)

        if self.__google_images:
            google_image_paths = [image_path for image_path in image_paths
                                  if image_path.rsplit('/')[-1].startswith('G')]

            google_indices = [image_paths.index(google_image_path)
                              for google_image_path in google_image_paths]

            google_targets = self.__filter_list(targets, google_indices, True)

            image_paths = self.__filter_list(image_paths, google_indices, False)
            targets = self.__filter_list(targets, google_indices, False)

        most_freq_labels = self.__get_most_frequent_labels(targets)

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

        if self.__k_cross_valid > 1:
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
                    image_paths_train[i] = google_image_paths + image_paths_train[i]
                    targets_train[i] = google_targets + targets_train[i]

                image_paths_valid.append(self.__filter_list(image_paths, indices_valid[i], True))
                targets_valid.append(self.__filter_list(targets, indices_valid[i], True))

            self.__image_paths_train = image_paths_train
            self.__targets_train = targets_train

            self.__image_paths_valid = image_paths_valid
            self.__targets_valid = targets_valid
        else:
            most_freq_labels = self.__get_most_frequent_labels(targets)

            strat_split_valid = StratifiedShuffleSplit(n_splits=1,
                                                       test_size=self.__validation_size * total_size / len(image_paths),
                                                       random_state=self.__seed)

            for _, indices_valid in strat_split_valid.split([0] * len(image_paths), most_freq_labels):
                break

            indices_valid = list(indices_valid)

            self.__image_paths_train = google_image_paths + self.__filter_list(image_paths, indices_valid, False)
            self.__targets_train = google_targets + self.__filter_list(targets, indices_valid, False)

            self.__image_paths_valid = self.__filter_list(image_paths, indices_valid, True)
            self.__targets_valid = self.__filter_list(targets, indices_valid, True)

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
