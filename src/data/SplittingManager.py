from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
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

        dataset = FuseDataset(images_path=RESIZED_LEARNING_PATH,
                              targets_path=TARGETS_LEARNING_PATH,
                              num_workers=num_workers,
                              google_images=google_images,
                              load_to_ram=False)

        image_paths_train = dataset.image_paths
        targets_tain = dataset.targets

        # image_paths_test =

        self.__train_image_paths = []
        self.__valid_image_paths = []
        self.__test_image_paths = []

        if k_cross_valid == 1:
            self.__split_dataset(dataset.image_paths, dataset.targets)
        else:
            for k in range(k_cross_valid):
                pass

    @property
    def train_indices_list(self) -> list:
        return self.__train_indices_list

    @property
    def valid_indices_list(self) -> list:
        return self.__valid_indices_list

    @property
    def test_indices_list(self) -> list:
        return self.__test_indices_list

    def __split_dataset(self,
                        image_paths_in: List[str],
                        targets_in: list,
                        image_paths_out: List[str],
                        targets_out: list,
                        split_size: float,
                        total_size: int) -> List[int]:
        """
        Split a dataset into two sub-datasets, used to create the validation and testing dataset splits

        :param dataset_in: FuseDataset, input dataset from which to extract data
        :param dataset_out: FuseDataset, output dataset into which we insert data
        :param split_size: float, size of the split
        :param total_size: int, total size of the original dataset
        :return: tuple of two FuseDataset objects, which are the dataset_in and dataset_out after splitting
        """

        split_size = split_size / (1 - HOLDOUT_SIZE)

        if 0 < split_size < 1:
            if self._google_images:
                google_image_paths = [image_path for image_path in self.__dataset.image_paths
                                      if image_path.rsplit('/')[-1].startswith('G')]

                google_indices = [self.__dataset.image_paths.index(google_image_path)
                                  for google_image_path in google_image_paths]

                image_paths = [image_paths[i] for i in range(len(image_paths)) if i not in google_indices]
                targets = [targets[i] for i in range(len(targets)) if i not in google_indices]

            most_freq_labels = [max(set(target['labels'].tolist()), key=target['labels'].tolist().count)
                                for target in self.__dataset.targets]

            strat_split = StratifiedShuffleSplit(n_splits=1,
                                                 test_size=split_size * total_size / len(self.__dataset),
                                                 random_state=self._seed)

            # Iterate through generator object once and break
            for _, indices in strat_split.split([0] * len(self.__dataset), most_freq_labels):
                break

            # # Extract the image paths, images and targets from dataset_in
            # image_paths, _, targets = self.__dataset.extract_data(index_list=indices)
            #
            # # Insert the extracted image paths, images and targets into dataset_out
            # dataset_out.add_data(image_paths, images, targets)
            #
            # if self._google_images:
            #     dataset_in.add_data(google_image_paths, google_images, google_targets)
            #
            #
            return (image_paths_in, targets_in), (image_paths_out, targets_out)
        else:
            return (self.__dataset.image_paths, self.__dataset.targets), (None, None)