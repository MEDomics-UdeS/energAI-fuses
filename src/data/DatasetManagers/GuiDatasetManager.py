from src.data.Datasets.GuiDataset import GuiDataset
from src.data.DatasetManagers.CustomDatasetManager import CustomDatasetManager, ray_resize_images_for_gui
import os
from src.utils.constants import IMAGE_EXT, GUI_RESIZED_PATH, MEAN, STD
import ray
from tqdm import trange

class GuiDatasetManager(CustomDatasetManager):

    def __init__(self,
                 image_size: int,
                 images_path: str = None,
                 num_workers: int = None,
                 norm: str = 'none') -> None:
        # Check if any image exists in the image_path selected
        if any(file.endswith(f'.{IMAGE_EXT}') for file in os.listdir(images_path)):

            # Removes the content of the directory
            for file in os.listdir(GUI_RESIZED_PATH):
                if file.startswith('.') is False:
                    os.remove(f'{GUI_RESIZED_PATH}{file}')

            # Resize all images
            self._resize_images(image_size, num_workers, images_path)

        self._dataset = GuiDataset(images_path, num_workers)
        
        self._dataset.transforms = self._transforms_base(MEAN, STD)
    
    
    @property
    def dataset(self):
        return self._dataset
    

    @staticmethod
    def _resize_images(image_size: int, num_workers: int, img_path: str) -> None:
        """
            Method to resize all images in the data/raw folder and save them to the data/resized folder

            :param image_size: int, maximum image size in pixels (will be used for height & width)
            :param num_workers: int, number of workers for multiprocessing
            """
        # Initialize ray
        ray.init(include_dashboard=False)

        # Get list of image and exclude the hidden .gitkeep file
        imgs = [img for img in sorted(os.listdir(
            img_path)) if img.startswith('.') is False]

        # Create image paths
        image_paths = [os.path.join(img_path, img) for img in imgs]

        # Get dataset size
        size = len(image_paths)

        # Declare empty lists to save the resize ratios and targets
        resize_ratios = [None] * size

        # Get ray workers IDs
        if size < num_workers:
            ids = [ray_resize_images_for_gui.remote(
                image_paths, image_size, i) for i in range(size)]
        else:
            ids = [ray_resize_images_for_gui.remote(
                image_paths, image_size, i) for i in range(num_workers)]

        # Calculate initial number of jobs left
        nb_job_left = size - num_workers

        # Multiprocessing loop
        for _ in trange(size, desc='Resizing images...'):
            # Handle ray workers ready states and IDs
            ready, ids = ray.wait(ids, num_returns=1)

            # Get resize ratios and indices
            resize_ratio, idx = ray.get(ready)[0]

            # Save the resize ratios and targets to lists
            resize_ratios[idx] = resize_ratio

            # Check if there are jobs left
            if nb_job_left > 0:
                # Assign workers to the remaining tasks
                ids.extend([ray_resize_images_for_gui.remote(
                    image_paths, image_size, size - nb_job_left)])

                # Decreasing the number of jobs left
                nb_job_left -= 1

        # Shutting down ray
        ray.shutdown()

        # Displaying where files have been saved to
        print(f'\nResized images have been saved to:\t\t{GUI_RESIZED_PATH}')
