import glob
import json
from shutil import copy
from PIL import Image


def create_json_file(source_file: str, target_file: str, root_path: str, start: int, end: int) -> None:
    """Creates a DE:TR compliant json file from a source json file

    Args:
        source_file (str): source json file from containing the fuses infos
        target_file (str): absolute path destination of the new json file
        root_path (str): absolute path of the jpg images for the fuse dataset
        start (int): starting image id for the new data subset
        end (int): ending image id for the new data subset
    """
    
    categories = [
        {
            "supercategory": "fuse",
            "name": "Gould-Ferraz Shawmut A4J",
            "id": 0
        },
        {
            "supercategory": "fuse",
            "name": "Ferraz Shawmut AJT",
            "id": 1
        },
        {
            "supercategory": "fuse",
            "name": "English Electric C-J",
            "id": 2
        },
        {
            "supercategory": "fuse",
            "name": "Ferraz Shawmut CRS",
            "id": 3
        },
        {
            "supercategory": "fuse",
            "name": "Gould Shawmut AJT",
            "id": 4
        },
        {
            "supercategory": "fuse",
            "name": "GEC HRC I-J",
            "id": 5
        },
        {
            "supercategory": "fuse",
            "name": "Gould Shawmut TRSR",
            "id": 6
        },
        {
            "supercategory": "fuse",
            "name": "English Electric Form II",
            "id": 7
        },
        {
            "supercategory": "fuse",
            "name": "Bussmann LPJ",
            "id": 8
        },
        {
            "supercategory": "fuse",
            "name": "Gould Shawmut CJ",
            "id": 9
        }
    ]


    res_file = {
        "categories": categories,
        "images": [],
        "annotations": []
    }

    image_id = 0

    for filename in sorted(glob.glob(root_path + '*.jpg')):
        img = Image.open(filename)
        img_w, img_h = img.size
        img_elem = {"file_name": filename,
                    "height": img_h,
                    "width": img_w,
                    "id": image_id}

        res_file["images"].append(img_elem)

        image_id += 1

    annotations = create_annotations(json_file=source_file, start=start, end=end)
    res_file['annotations'] = annotations

    with open(target_file, "w") as f:
        json_str = json.dumps(res_file)
        f.write(json_str)
        
def create_annotations(json_file: str, start:int, end: int) -> list:
    """Create custom annoations for the DE:TR json files by parsing the source json file

    Args:
        json_file (str): source json file used for parsing
        start (int): starting image id for the new data subset
        end (int): ending image id for the new data subset

    Returns:
        list: annotations used by the DE:TR model for every picture in the data subset
    """

    id = 0
    image_id = 0
    annotations = []

    with open(json_file) as f:
        data = json.load(f)

        # Creation of the training.json file
        for entry in data[start:end]:

            for i, box in enumerate(entry['boxes']):
                bbox = [box[0], box[1], box[2] - box[0], box[3] - box[1]]

                ann = {
                    "id": id,
                    "bbox": bbox,
                    "segmentation": [
                        [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                         [bbox[0], bbox[1] + bbox[3]]]],
                    "image_id": image_id,
                    "ignore": 0,
                    "category_id": entry['labels'][i],
                    "iscrowd": 0,
                    "area": (box[2] - box[0]) * (box[3] - box[1])
                }

                id += 1
                annotations.append(ann)

            image_id += 1
            
    return annotations


def calculate_datasets_indexes(source_file: str, train_size: float = 0.8, valid_size: float = 0.1) -> dict:
    """Function used to calculate the indexes used when splitting the images in the appropriate directories.

    Args:
        source_file (str): original json file containing every fuse images
        train_size (float, optional): size percentage of the fuse dataset to be used as the trainig set. Defaults to 0.8.
        valid_size (float, optional): size percentage of the fuse dataset to be used as the validation set. Defaults to 0.1.

    Returns:
        dict: indexes for the different sets to know how many pictures to copy in the right directory
    """
    
    datasets_indexes = {
        
        "training_start": int,
        "training_end": int,
        "valid_start": int,
        "valid_end": int
    }
    
    with open(source_file) as f:
        
        data = json.load(f)
        
        datasets_indexes["training_start"] = 0
        datasets_indexes["training_end"] = int(len(data) * train_size) - 1
        datasets_indexes["valid_start"] = int(len(data) * train_size)
        datasets_indexes["valid_end"] = int(len(data) * (train_size + valid_size)) - 1
        
    return datasets_indexes


def split_pictures(root_path: str, train_path: str, valid_path: str, test_path: str, indexes: dict) -> None:
    """Function used to split the .jpg images in the data/resized directory to be used by the DE:TR model.
    Currently stores images in an ordered fashion in the resized direcotory.

    Args:
        root_path (str): path to the resized images to be split
        train_path (str): path to the training directory for the DE:TR model
        valid_path (str): path to the validation directory for the DE:TR model
        test_path (str): path to the test directory for the DE:TR model
        indexes (dict): indexes for the different sets to know how many pictures to copy in the right directory
    """
    
    for i, filename in enumerate(sorted(glob.glob(root_path + '*.jpg'))):
        
        if indexes["training_start"] <= i <= indexes["training_end"]:
            copy(src=filename, dst=train_path)
            
        elif indexes["valid_start"] <= i <= indexes["valid_end"]:
            copy(src=filename, dst=valid_path)
            
        else:
            copy(src=filename, dst=test_path)


if __name__ == '__main__':
    """Building the fuse dataset from the resized pictures to be compliant with the DE:TR model.
    Creates a reformated training.json and validation.json from the targets_resized.json
    """

    source_file = "/home/simon/Documents/GitHub/energAI-fuses/data/annotations/targets_resized.json"
    root_path = "/home/simon/Documents/GitHub/energAI-fuses/data/resized/"
    train_path = "/home/simon/Documents/GitHub/energAI-fuses/src/detr/dataset/train/"
    valid_path = "/home/simon/Documents/GitHub/energAI-fuses/src/detr/dataset/val/"
    test_path = "/home/simon/Documents/GitHub/energAI-fuses/src/detr/dataset/test/"
    train_json = "/home/simon/Documents/GitHub/energAI-fuses/src/detr/dataset/training.json"
    valid_json = "/home/simon/Documents/GitHub/energAI-fuses/src/detr/dataset/validation.json"

    dataset_indexes = calculate_datasets_indexes(source_file=source_file, train_size=0.8, valid_size=0.1)

    split_pictures(root_path=root_path, train_path=train_path,
                   valid_path=valid_path, test_path=test_path, indexes=dataset_indexes)

    # Creation of the two json files
    create_json_file(source_file=source_file, target_file=train_json,
                     root_path=train_path, start=dataset_indexes["training_start"], end=dataset_indexes["training_end"])

    create_json_file(source_file=source_file, target_file=valid_json,
                     root_path=valid_path, start=dataset_indexes["valid_start"], end=dataset_indexes["valid_end"])

    print("Done.")
