LEARNING_RATE = 0.005
GRAD_CLIP = 5
TRAIN_DATAPATH = '/media/simon/NVMe/Google Drive/Maîtrise SGL CIMA+/General/FuseDataFull/'
SAVE_PATH = '/media/simon/NVMe/Google Drive/Maîtrise SGL CIMA+/General/FuseDataFull/'
RESIZED_IMAGES_SAVE_PATH = '/media/simon/NVMe/image_save_test/'
ANNOTATION_FILE = "annotation_sgl.csv"
TRAIN_TEST_SPLIT = 0.1
NO_OF_CLASSES = 10
NUM_WORKERS_RAY = 8
NUM_WORKERS_DL = 4
class_dictionary = {
    "Gould-Ferraz Shawmut A4J": 1,
    # 'Gould Shawmut A4J':1,
    "Ferraz Shawmut AJT": 2,
    "English Electric C-J": 3,
    "Ferraz Shawmut CRS": 4,
    "Gould Shawmut AJT": 5,
    "GEC HRC I-J": 6,
    "Gould Shawmut TRSR": 7,
    "English Electric Form II": 8,
    "Bussmann LPJ": 9,
    "Gould Shawmut CJ": 10,
}

"""
train_datapath
- images
- annotations
- models
"""
