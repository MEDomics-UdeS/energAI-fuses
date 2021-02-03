LEARNING_RATE = 0.005
TRAIN_DATAPATH = '/data/energAI/FuseDataFull/'
SAVE_PATH = '/home/local/USHERBROOKE/kuls3101/fuse_detection/'
ANNOTATION_FILE = "annotation_updated.csv"
train_test_split_index = 200
NO_OF_CLASSES = 10
class_dictionary = {
    "Gould-Ferraz Shawmut A4J":1,
    # 'Gould Shawmut A4J':1,
    "Ferraz Shawmut AJT":2,
    "English Electric C-J":3,
    "Ferraz Shawmut CRS":4,
    "Gould Shawmut AJT":5,
    "GEC HRC I-J":6,
    "Gould Shawmut TRSR":7,
    "English Electric Form II":8,
    "Bussmann LPJ":9,
    "Gould Shawmut CJ":10,
}

"""
train_datapath
- images
- annotations
- models
"""