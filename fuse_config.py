EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 0.005
TRAIN_DATAPATH = 'dataset_partial\\gould\\'
train_test_split_index = 10
NO_OF_CLASSES = 10
class_dictionary = {
    # "Gould-Ferraz Shawmut A4J":1,
    'Gould Shawmut A4J':1,
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