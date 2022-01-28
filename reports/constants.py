"""
File:
    reports/constants.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    File containing constants used for results parsing
"""

from os.path import join

PATH = '/home/simon/Desktop/Results_Fuses'

PATH_A = join(PATH, 'A')
PATH_C = join(PATH, 'C')

RESULTS_B_DICT = {
    'image_size': join(PATH, 'B1'),
    'no_pretrained': join(PATH, 'B2'),
    'no_google_images': join(PATH, 'B3')
}

AP_DICT = {
    'AP @ [IoU=0.50:0.95 | area=all | maxDets=100]': 'AP',
    'AP @ [IoU=0.50 | area=all | maxDets=100]': 'AP_{50}',
    'AP @ [IoU=0.75 | area=all | maxDets=100]': 'AP_{75}',
    'AP @ [IoU=0.50:0.95 | area=small | maxDets=100]': 'AP_{S}',
    'AP @ [IoU=0.50:0.95 | area=medium | maxDets=100]': 'AP_{M}',
    'AP @ [IoU=0.50:0.95 | area=large | maxDets=100]': 'AP_{L}'
}

PHASES_LIST = ['Validation', 'Testing']
