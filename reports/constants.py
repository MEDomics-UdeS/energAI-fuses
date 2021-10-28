PATH = '/home/simon/Desktop/Results_Fuses'

PATH_A = PATH + '/A'
PATH_C = PATH + '/C'

RESULTS_B_DICT = {
    'image_size': '/home/simon/Desktop/Results_Fuses/B1',
    'no_pretrained': '/home/simon/Desktop/Results_Fuses/B2',
    'no_google_images': '/home/simon/Desktop/Results_Fuses/B3'
}

AP_DICT = {
    'AP @ [IoU=0.50:0.95 | area=all | maxDets=100]': 'AP',
    'AP @ [IoU=0.50 | area=all | maxDets=100]': 'AP_{50}',
    'AP @ [IoU=0.75 | area=all | maxDets=100]': 'AP_{75}',
    'AP @ [IoU=0.50:0.95 | area=small | maxDets=100]': 'AP_{S}',
    'AP @ [IoU=0.50:0.95 | area=medium | maxDets=100]': 'AP_{M}',
    'AP @ [IoU=0.50:0.95 | area=large | maxDets=100]': 'AP_{L}'
}

SCALARS_B_DICT = {}
SCALARS_C_DICT = {}

for key, value in AP_DICT.items():
    SCALARS_B_DICT['hparams/Validation/' + key] = value
    SCALARS_C_DICT['hparams/Testing/' + key] = value
