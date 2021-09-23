RESULTS_B_DICT = {
    'image_size': '/home/simon/Desktop/Results_Fuses/B1',
    'no_pretrained': '/home/simon/Desktop/Results_Fuses/B2',
    'no_google_images': '/home/simon/Desktop/Results_Fuses/B3'
}

SCALARS_B_DICT = {
    'hparams/Validation/AP @ [IoU=0.50:0.95 | area=all | maxDets=100]': 'AP',
    'hparams/Validation/AP @ [IoU=0.50 | area=all | maxDets=100]': 'AP_{50}',
    'hparams/Validation/AP @ [IoU=0.75 | area=all | maxDets=100]': 'AP_{75}',
    'hparams/Validation/AP @ [IoU=0.50:0.95 | area=small | maxDets=100]': 'AP_{S}',
    'hparams/Validation/AP @ [IoU=0.50:0.95 | area=medium | maxDets=100]': 'AP_{M}',
    'hparams/Validation/AP @ [IoU=0.50:0.95 | area=large | maxDets=100]': 'AP_{L}'
}
