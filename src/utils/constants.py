"""
File:
    src/utils/constants.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Constants used in the project
"""

# Environment constant
REQUIRED_PYTHON = "python3"

# Image file paths
RAW_PATH = 'data/raw/'
LEARNING_PATH = 'data/raw/learning/'
HOLDOUT_PATH = 'data/raw/holdout/'
RESIZED_PATH = 'data/resized/'
INFERENCE_PATH = 'data/inference/'
GUI_RESIZED_PATH = 'data/gui_resized/'

# Annotations file paths
TARGETS_PATH = 'data/annotations/targets_resized.json'
GUI_TARGETS_PATH = 'data/annotations/gui_resized.json'
ANNOTATIONS_PATH = 'data/annotations/annotations_raw.csv'

# GUI application settings
GUI_SETTINGS = "src/gui/gui_settings.json"

# Models file path
MODELS_PATH = 'saved_models/'

# Logging file path for tensorboard
LOG_PATH = 'logdir/'

# Font file path for inference when drawing bounding boxes confidence scores
FONT_PATH = '/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf'

# Google Drive file ID Hashes
IMAGES_ID = '1U8uE7a1tEOXpj4W0F1rV87ffeG_NfW60'
ANNOTATIONS_ID = '1FdFLlb-RJY9IZ2zHLScTrwWlT4EPB2Oj'

# Dataset mean and standard deviation per channel (R, G, B)
MEAN = (0.6221349083958952, 0.6097851650674193, 0.592938912173587)
STD = (0.33986575693672466, 0.3446239730624245, 0.3541046569741213)

# Colot palette of the GUI. Uses the Dracula color palette https://draculatheme.com/
COLOR_PALETTE = {
    "bg": "#282a36",
    "fg": "#f8f8f2",
    "active": "#6272a4",
    "widgets": "#44475a",
    "cyan": "#8be9fd",
    "green": "#50fa7b",
    "orange": "#ffb86c",
    "pink": "#ff79c6",
    "purple": "#bd93f9",
    "red": "#ff5555",
    "yellow": "#f1fa8c"
}

COCO_PARAMS_LIST = [
    'AP @ [IoU=0.50:0.95 | area=all | maxDets=100]',
    'AP @ [IoU=0.50 | area=all | maxDets=100]',
    'AP @ [IoU=0.75 | area=all | maxDets=100]',
    'AP @ [IoU=0.50:0.95 | area=small | maxDets=100]',
    'AP @ [IoU=0.50:0.95 | area=medium | maxDets=100]',
    'AP @ [IoU=0.50:0.95 | area=large | maxDets=100]',
    'AR @ [IoU=0.50:0.95 | area=all | maxDets=1]',
    'AR @ [IoU=0.50:0.95 | area=all | maxDets=10]',
    'AR @ [IoU=0.50:0.95 | area=all | maxDets=100]',
    'AR @ [IoU=0.50:0.95 | area=small | maxDets=100]',
    'AR @ [IoU=0.50:0.95 | area=medium | maxDets=100]',
    'AR @ [IoU=0.50:0.95 | area=large | maxDets=100]'
]

"""
Original Console Outputs:

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.255
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.454
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.271
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.077
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.245
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.280
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.181
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.513
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.518
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.098
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.457
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.564
 """

# Evaluation metric for early stopping and best model saving
EVAL_METRIC = COCO_PARAMS_LIST[0]

# Image extension
IMAGE_EXT = 'jpg'

# Holdout set size
HOLDOUT_SIZE = 0.1

# Class dictionary
CLASS_DICT = {
    "Background":               0,
    "Gould-Ferraz Shawmut A4J": 1,
    "Ferraz Shawmut AJT":       2,
    "English Electric C-J":     3,
    "Ferraz Shawmut CRS":       4,
    "Gould Shawmut AJT":        5,
    "GEC HRC I-J":              6,
    "Gould Shawmut TRSR":       7,
    "English Electric Form II": 8,
    "Bussmann LPJ":             9,
    "Gould Shawmut CJ":         10
}

# OCR dictionary with keywords related to each class
OCR_DICT = {
    "Gould-Ferraz Shawmut A4J": ['Gould', 'Shawmut', 'Amp-trap', 'Class J', 'Current', 'Limiting', 'A4J###',
                                 '### Amps.', '### Amp.', '### Amp', '600 VAC or Less', 'HRC I-J', 'UND. LAB. INC.',
                                 'LISTED FUSE', 'Gould Inc.', 'Newburyport, MA', 'Gould Shawmut', 'Toronto, Canada',
                                 '600VAC', '200kA I.R.', '300VDC 20kA I.R.', 'Ferraz Shawmut', 'Certified',
                                 'Assy. in Mexico', 'Assy. in CANADA', '200kA IR 600V AC', '20kA IR 300V DC',
                                 '600 V.A.C. or Less', 'Interrupting Rating', '200,000 RMS. SYM. Amps.',
                                 'Assembled in Mexico', 'IR 200kA', 'Electronics Inc.', 'U.S. Pat. Nos. 4,216,457:',
                                 '4,300,281 and 4,320,376', 'Newburyport, MA 01950', '600 V ~', '200k A I.R.',
                                 'PATENTED', 'Ferraz', '200kA IR AC', '300VDC or Less', '100kA IR DC', 'LISTED', 'FUSE',
                                 'SA'],
    "Ferraz Shawmut AJT": ['AMP-TRAP', '2000', 'DUAL ELEMENT', 'TIME DELAY', 'Smart', 'SpOt', 'AJT###', '###A',
                           '600V AC', '500V DC', 'Ferraz', 'Shawmut', 'Any Red - Open', 'Amp-Trap 2000', 'Dual Element',
                           'Time Delay', 'Mersen', 'Ferraz Shawmut', 'Newburyport, MA 01950', 'Class J Fuse',
                           '200,000A IR 600V AC', '100,000A IR 500V DC', 'Current Limiting', 'LISTED', 'FUSE',
                           'ISSUE NO.: ND57-62', 'Ferraz Shawmut Certified', '300,000A IR 600V AC', '600V AC/500V DC',
                           'Any Red = Open'],
    "English Electric C-J": ['ENGLISH', 'ELECTRIC', 'HRC', 'ENERGY', 'LIMITING FUSE', '### AMPS',
                             '600 VOLTS A.C. OR LESS', 'CATALOG No. C###J', 'ENSURE CONTINUED', 'SAFE PROTECTION',
                             'REPLACE WITH', 'CATALOG No.', 'Made in England', 'CSA SPEC', 'CSA STD.', 'HRC1',
                             'C22.2-106', 'TESTED AT', '200,000 AMPS', 'EASTERN ELECTRIC', '5775 Ferrier Street.',
                             'Montreal PQ Canada'],
    "Ferraz Shawmut CRS": ['Ferraz', 'Shawmut', 'Type D', 'Time Delay', 'Temporise', 'CRS###amp', '600V A.C. or less',
                           'C.A. ou moins', '10kA IR', 'Ferraz Shawmut', 'Toronto, Canada', 'cefcon', 'LR14742',
                           'Mexico'],
    "Gould Shawmut AJT": ['AMP-TRAP', '2000', 'TIME DELAY', 'AJT###', '##A 600V AC', 'GOULD SHAWMUT', 'HRCI-J',
                          'UND. LAB. INC.', 'LISTED FUSE', 'U.S. PAT. NO. 4,344,058', 'Dual Element', 'Time Delay',
                          '## Amp.', '600V AC', '500V DC', 'DUAL ELEMENT', 'Class J Fuse', '200,000A IR 600V AC',
                          '100,000A IR 500V DC', 'Current Limiting', 'Gould Certified', '300,000A IR 600V AC',
                          'Gould Shawmut', '(508) 462-6662', 'Gould, Inc.', 'Newburyport, Mass., U.S.A.',
                          'Toronto, Ontario, Canada', 'Made in U.S.A.', 'U.S. Pat. 4,320,376', 'Nos. 4,300,281'],
    "GEC HRC I-J": ['GEC', 'HRC I-J', 'Rating', '### Amp', 'CSA', 'C22.2', 'No. 106-M1985', 'IR 200kA', '~ 60Hz', '600',
                    'VOLTS', 'Can. Pat. No. 148995', '188', 'Made in English', 'C###J', 'Cat No.', 'No. 106-M92',
                    'GEC ALSTHOM'],
    "Gould Shawmut TRSR": ['GOULD', 'Shawmut', 'Tri-onic', 'TRSR ###', 'Time Delay', 'Temporisé', 'HRCI-R', '###A',
                           'LR14742', '600V ~', '200k.A.I.R', 'Dual Element', '600 V AC', '600 V DC', '300V DC',
                           '600V AC', 'Current Limiting', 'Class RK5 Fuse', 'UND. LAB. INC.', 'LISTED FUSE',
                           '200.000A IR', '20.000A IR', 'Gould Shawmut', '198L', '(508) 462-6662', 'Action',
                           'Temporisée', 'HRC I', '600V A.C. or less', 'C.A. ou moins', 'TRS###R', '### Amps',
                           '600 VAC or Less'],
    "English Electric Form II": ['THE CAT. No.', 'AND RATING', '(MARKED ON THIS CAP)', 'SHOULD BE QUOTED',
                                 'WHEN RE-ORDERING', 'ENGLISH', 'ELECTRIC', 'TESTED AT', '200,000 Amps', 'FORM II',
                                 'H.R.C. FUSE', 'SA', 'C.S.A.Spec.C22-2No.106', 'EASTERN ELECTRIC COMPANY LTD.', '600',
                                 'VOLTS', 'or less', 'A.C. 60 cycle', 'EASTERN ELECTRIC FUSE PATENTED', 'CF###A',
                                 'CC.###', 'CAT.NO.CC###.', 'Complies with', 'IEC 269-2', 'CSA STD', 'C22-2', 'No 106',
                                 'Tested at', '200,000 Amps', '600V (or less)', 'AC 60HZ', '100,000 AMP RMS ASYM',
                                 'C.S.A. APP. N°12203.', '600V. 60 CYCLE A.C.', 'FORM II.H.R.C.FUSE'],
    "Bussmann LPJ": ['BUSS', 'LOW-PEAK', 'DUAL-ELEMENT TIME-DELAY', 'FUSE', 'LPJ-###SP', '600 VAC OR LESS',
                     'CURRENT LIMITING', 'AMP', 'THIS FUSE MAY SUBSTITUTE FOR', 'A LISTED CLASS J FUSE', 'HRCI-J',
                     'IR 200kA AC', 'IR 100kA DC', 'TYPE D', 'UL LISTED', 'SPECIAL PURPOSE FUSE FP33-##',
                     'IR 300kA AC, IR 100kA DC', '600 VAC', '300 VDC', 'CLASS J', 'LISTED FUSE DL92-##', 'Bussmann LPJ',
                     'LOW-PEAK', 'ULTIMATE PROTECTION', 'CLASS J FUSE', '600Vac', 'AC IR 300kA', '300Vdc',
                     'DC IR 100kA', 'Self-certified DC rating', 'Cooper Bussmann, LLC', 'St. Louis, MO 63178',
                     'Assembled in Mexico', 'www.bussmann.com', 'Cooper Industries', 'Bussmann Division',
                     'St. Louis, MO', 'MADE IN U.S.A.', 'LISTED SPECIAL PURPOSE'],
    "Gould Shawmut CJ": ['GOULD', 'Shawmut', 'CJ ###', 'HRCI-J', '###A', 'LR14742', 'Class J', 'Cat. No.', '### Amps',
                         'Amp-trap', '600 V.A.C. or less', '200,000 Amps A.C.', 'Interrupting Rating',
                         'Current Limiting', '600 V.C.A. ou moins', '200,000 Amps C.A.', 'Intensité de Rupture',
                         'Limitant de Courant', '200,000 A.I.R.', 'Mfd. By/Fab. Par', 'Gould Shawmut',
                         'Toronto, Canada', 'Int. De Rupt.', 'Int. Rating', 'Gould Elec. Fuse Div.', 'HRC I',
                         '200k A.I.R.', '600V ~']
}
