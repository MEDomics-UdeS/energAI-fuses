"""
File:
    src/gui/modules/ResetButton.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Class responsible to reset the settings JSON file and the GUI's state
"""

import json
from tkinter import *
from tkinter.messagebox import WARNING, askokcancel

from src.utils.constants import GUI_SETTINGS, COLOR_PALETTE, FONT_PATH
from src.utils.helper_functions import enter_default_json
from src.gui.modules.ModelLoader import ModelLoader
from src.gui.modules.ImageLoader import ImageLoader
from src.gui.modules.IOUSlider import IOUSlider
from src.gui.modules.ScoreSlider import ScoreSlider
from src.gui.modules.DeviceSelector import DeviceSelector
from src.gui.modules.CSVFileLoader import CSVFileLoader


class ResetButton:
    """Class responsible to reset the settings JSON file and the GUI's state"""
    def __init__(self, 
                 window: Toplevel, 
                 model: ModelLoader, 
                 img_dir: ImageLoader, 
                 iou: IOUSlider, 
                 score: ScoreSlider, 
                 device: DeviceSelector, 
                 gt_csv: CSVFileLoader) -> None:
        """Class constructor

        Args:
            window(Toplevel): Root window
            model(ModelLoader): Object containing the trained model file path
            img_dir(ImageLoader): Object containing the raw image directory path
            iou(IOUSlider): Object containing the IoU value
            score(ScoreSlider): Object containing the score treshold value
            device(DeviceSelector): Object containing the selected PyTorch device
            gt_csv(CSVFileLoader): Object containing the ground truth CSV file

        """

        Button(window,
               background=COLOR_PALETTE["widgets"],
               foreground=COLOR_PALETTE["fg"],
               activebackground=COLOR_PALETTE["active"],
               activeforeground=COLOR_PALETTE["fg"],
               highlightbackground=COLOR_PALETTE["active"],
               text="Restore defaults",
               font=(FONT_PATH, 12),
               command=lambda: self.restore_defaults(model, img_dir, iou, score, device, gt_csv)
               ).grid(row=6, column=1, pady=10)

    @staticmethod
    def restore_defaults(model: ModelLoader,
                         img_dir: ImageLoader,
                         iou: IOUSlider,
                         score: ScoreSlider,
                         device: DeviceSelector,
                         gt_csv: CSVFileLoader) -> None:
        """Restores the GUI's widgets to default values

        Args:
            model(ModelLoader): Object containing the trained model file path
            img_dir(ImageLoader): Object containing the raw image directory path
            iou(IOUSlider): Object containing the IoU value
            score(ScoreSlider): Object containing the score treshold value
            device(DeviceSelector): Object containing the selected PyTorch device
            gt_csv(CSVFileLoader): Object containing the ground truth CSV file

        """

        # Asking for user confirmation
        answer = askokcancel(title="Confirm reset",
                             message="Are you sure you want to reset the settings?",
                             icon=WARNING)
        
        if answer:
            # Reading the settings JSON file
            with open(GUI_SETTINGS, "r+") as f_obj:
                f_obj.seek(0)
                f_obj.truncate()
                enter_default_json(f_obj)
                f_obj.seek(0)

                # Loading the settings dictionnary
                settings_dict = json.load(f_obj)

            # Resetting the widgets to defaults value
            model.model_label.config(foreground=COLOR_PALETTE["red"],
                                     text="No model selected")
            img_dir.img_dir_label.config(foreground=COLOR_PALETTE["red"],
                                         text="No image directory selected")

            # Restoring the defaults advanced options
            iou.slider.set(settings_dict["iou_treshold"])
            score.slider.set(settings_dict["score_treshold"])
            device.device_option.set(settings_dict["device"])
            gt_csv.reset()
