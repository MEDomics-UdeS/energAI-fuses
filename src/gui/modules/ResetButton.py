import json
from tkinter import *
from src.utils.constants import GUI_SETTINGS, COLOR_PALETTE, FONT_PATH
from src.utils.helper_functions import enter_default_json

class ResetButton:

    def __init__(self, window, model, imgdir, iou, score, device, gt_json) -> None:

        Button(window,
               background=COLOR_PALETTE["widgets"],
               foreground=COLOR_PALETTE["fg"],
               activebackground=COLOR_PALETTE["active"],
               activeforeground=COLOR_PALETTE["fg"],
               highlightbackground=COLOR_PALETTE["active"],
               text="Restore defaults",
               font=(FONT_PATH, 12),
               command=lambda: self.restore_defaults(model, imgdir, iou, score, device, gt_json)).grid(row=3, column=1)

    def restore_defaults(self, model, imgdir, iou, score, device, gt_json):

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
        imgdir.img_dir_label.config(foreground=COLOR_PALETTE["red"],
                                    text="No image directory selected")

        # Restoring the defaults advanced options
        iou.slider.set(settings_dict["iou_treshold"])
        score.slider.set(settings_dict["score_treshold"])
        device.device_option.set(settings_dict["device"])
        #TODO add gtjson file to defaults
        
