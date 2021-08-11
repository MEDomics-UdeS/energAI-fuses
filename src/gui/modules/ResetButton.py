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

    def __init__(self, 
                 window: Toplevel, 
                 model: ModelLoader, 
                 imgdir: ImageLoader, 
                 iou: IOUSlider, 
                 score: ScoreSlider, 
                 device: DeviceSelector, 
                 gt_json: CSVFileLoader) -> None:

        Button(window,
               background=COLOR_PALETTE["widgets"],
               foreground=COLOR_PALETTE["fg"],
               activebackground=COLOR_PALETTE["active"],
               activeforeground=COLOR_PALETTE["fg"],
               highlightbackground=COLOR_PALETTE["active"],
               text="Restore defaults",
               font=(FONT_PATH, 12),
               command=lambda: self.restore_defaults(model, imgdir, iou, score, device, gt_json)
               ).grid(row=6, column=1, pady=10)

    def restore_defaults(self,
                         model: ModelLoader,
                         imgdir: ImageLoader,
                         iou: IOUSlider,
                         score: ScoreSlider,
                         device: DeviceSelector,
                         gt_json: CSVFileLoader) -> None:

        answer = askokcancel(title="Confirm reset",
                             message="Are you sure you want to reset the settings?",
                             icon=WARNING)
        
        if answer:
            
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
            gt_json.reset()
        
