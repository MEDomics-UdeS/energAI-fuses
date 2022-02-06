"""
File:
    src/gui/modules/IOUSlider.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Class responsible of handling the IoU value for inference
"""

import json
from tkinter import *

from src.utils.constants import GUI_SETTINGS, FONT_PATH, COLOR_PALETTE


class IOUSlider:
    """Class responsible of handling the IoU value for inference"""
    def __init__(self,
                 window: Toplevel) -> None:
        """Class constructor

        Args:
            window(Toplevel): Root window
        
        Notes:
            By default, the IoU will be set to 0.5
            
        """

        # Creating the sliding button widget
        Label(window,
              background=COLOR_PALETTE["bg"],
              foreground=COLOR_PALETTE["fg"],
              text="Select the IoU threshold",
              font=(FONT_PATH, 14),
              width=30
              ).grid(row=0, column=0, padx=10, pady=10)
        
        self.__slider = Scale(window,
                              background=COLOR_PALETTE["widgets"],
                              foreground=COLOR_PALETTE["fg"],
                              activebackground=COLOR_PALETTE["active"],
                              highlightbackground=COLOR_PALETTE["active"],
                              troughcolor=COLOR_PALETTE["bg"],
                              font=(FONT_PATH, 12),
                              from_=0.1,
                              to=1.0,
                              orient=HORIZONTAL,
                              resolution=0.1,
                              command=self.__slide)

        # Loading the current saved value for score threshold
        with open(GUI_SETTINGS, "r") as f_obj:
            self.__slider.set(json.load(f_obj)["iou_threshold"])

        # Putting the widget on screen
        self.__slider.grid(row=1, column=0)

    @staticmethod
    def __slide(value: DoubleVar) -> None:
        """Updates the settings JSON file when the user interacts with the slider

        Args:
            value(DoubleVar): Current value of the slider

        """
        
        # Overwriting the settings json file
        with open(GUI_SETTINGS, "r+") as f_obj:
            settings_dict = json.load(f_obj)
            f_obj.seek(0)
            f_obj.truncate()

            settings_dict["iou_threshold"] = value
            json.dump(settings_dict, f_obj)

    @property
    def slider(self) -> Scale:
        """Get the slider widget"""
        
        return self.__slider
