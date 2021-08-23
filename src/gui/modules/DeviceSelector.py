"""
File:
    src/gui/modules/DeviceSelector.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Class responsible of handling the device for inference
"""

from tkinter import *
import torch
from src.utils.constants import COLOR_PALETTE, FONT_PATH, GUI_SETTINGS
import json

class DeviceSelector:
    """Class responsible of handling the device for inference"""

    def __init__(self, window: Toplevel) -> None:
        """Class constructor

        Args:
            window (Toplevel): Root window

        Notes:
            If available, CUDA will always be the default option for the PyTorch device and the user
            will be able to switch for the CPU if desired

            When CUDA is not available, the buttons will be disabled with CPU as the only device option
        """

        self.__device_option = StringVar()

        devices = [('CUDA', 'cuda'),
                    ('CPU', 'cpu')]

        Label(window,
              background=COLOR_PALETTE["bg"],
              foreground=COLOR_PALETTE["fg"],
              text="Select device for inference",
              font=(FONT_PATH, 14)
              ).grid(row=4, column=0, padx=10, pady=10)

        for i, (option, value) in enumerate(devices):
            Radiobutton(window,
                        bg=COLOR_PALETTE["bg"],
                        foreground=COLOR_PALETTE["fg"],
                        highlightbackground=COLOR_PALETTE["bg"],
                        activebackground=COLOR_PALETTE["active"],
                        activeforeground=COLOR_PALETTE["fg"],
                        selectcolor=COLOR_PALETTE["bg"],
                        text=option,
                        font=(FONT_PATH, 12),
                        variable=self.__device_option,
                        state=DISABLED if torch.cuda.is_available() is False else NORMAL,
                        command=self.__select,
                        value=value
                        ).grid(row=i + 5, column=0)

        # Reading the settings JSON file
        with open(GUI_SETTINGS, "r") as f_obj:
            self.__device_option.set(json.load(f_obj)["device"])

    def __select(self) -> None:
        """Selects a device option"""
        
        # Overwriting the settings json file
        with open(GUI_SETTINGS, "r+") as f_obj:
            settings_dict = json.load(f_obj)
            f_obj.seek(0)
            f_obj.truncate()

            settings_dict["device"] = self.__device_option.get()
            json.dump(settings_dict, f_obj)

    @property
    def device_option(self):
        """Get the device option widget"""
        
        return self.__device_option
