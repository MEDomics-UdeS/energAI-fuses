from tkinter import *
import torch
from src.utils.constants import COLOR_PALETTE, FONT_PATH, GUI_SETTINGS
import json

class DeviceSelector:

    def __init__(self, window: Toplevel) -> None:

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

        with open(GUI_SETTINGS, "r") as f_obj:
            self.__device_option.set(json.load(f_obj)["device"])

    def __select(self) -> None:
        # Overwriting the settings json file
        with open(GUI_SETTINGS, "r+") as f_obj:
            settings_dict = json.load(f_obj)
            f_obj.seek(0)
            f_obj.truncate()

            settings_dict["device"] = self.__device_option.get()
            json.dump(settings_dict, f_obj)

    @property
    def device_option(self):
        return self.__device_option