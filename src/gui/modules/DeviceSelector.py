from tkinter import *
from tkinter.ttk import Radiobutton
import torch
from src.utils.constants import GUI_SETTINGS
import json

class DeviceSelector:

    def __init__(self, window) -> None:

        self.__device_option = StringVar()

        devices = [('cuda', 'cuda'),
                    ('cpu', 'cpu')]


        Label(window, text="Select device for inference").grid(row=4, column=0, padx=10, pady=10)

        for i, (option, value) in enumerate(devices):
            Radiobutton(window,
                        text=option,
                        variable=self.__device_option,
                        state=DISABLED if torch.cuda.is_available() is False else NORMAL,
                        command=self.__select,
                        value=value).grid(row=i + 5, column=0)

        with open(GUI_SETTINGS, "r") as f_obj:
            self.__device_option.set(json.load(f_obj)["device"])

    def __select(self):
        # Overwriting the settings json file
        with open(GUI_SETTINGS, "r+") as f_obj:
            settings_dict = json.load(f_obj)
            f_obj.seek(0)
            f_obj.truncate()

            settings_dict["device"] = self.__device_option.get()
            json.dump(settings_dict, f_obj)
