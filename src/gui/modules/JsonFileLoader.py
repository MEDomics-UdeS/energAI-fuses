from tkinter import *
from tkinter import filedialog
import json
from src.utils.constants import COLOR_PALETTE, FONT_PATH, GUI_SETTINGS

class JsonFileLoader:

    def __init__(self, window) -> None:

        Label(window, 
              background=COLOR_PALETTE["bg"],
              foreground=COLOR_PALETTE["fg"],
              text="Ground truth file",
              font=(FONT_PATH, 14),
              width=30
              ).grid(row=0, column=1, padx=10, pady=10)
        
        Button(window,
               background=COLOR_PALETTE["widgets"],
               foreground=COLOR_PALETTE["fg"],
               activebackground=COLOR_PALETTE["active"],
               activeforeground=COLOR_PALETTE["fg"],
               highlightbackground=COLOR_PALETTE["active"],
               text="Select",
               font=(FONT_PATH, 12),
               command=lambda: self.__select_file(window)
               ).grid(row=1, column=1, padx=10)

    def __select_file(self, window):
        window.filename = filedialog.askopenfile(
            initialdir=".", title="Select a ground truth JSON file")

        if window.filename:
            # Overwriting the settings json file
            with open(GUI_SETTINGS, "r+") as f_obj:
                settings_dict = json.load(f_obj)
                f_obj.seek(0)
                f_obj.truncate()

                settings_dict["ground_truth"] = window.filename.name
                json.dump(settings_dict, f_obj)
