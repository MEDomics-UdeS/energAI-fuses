from tkinter import *
from tkinter import filedialog
import json
from src.utils.constants import GUI_SETTINGS

class JsonFileLoader:

    def __init__(self, window) -> None:

        Label(window, text="Ground truth file").grid(row=0, column=1, padx=10, pady=10)
        model_button = Button(window, text="Select", command=lambda: self.__select_file(window))
        model_button.grid(row=1, column=1, padx=10)

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
