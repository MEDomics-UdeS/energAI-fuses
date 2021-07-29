from tkinter import *
import json
from src.utils.constants import GUI_SETTINGS


class ScoreSlider:

    def __init__(self, window) -> None:
        Label(window, text="Select the score treshold").grid(row=2, column=0, padx=10, pady=10)

        # Creating the sliding button widget
        slider = Scale(window, from_=0.1, to=1.0, orient=HORIZONTAL, resolution=0.1,
              command=self.__slide)

        # Loading the current saved value for score treshold
        with open(GUI_SETTINGS, "r") as f_obj:
            slider.set(json.load(f_obj)["score_treshold"])

        # Putting the widget on screen
        slider.grid(row=3, column=0, pady=10)

    def __slide(self, value):
        # Overwriting the settings json file
        with open(GUI_SETTINGS, "r+") as f_obj:
            settings_dict = json.load(f_obj)
            f_obj.seek(0)
            f_obj.truncate()
            
            settings_dict["score_treshold"] = value
            json.dump(settings_dict, f_obj)
